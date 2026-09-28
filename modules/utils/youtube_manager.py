import os
import shutil
import subprocess
import tempfile
import logging
import socket
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from urllib.parse import urlparse

from pytubefix import YouTube
from pytubefix import request as pytube_request
from pytubefix.contrib.channel import Channel

YT_DOWNLOAD_DIR_PREFIX = "whisper_webui_yt_"
YOUTUBE_REQUEST_TIMEOUT = 20.0
YOUTUBE_METADATA_TIMEOUT = 15.0
_REQUEST_DEADLINE = ContextVar("youtube_request_deadline", default=None)
logger = logging.getLogger(__name__)


def _install_request_timeout():
    """Bound pytubefix's otherwise unlimited socket waits without changing global socket defaults."""
    original = pytube_request._execute_request
    if getattr(original, "_whisperwebui_timeout", False):
        return

    @wraps(original)
    def execute(url, method=None, headers=None, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
        limit = YOUTUBE_REQUEST_TIMEOUT
        deadline = _REQUEST_DEADLINE.get()
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("YouTube metadata request timed out. Check the connection or try again later.")
            limit = min(limit, remaining)
        if timeout is not None and timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
            limit = min(limit, float(timeout))
        return original(url, method=method, headers=headers, data=data, timeout=limit)

    execute._whisperwebui_timeout = True
    pytube_request._execute_request = execute


_install_request_timeout()


@contextmanager
def _metadata_deadline():
    token = _REQUEST_DEADLINE.set(time.monotonic() + YOUTUBE_METADATA_TIMEOUT)
    try:
        yield
    finally:
        _REQUEST_DEADLINE.reset(token)


def get_ytdata(link):
    return YouTube(link)


def is_channel_link(link: str) -> bool:
    normalized = str(link or "").strip()
    if not normalized:
        return False

    try:
        parsed = urlparse(normalized)
    except ValueError:
        return False

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").rstrip("/")

    if "youtu.be" in host:
        return False
    if path.startswith("/watch") or path.startswith("/shorts/") or path.startswith("/live/"):
        return False
    if path.startswith("/playlist") and "list=" in (parsed.query or ""):
        return False

    return any(path.startswith(prefix) for prefix in ("/@", "/channel/", "/c/", "/user/"))


def get_ytchannel(link):
    normalized = str(link or "").strip()
    if is_channel_link(normalized):
        return Channel(normalized)

    yt = get_ytdata(normalized)
    if not yt.channel_id:
        # On blocked/unavailable videos pytubefix can return channel_id=None and
        # fabricate /channel/None, hiding the real cause behind an unrelated 404.
        yt.check_availability()
        raise ValueError("Could not identify this video's channel. Try its direct YouTube channel URL.")
    return Channel(yt.channel_url)


def get_ytmetas(link):
    if not str(link or "").strip():
        return None, "", ""
    try:
        with _metadata_deadline():
            if is_channel_link(link):
                channel = get_ytchannel(link)
                return channel.thumbnail_url, channel.channel_name, channel.description

            yt = get_ytdata(link)
            return yt.thumbnail_url, yt.title, yt.description
    except Exception as exc:
        message = f"Could not load YouTube details: {type(exc).__name__}: {exc}"
        logger.warning(message)
        return None, "", message


def get_latest_channel_videos(link, limit: int = 100):
    channel = get_ytchannel(link)
    safe_limit = max(1, min(9999, int(limit or 100)))
    videos = []

    for index, video in enumerate(channel.videos, start=1):
        videos.append(video)
        if index >= safe_limit:
            break

    return videos


def get_ytaudio(ytdata: YouTube):
    # Somehow the audio is corrupted so need to convert to valid audio file.
    # Fix for : https://github.com/jhj0517/Whisper-WebUI/issues/304
    # pytubefix strips path separators from `filename` (modules/yt_tmp.wav became modulesyt_tmp.wav in the
    # working folder), so every download gets its own temporary folder; that also keeps two jobs, or a file
    # left behind by a cancelled job, from sharing one audio file.
    download_dir = tempfile.mkdtemp(prefix=YT_DOWNLOAD_DIR_PREFIX)
    audio_path = os.path.join(download_dir, "yt_tmp.wav")

    try:
        stream = ytdata.streams.get_audio_only()
        if stream is None:
            raise RuntimeError("This YouTube video does not offer an audio stream.")
        source_path = stream.download(
            output_path=download_dir,
            filename="yt_source",
            skip_existing=False,
            timeout=YOUTUBE_REQUEST_TIMEOUT,
            max_retries=1,
        )
        # -hide_banner/-loglevel error: without them ffmpeg prints about 50 lines of build info into CMD
        subprocess.run([
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', source_path,
            audio_path
        ], check=True)

        os.remove(source_path)
        return audio_path
    except BaseException:
        # Includes download/access failures and cancellation before conversion starts.
        remove_ytaudio(audio_path)
        raise


def remove_ytaudio(audio_path):
    """Delete audio returned by get_ytaudio, together with its temporary download folder."""
    if not audio_path:
        return
    download_dir = os.path.dirname(os.path.abspath(audio_path))
    if os.path.basename(download_dir).startswith(YT_DOWNLOAD_DIR_PREFIX):
        shutil.rmtree(download_dir, ignore_errors=True)
    elif os.path.exists(audio_path):
        os.remove(audio_path)
