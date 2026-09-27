import os
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse

from pytubefix import YouTube
from pytubefix.contrib.channel import Channel

YT_DOWNLOAD_DIR_PREFIX = "whisper_webui_yt_"


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
    return Channel(yt.channel_url)


def get_ytmetas(link):
    try:
        if is_channel_link(link):
            channel = get_ytchannel(link)
            return channel.thumbnail_url, channel.channel_name, channel.description

        yt = get_ytdata(link)
        return yt.thumbnail_url, yt.title, yt.description
    except Exception:
        return None, "", ""


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
    source_path = ytdata.streams.get_audio_only().download(
        output_path=download_dir,
        filename="yt_source",
        skip_existing=False,
    )
    audio_path = os.path.join(download_dir, "yt_tmp.wav")

    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-i', source_path,
            audio_path
        ], check=True)

        os.remove(source_path)
        return audio_path
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg conversion: {e}")
        remove_ytaudio(audio_path)
        return None


def remove_ytaudio(audio_path):
    """Delete audio returned by get_ytaudio, together with its temporary download folder."""
    if not audio_path:
        return
    download_dir = os.path.dirname(os.path.abspath(audio_path))
    if os.path.basename(download_dir).startswith(YT_DOWNLOAD_DIR_PREFIX):
        shutil.rmtree(download_dir, ignore_errors=True)
    elif os.path.exists(audio_path):
        os.remove(audio_path)
