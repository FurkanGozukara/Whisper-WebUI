import os
import subprocess
from urllib.parse import urlparse

from pytubefix import YouTube
from pytubefix.contrib.channel import Channel


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

    audio_path = ytdata.streams.get_audio_only().download(filename=os.path.join("modules", "yt_tmp.wav"))
    temp_audio_path = os.path.join("modules", "yt_tmp_fixed.wav")

    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-i', audio_path,
            temp_audio_path
        ], check=True)

        os.replace(temp_audio_path, audio_path)
        return audio_path
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg conversion: {e}")
        return None
