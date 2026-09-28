from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from modules.utils import youtube_manager as youtube


def test_youtube_requests_use_bounded_network_timeouts(monkeypatch):
    calls = []
    monkeypatch.setattr(youtube.pytube_request, "_execute_request", lambda url, **kwargs: calls.append(kwargs))
    youtube._install_request_timeout()
    youtube.pytube_request._execute_request("https://example.invalid/")
    youtube.pytube_request._execute_request("https://example.invalid/", timeout=3)
    assert [call["timeout"] for call in calls] == [youtube.YOUTUBE_REQUEST_TIMEOUT, 3]


def test_metadata_deadline_caps_each_request_and_expires(monkeypatch):
    calls = []
    clock = [100.0]
    monkeypatch.setattr(youtube.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(youtube.pytube_request, "_execute_request", lambda url, **kwargs: calls.append(kwargs))
    youtube._install_request_timeout()
    with youtube._metadata_deadline():
        clock[0] = 110.0
        youtube.pytube_request._execute_request("https://example.invalid/")
        assert calls[0]["timeout"] == 5.0
        clock[0] = 116.0
        with pytest.raises(TimeoutError, match="timed out"):
            youtube.pytube_request._execute_request("https://example.invalid/")
    assert youtube._REQUEST_DEADLINE.get() is None


def test_metadata_error_is_visible_instead_of_silently_blank(monkeypatch):
    def fail(_link):
        raise TimeoutError("server did not respond")
    monkeypatch.setattr(youtube, "get_ytdata", fail)
    thumbnail, title, description = youtube.get_ytmetas("https://www.youtube.com/watch?v=example")
    assert thumbnail is None and not title
    assert "Could not load YouTube details" in description
    assert "server did not respond" in description


def test_missing_channel_preserves_video_access_error_without_requesting_channel_none(monkeypatch):
    from pytubefix.exceptions import BotDetection

    def blocked():
        raise BotDetection("test-video")

    video = SimpleNamespace(channel_id=None, channel_url="https://www.youtube.com/channel/None",
                            check_availability=blocked)
    monkeypatch.setattr(youtube, "get_ytdata", lambda _: video)
    monkeypatch.setattr(youtube, "Channel", lambda _: pytest.fail("Must not request /channel/None"))
    with pytest.raises(BotDetection):
        youtube.get_ytchannel("https://www.youtube.com/watch?v=test-video")


def test_missing_channel_with_available_video_has_actionable_error(monkeypatch):
    video = SimpleNamespace(channel_id=None, check_availability=lambda: None)
    monkeypatch.setattr(youtube, "get_ytdata", lambda _: video)
    monkeypatch.setattr(youtube, "Channel", lambda _: pytest.fail("Must not request a missing channel"))
    with pytest.raises(ValueError, match="direct YouTube channel URL"):
        youtube.get_ytchannel("https://www.youtube.com/watch?v=test-video")


def test_video_resolves_to_its_valid_channel_and_direct_channel_needs_no_video_lookup(monkeypatch):
    expected = "https://www.youtube.com/channel/UCvalid-channel"
    calls = []
    monkeypatch.setattr(youtube, "get_ytdata", lambda link: calls.append(link) or
                        SimpleNamespace(channel_id="UCvalid-channel", channel_url=expected))
    monkeypatch.setattr(youtube, "Channel", lambda link: link)
    video = "https://www.youtube.com/watch?v=test-video"
    assert youtube.get_ytchannel(video) == expected
    assert youtube.get_ytchannel(expected) == expected
    assert calls == [video]


@pytest.mark.parametrize("failure_phase", ["streams", "download", "conversion"])
def test_failed_youtube_download_removes_its_temporary_directory(tmp_path, monkeypatch, failure_phase):
    folder = tmp_path / (youtube.YT_DOWNLOAD_DIR_PREFIX + "failure")
    folder.mkdir()
    monkeypatch.setattr(youtube.tempfile, "mkdtemp", lambda **kwargs: str(folder))

    class Video:
        @property
        def streams(self):
            if failure_phase == "streams":
                raise RuntimeError("access denied")
            return SimpleNamespace(get_audio_only=lambda: self)

        def download(self, **kwargs):
            source = Path(kwargs["output_path"]) / kwargs["filename"]
            source.write_bytes(b"partial audio")
            if failure_phase == "download":
                raise TimeoutError("download stalled")
            return str(source)

    def fail_conversion(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg")
    monkeypatch.setattr(youtube.subprocess, "run", fail_conversion)

    with pytest.raises((RuntimeError, TimeoutError, subprocess.CalledProcessError)):
        youtube.get_ytaudio(Video())
    assert not folder.exists()
