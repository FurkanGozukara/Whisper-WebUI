import time
from pathlib import Path

from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import Segment, TranscriptionPipelineParams


class DummyLivePipeline(BaseTranscriptionPipeline):
    def __init__(self, output_dir: Path):
        self.output_dir = str(output_dir)
        self.model = None
        self.device = "cpu"

    def transcribe(self, audio, progress=None, progress_callback=None, *whisper_params):
        return [], 0.0

    def update_model(self, model_size, compute_type, progress=None):
        return None

    def run(self, audio, progress=None, file_format="SRT", add_timestamp=True, progress_callback=None, *pipeline_params):
        segments = []
        for index in range(35):
            segment = Segment(
                start=float(index),
                end=float(index) + 0.5,
                text=f"line {index + 1}",
            )
            segments.append(segment)
            if progress_callback is not None:
                progress_callback((index + 1) / 35.0, segment)
            time.sleep(0.005)
        return segments, 1.0


def test_live_transcription_streams_recent_segment_history(monkeypatch, tmp_path):
    media_file = tmp_path / "clip.wav"
    media_file.write_bytes(b"fake")

    output_file = tmp_path / "clip.srt"
    output_file.write_text("subtitle", encoding="utf-8")

    monkeypatch.setattr(
        "modules.whisper.base_transcription_pipeline.generate_file",
        lambda **kwargs: ("subtitle", str(output_file)),
    )

    pipeline = DummyLivePipeline(output_dir=tmp_path)
    pipeline_params = TranscriptionPipelineParams().to_list()

    updates = list(
        pipeline.transcribe_file_with_live_output(
            [str(media_file)],
            False,
            None,
            False,
            False,
            str(tmp_path),
            ["SRT"],
            False,
            None,
            *pipeline_params,
        )
    )

    assert len(updates) > 2

    first_live_output, first_result, _ = updates[0]
    final_live_output, final_result, final_paths = updates[-1]

    assert "Processing: clip" in first_live_output
    assert first_result == ""
    assert "Done! 35 segments" in final_result
    assert final_paths == [str(output_file)]
    assert "line 35" in final_live_output
    assert "[00:00:00.000 → 00:00:00.500] line 1" not in final_live_output
    assert len(final_live_output.splitlines()) <= BaseTranscriptionPipeline.LIVE_TRANSCRIPTION_HISTORY_LINES
