from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from modules.whisper.data_classes import Segment, WhisperParams
from modules.whisper.faster_whisper_inference import FasterWhisperInference
from modules.utils.logger import get_logger


logger = get_logger()


def read_request(request_path: str) -> dict:
    return json.loads(Path(request_path).read_text(encoding="utf-8"))


def emit(event: str, payload: dict) -> None:
    sys.stdout.write(json.dumps({"event": event, "payload": payload}, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def release_temp_model(model, device: str) -> None:
    if model is not None:
        del model

    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        elif device == "xpu":
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                xpu.empty_cache()
                xpu.reset_accumulated_memory_stats()
                xpu.reset_peak_memory_stats()
    except Exception:
        pass


def run_slice(request: dict) -> None:
    import faster_whisper

    params = WhisperParams(**request["params"])
    audio = np.load(request["audio_path"], allow_pickle=False)
    model = None

    try:
        logger.info(
            "Loading model: Base Model=Whisper (faster-whisper / CTranslate2), "
            "selected=%s, resolved=%s, device=%s, compute_type=%s, slice=P%s",
            params.model_size,
            request["model_size_or_path"],
            request["device"],
            params.compute_type,
            int(request["slice_index"]) + 1,
        )
        model = faster_whisper.WhisperModel(
            device=request["device"],
            model_size_or_path=request["model_size_or_path"],
            download_root=request["model_dir"],
            compute_type=params.compute_type,
            local_files_only=bool(request.get("local_files_only", False)),
        )
        logger.info(
            "Model loaded: Base Model=Whisper (faster-whisper / CTranslate2), "
            "selected=%s, active=%s, device=%s, compute_type=%s, slice=P%s",
            params.model_size,
            request["model_size_or_path"],
            request["device"],
            params.compute_type,
            int(request["slice_index"]) + 1,
        )
        segments, info = FasterWhisperInference.run_standard_pipeline_with_model(
            model=model,
            audio=audio,
            params=params,
        )
        segment_payloads = []
        for segment in segments:
            segment_payload = Segment.from_faster_whisper(segment).model_dump()
            segment_payloads.append(segment_payload)
            emit(
                "segment",
                {
                    "slice_index": request["slice_index"],
                    "segment": segment_payload,
                },
            )
        result_payload = {
            "slice_index": request["slice_index"],
            "offset_seconds": request.get("offset_seconds", 0.0),
            "duration": getattr(info, "duration", None),
            "segments": segment_payloads,
        }
        Path(request["result_path"]).write_text(
            json.dumps(result_payload, ensure_ascii=False),
            encoding="utf-8",
        )
    finally:
        release_temp_model(model, request["device"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = read_request(args.request)
    run_slice(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
