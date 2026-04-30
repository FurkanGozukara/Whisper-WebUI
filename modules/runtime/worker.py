from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict


_JSON_STDOUT = sys.stdout


def emit(event: str, payload: Any = None, **extra: Any) -> None:
    message = {"event": event}
    if payload is not None:
        message["payload"] = payload
    message.update(extra)
    _JSON_STDOUT.write(json.dumps(message, ensure_ascii=True) + "\n")
    _JSON_STDOUT.flush()


def read_request(request_path: str) -> Dict[str, Any]:
    return json.loads(Path(request_path).read_text(encoding="utf-8"))


def build_args_namespace(raw_args: Dict[str, Any]) -> Namespace:
    return Namespace(**raw_args)


def create_whisper_inferencer(args: Namespace):
    from modules.whisper.whisper_factory import WhisperFactory
    from modules.utils.paths import CANARY_QWEN_MODELS_DIR

    return WhisperFactory.create_whisper_inference(
        whisper_type=args.whisper_type,
        whisper_model_dir=args.whisper_model_dir,
        faster_whisper_model_dir=args.faster_whisper_model_dir,
        insanely_fast_whisper_model_dir=args.insanely_fast_whisper_model_dir,
        canary_qwen_model_dir=getattr(args, "canary_qwen_model_dir", None) or CANARY_QWEN_MODELS_DIR,
        diarization_model_dir=args.diarization_model_dir,
        uvr_model_dir=args.uvr_model_dir,
        output_dir=args.output_dir,
    )


def selected_whisper_type(request: Dict[str, Any]) -> str:
    from modules.whisper.data_classes import TranscriptionPipelineParams, WhisperImpl

    explicit_type = request.get("whisper_type")
    if explicit_type:
        return str(explicit_type)

    pipeline_params = request.get("pipeline_params")
    if pipeline_params:
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            return params.whisper.whisper_type
        except Exception:
            pass

    return str(request.get("args", {}).get("whisper_type") or WhisperImpl.FASTER_WHISPER.value)


def args_for_request(request: Dict[str, Any]) -> Namespace:
    args = build_args_namespace(request["args"])
    args.whisper_type = selected_whisper_type(request)
    return args


def whisper_metadata_payload(whisper_inf, gpu_total_memory_gb=None, gpu_name=None) -> Dict[str, Any]:
    return {
        "device": whisper_inf.device,
        "available_models": list(whisper_inf.available_models),
        "available_langs": list(whisper_inf.available_langs),
        "available_compute_types": list(whisper_inf.available_compute_types),
        "current_compute_type": whisper_inf.current_compute_type,
        "gpu_total_memory_gb": gpu_total_memory_gb,
        "gpu_name": gpu_name,
        "music_separator": {
            "device": whisper_inf.music_separator.device,
            "available_devices": list(whisper_inf.music_separator.available_devices),
            "available_models": list(whisper_inf.music_separator.available_models),
        },
        "diarizer": {
            "device": whisper_inf.diarizer.device,
            "available_device": list(whisper_inf.diarizer.available_device),
        },
    }


def query_metadata(request: Dict[str, Any]) -> Dict[str, Any]:
    args = build_args_namespace(request["args"])

    gpu_total_memory_gb = None
    gpu_name = None
    try:
        import torch

        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(device_index)
            gpu_total_memory_gb = device_properties.total_memory / (1024 ** 3)
            gpu_name = getattr(device_properties, "name", None)
        else:
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                device_index = xpu.current_device() if hasattr(xpu, "current_device") else 0
                properties = xpu.get_device_properties(device_index)
                total_memory = getattr(properties, "total_memory", None)
                if total_memory:
                    gpu_total_memory_gb = total_memory / (1024 ** 3)
                gpu_name = getattr(properties, "name", None)
    except Exception:
        gpu_total_memory_gb = None
        gpu_name = None

    from modules.whisper.data_classes import WhisperImpl

    implementation_metadata = {}
    for whisper_type in (WhisperImpl.FASTER_WHISPER.value, WhisperImpl.CANARY_QWEN.value):
        typed_args = Namespace(**vars(args))
        typed_args.whisper_type = whisper_type
        implementation_metadata[whisper_type] = whisper_metadata_payload(
            create_whisper_inferencer(typed_args),
            gpu_total_memory_gb=gpu_total_memory_gb,
            gpu_name=gpu_name,
        )

    selected_type = args.whisper_type if args.whisper_type in implementation_metadata else WhisperImpl.FASTER_WHISPER.value
    whisper_payload = dict(implementation_metadata[selected_type])
    whisper_payload["implementations"] = implementation_metadata

    from modules.translation.nllb_inference import NLLBInference

    nllb_inf = NLLBInference(
        model_dir=args.nllb_model_dir,
        output_dir=f"{args.output_dir}/translations",
    )

    return {
        "whisper": whisper_payload,
        "nllb": {
            "available_models": list(nllb_inf.available_models),
            "available_source_langs": list(nllb_inf.available_source_langs),
            "available_target_langs": list(nllb_inf.available_target_langs),
        },
    }


def transcribe_file_stream(request: Dict[str, Any]) -> None:
    import gradio as gr

    args = args_for_request(request)
    whisper_inf = create_whisper_inferencer(args)

    for live_output, result_str, collected_paths in whisper_inf.transcribe_file_with_live_output(
        request.get("files"),
        request.get("batch_mode", False),
        request.get("input_folder_path"),
        request.get("include_subdirectory"),
        request.get("overwrite_existing", False),
        request.get("output_dir"),
        request.get("file_formats", "SRT"),
        request.get("add_timestamp", True),
        gr.Progress(),
        *request.get("pipeline_params", []),
    ):
        emit(
            "update",
            {
                "live_output": live_output,
                "result_str": result_str,
                "paths": collected_paths,
            },
        )

    emit("complete")


def transcribe_mic_stream(request: Dict[str, Any]) -> None:
    import gradio as gr

    args = args_for_request(request)
    whisper_inf = create_whisper_inferencer(args)

    for live_output, result_str, collected_paths in whisper_inf.transcribe_mic_with_live_output(
        request.get("mic_audio"),
        request.get("file_format", "SRT"),
        request.get("add_timestamp", True),
        gr.Progress(),
        *request.get("pipeline_params", []),
    ):
        emit(
            "update",
            {
                "live_output": live_output,
                "result_str": result_str,
                "paths": collected_paths,
            },
        )

    emit("complete")


def transcribe_youtube_result(request: Dict[str, Any]) -> Any:
    import gradio as gr

    args = args_for_request(request)
    whisper_inf = create_whisper_inferencer(args)
    return whisper_inf.transcribe_youtube(
        request["youtube_link"],
        request.get("file_format", "SRT"),
        request.get("add_timestamp", True),
        request.get("mass_transcribe_channel", False),
        request.get("latest_video_count", 100),
        gr.Progress(),
        *request.get("pipeline_params", []),
    )


def transcribe_mic_result(request: Dict[str, Any]) -> Any:
    import gradio as gr

    args = args_for_request(request)
    whisper_inf = create_whisper_inferencer(args)
    return whisper_inf.transcribe_mic(
        request["mic_audio"],
        request.get("file_format", "SRT"),
        request.get("add_timestamp", True),
        gr.Progress(),
        *request.get("pipeline_params", []),
    )


def separate_bgm_result(request: Dict[str, Any]) -> Any:
    import gradio as gr

    args = build_args_namespace(request["args"])
    whisper_inf = create_whisper_inferencer(args)
    return whisper_inf.music_separator.separate_files(
        files=request["files"],
        model_name=request["model_name"],
        device=request.get("device"),
        segment_size=request.get("segment_size", 256),
        save_file=request.get("save_file", True),
        progress=gr.Progress(),
    )


def translate_nllb_result(request: Dict[str, Any]) -> Any:
    import gradio as gr

    args = build_args_namespace(request["args"])
    from modules.translation.nllb_inference import NLLBInference

    nllb_inf = NLLBInference(
        model_dir=args.nllb_model_dir,
        output_dir=f"{args.output_dir}/translations",
    )
    return nllb_inf.translate_file(
        request["fileobjs"],
        request["model_size"],
        request["src_lang"],
        request["tgt_lang"],
        request.get("max_length", 200),
        request.get("add_timestamp", True),
        gr.Progress(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=[
        "metadata",
        "transcribe_file",
        "transcribe_mic_stream",
        "transcribe_youtube",
        "transcribe_mic",
        "separate_bgm",
        "translate_nllb",
    ])
    parser.add_argument("--request", required=True)
    parsed = parser.parse_args()

    request = read_request(parsed.request)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            if parsed.action == "metadata":
                emit("result", query_metadata(request))
            elif parsed.action == "transcribe_file":
                transcribe_file_stream(request)
            elif parsed.action == "transcribe_mic_stream":
                transcribe_mic_stream(request)
            elif parsed.action == "transcribe_youtube":
                emit("result", transcribe_youtube_result(request))
            elif parsed.action == "transcribe_mic":
                emit("result", transcribe_mic_result(request))
            elif parsed.action == "separate_bgm":
                emit("result", separate_bgm_result(request))
            elif parsed.action == "translate_nllb":
                emit("result", translate_nllb_result(request))
        return 0
    except Exception as exc:
        emit(
            "error",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
