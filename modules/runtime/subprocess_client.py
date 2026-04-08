from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from argparse import Namespace
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Deque, Dict, Generator, Optional, Tuple

from modules.utils.text import repair_mojibake_obj
from modules.whisper.data_classes import TranscriptionPipelineParams


@dataclass
class WorkerHandle:
    process: subprocess.Popen
    request_path: str
    stderr_lines: Deque[str]
    stderr_thread: Thread


class RuntimeWorkerClient:
    def __init__(self, args: Namespace):
        self.args_dict = dict(vars(args))

    def _write_request(self, action_payload: Dict[str, Any]) -> str:
        request = {
            "args": self.args_dict,
            **action_payload,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="whisper_webui_worker_",
            delete=False,
            encoding="utf-8",
        ) as temp_file:
            json.dump(request, temp_file, ensure_ascii=False)
            return temp_file.name

    def start_worker(self, action: str, action_payload: Dict[str, Any]) -> WorkerHandle:
        request_path = self._write_request(action_payload)
        repo_root = str(Path(__file__).resolve().parents[2])
        command = [
            sys.executable,
            "-u",
            "-m",
            "modules.runtime.worker",
            action,
            "--request",
            request_path,
        ]

        popen_kwargs: Dict[str, Any] = {
            "cwd": repo_root,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "bufsize": 0,
        }

        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["preexec_fn"] = os.setsid

        process = subprocess.Popen(command, **popen_kwargs)
        stderr_lines: Deque[str] = deque(maxlen=200)

        def drain_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                decoded_line = line.decode("utf-8", errors="replace").rstrip()
                stderr_lines.append(decoded_line)
                try:
                    sys.stderr.write(decoded_line + "\n")
                    sys.stderr.flush()
                except Exception:
                    pass

        stderr_thread = Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()
        return WorkerHandle(process=process, request_path=request_path, stderr_lines=stderr_lines, stderr_thread=stderr_thread)

    @staticmethod
    def _cleanup_request_file(request_path: str) -> None:
        try:
            os.remove(request_path)
        except OSError:
            pass

    def finalize_worker(self, handle: WorkerHandle) -> None:
        try:
            if handle.process.poll() is None:
                handle.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        handle.stderr_thread.join(timeout=1)
        self._cleanup_request_file(handle.request_path)

    def terminate_worker(self, handle: WorkerHandle) -> None:
        process = handle.process
        if process.poll() is not None:
            return

        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                pass

    def iter_events(self, handle: WorkerHandle) -> Generator[Dict[str, Any], None, None]:
        assert handle.process.stdout is not None
        for raw_line in handle.process.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    def call(self, action: str, action_payload: Dict[str, Any]) -> Any:
        handle = self.start_worker(action, action_payload)
        result_payload = None
        error_payload = None

        try:
            for event in self.iter_events(handle):
                if event["event"] == "result":
                    result_payload = event.get("payload")
                elif event["event"] == "error":
                    error_payload = event.get("payload") or {}
        finally:
            return_code = handle.process.wait()
            self.finalize_worker(handle)

        if error_payload:
            raise RuntimeError(self._format_error(error_payload, handle.stderr_lines))
        if return_code != 0:
            raise RuntimeError(self._format_error({}, handle.stderr_lines))
        return repair_mojibake_obj(result_payload)

    @staticmethod
    def _format_error(error_payload: Dict[str, Any], stderr_lines: Deque[str]) -> str:
        message = error_payload.get("error") or "Worker process failed."
        stderr_tail = "\n".join(stderr_lines).strip()
        if stderr_tail:
            return f"{message}\n{stderr_tail}"
        return message


class _DiarizerProxy:
    def __init__(self, metadata: Dict[str, Any]):
        self.device = metadata["device"]
        self.available_device = metadata["available_device"]


class _MusicSeparatorProxy:
    def __init__(self, client: RuntimeWorkerClient, metadata: Dict[str, Any]):
        self._client = client
        self.device = metadata["device"]
        self.available_devices = metadata["available_devices"]
        self.available_models = metadata["available_models"]

    def separate_files(
        self,
        files,
        model_name: str,
        device: Optional[str] = None,
        segment_size: int = 256,
        save_file: bool = True,
        progress=None,
    ):
        return self._client.call(
            "separate_bgm",
            {
                "files": files,
                "model_name": model_name,
                "device": device,
                "segment_size": segment_size,
                "save_file": save_file,
            },
        )


class SubprocessWhisperProxy:
    def __init__(self, args: Namespace, metadata: Dict[str, Any]):
        self._args = args
        self._client = RuntimeWorkerClient(args)
        self._active_lock = Lock()
        self._active_handle: Optional[WorkerHandle] = None
        self._cancelled_pids: set[int] = set()
        self._local_inferencer = None

        self.device = metadata["device"]
        self.available_models = metadata["available_models"]
        self.available_langs = metadata["available_langs"]
        self.available_compute_types = metadata["available_compute_types"]
        self.current_compute_type = metadata["current_compute_type"]
        self.gpu_total_memory_gb = metadata.get("gpu_total_memory_gb")
        self.gpu_name = metadata.get("gpu_name")
        self.music_separator = _MusicSeparatorProxy(self._client, metadata["music_separator"])
        self.diarizer = _DiarizerProxy(metadata["diarizer"])

    def _get_local_inferencer(self):
        if self._local_inferencer is None:
            from modules.whisper.whisper_factory import WhisperFactory

            self._local_inferencer = WhisperFactory.create_whisper_inference(
                whisper_type=self._args.whisper_type,
                whisper_model_dir=self._args.whisper_model_dir,
                faster_whisper_model_dir=self._args.faster_whisper_model_dir,
                insanely_fast_whisper_model_dir=self._args.insanely_fast_whisper_model_dir,
                diarization_model_dir=self._args.diarization_model_dir,
                uvr_model_dir=self._args.uvr_model_dir,
                output_dir=self._args.output_dir,
            )
        return self._local_inferencer

    @staticmethod
    def _use_subprocess(pipeline_params) -> bool:
        params = TranscriptionPipelineParams.from_list(list(pipeline_params))
        return bool(getattr(params.whisper, "start_as_subprocess", True))

    def cancel_active_generation(self) -> bool:
        with self._active_lock:
            handle = self._active_handle
            if handle is None or handle.process.poll() is not None:
                return False
            self._cancelled_pids.add(handle.process.pid)

        self._client.terminate_worker(handle)
        return True

    def _stream_transcription_worker(self, action_payload: Dict[str, Any]):
        handle = self._client.start_worker("transcribe_file", action_payload)
        last_live_output = ""
        last_result = ""
        last_paths = []
        worker_error = None

        with self._active_lock:
            self._active_handle = handle

        try:
            for event in self._client.iter_events(handle):
                if event["event"] == "update":
                    payload = repair_mojibake_obj(event.get("payload") or {})
                    last_live_output = payload.get("live_output", last_live_output)
                    last_result = payload.get("result_str", last_result)
                    last_paths = payload.get("paths", last_paths)
                    yield last_live_output, last_result, last_paths
                elif event["event"] == "error":
                    worker_error = event.get("payload") or {}
        finally:
            return_code = handle.process.wait()
            with self._active_lock:
                if self._active_handle is handle:
                    self._active_handle = None
            self._client.finalize_worker(handle)

        if handle.process.pid in self._cancelled_pids:
            self._cancelled_pids.discard(handle.process.pid)
            yield last_live_output, "Cancelled. Running subprocess was terminated.", last_paths
            return

        if worker_error:
            raise RuntimeError(self._client._format_error(worker_error, handle.stderr_lines))
        if return_code != 0:
            raise RuntimeError(self._client._format_error({}, handle.stderr_lines))

    def transcribe_file_with_live_output(
        self,
        files=None,
        batch_mode=False,
        input_folder_path=None,
        include_subdirectory=None,
        overwrite_existing=False,
        output_dir=None,
        file_formats="SRT",
        add_timestamp=True,
        progress=None,
        *pipeline_params,
    ):
        if not self._use_subprocess(pipeline_params):
            yield from self._get_local_inferencer().transcribe_file_with_live_output(
                files,
                batch_mode,
                input_folder_path,
                include_subdirectory,
                overwrite_existing,
                output_dir,
                file_formats,
                add_timestamp,
                progress,
                *pipeline_params,
            )
            return

        yield from self._stream_transcription_worker(
            {
                "files": files,
                "batch_mode": batch_mode,
                "input_folder_path": input_folder_path,
                "include_subdirectory": include_subdirectory,
                "overwrite_existing": overwrite_existing,
                "output_dir": output_dir,
                "file_formats": file_formats,
                "add_timestamp": add_timestamp,
                "pipeline_params": list(pipeline_params),
            }
        )

    def transcribe_youtube(
        self,
        youtube_link: str,
        file_format="SRT",
        add_timestamp=True,
        progress=None,
        *pipeline_params,
    ):
        if not self._use_subprocess(pipeline_params):
            return self._get_local_inferencer().transcribe_youtube(
                youtube_link,
                file_format,
                add_timestamp,
                progress,
                *pipeline_params,
            )

        return self._client.call(
            "transcribe_youtube",
            {
                "youtube_link": youtube_link,
                "file_format": file_format,
                "add_timestamp": add_timestamp,
                "pipeline_params": list(pipeline_params),
            },
        )

    def transcribe_mic(
        self,
        mic_audio: str,
        file_format="SRT",
        add_timestamp=True,
        progress=None,
        *pipeline_params,
    ):
        if not self._use_subprocess(pipeline_params):
            return self._get_local_inferencer().transcribe_mic(
                mic_audio,
                file_format,
                add_timestamp,
                progress,
                *pipeline_params,
            )

        return self._client.call(
            "transcribe_mic",
            {
                "mic_audio": mic_audio,
                "file_format": file_format,
                "add_timestamp": add_timestamp,
                "pipeline_params": list(pipeline_params),
            },
        )


class SubprocessNLLBProxy:
    def __init__(self, args: Namespace, metadata: Dict[str, Any]):
        self._client = RuntimeWorkerClient(args)
        self.available_models = metadata["available_models"]
        self.available_source_langs = metadata["available_source_langs"]
        self.available_target_langs = metadata["available_target_langs"]

    def translate_file(
        self,
        fileobjs,
        model_size: str,
        src_lang: str,
        tgt_lang: str,
        max_length: int = 200,
        add_timestamp: bool = True,
        progress=None,
    ):
        return self._client.call(
            "translate_nllb",
            {
                "fileobjs": fileobjs,
                "model_size": model_size,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "max_length": max_length,
                "add_timestamp": add_timestamp,
            },
        )


def load_runtime_metadata(args: Namespace) -> Dict[str, Any]:
    client = RuntimeWorkerClient(args)
    return client.call("metadata", {})


def build_runtime_proxies(args: Namespace) -> Tuple[SubprocessWhisperProxy, SubprocessNLLBProxy]:
    metadata = repair_mojibake_obj(load_runtime_metadata(args))
    whisper_proxy = SubprocessWhisperProxy(args, metadata["whisper"])
    nllb_proxy = SubprocessNLLBProxy(args, metadata["nllb"])
    return whisper_proxy, nllb_proxy
