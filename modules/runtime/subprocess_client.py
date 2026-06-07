from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import gradio as gr
from argparse import Namespace
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Deque, Dict, Generator, Optional, Tuple

from modules.utils.logger import get_logger
from modules.utils.text import repair_mojibake_obj
from modules.whisper.data_classes import TranscriptionPipelineParams, WhisperImpl


logger = get_logger()


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
        for pipe in (handle.process.stdout, handle.process.stderr):
            if pipe is None:
                continue
            try:
                pipe.close()
            except OSError:
                pass
        handle.stderr_thread.join(timeout=1)
        self._cleanup_request_file(handle.request_path)

    @staticmethod
    def _collect_process_tree(process: subprocess.Popen) -> list:
        try:
            import psutil
        except Exception:
            return []

        try:
            parent = psutil.Process(process.pid)
        except Exception:
            return []

        try:
            return parent.children(recursive=True) + [parent]
        except Exception:
            return [parent]

    @staticmethod
    def _kill_known_processes(processes: list, timeout: float = 5.0) -> list[int]:
        if not processes:
            return []

        try:
            import psutil
        except Exception:
            return []

        targets = []
        for proc in processes:
            try:
                if proc.is_running():
                    targets.append(proc)
            except Exception:
                continue

        for proc in targets:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception:
                logger.warning("Failed to kill worker child process pid=%s.", getattr(proc, "pid", "unknown"))

        try:
            _, alive = psutil.wait_procs(targets, timeout=timeout)
        except Exception:
            alive = []

        return [int(proc.pid) for proc in alive if getattr(proc, "pid", None) is not None]

    def terminate_worker(self, handle: WorkerHandle) -> None:
        process = handle.process
        if process.poll() is not None:
            return

        logger.info("Cancellation requested. Force terminating worker process tree pid=%s.", process.pid)
        known_processes = self._collect_process_tree(process)

        if os.name == "nt":
            taskkill_result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if taskkill_result.returncode != 0 and process.poll() is None:
                stderr = taskkill_result.stderr.decode("utf-8", errors="replace").strip()
                logger.warning("taskkill failed for worker pid=%s: %s", process.pid, stderr or taskkill_result.returncode)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                alive_pids = self._kill_known_processes(known_processes)
                if alive_pids:
                    logger.warning("Worker process tree still alive after psutil kill: %s", alive_pids)
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=2)
                except Exception:
                    pass
            else:
                alive_pids = self._kill_known_processes(known_processes, timeout=1.0)
                if alive_pids:
                    logger.warning("Worker child process(es) remained after taskkill: %s", alive_pids)
            logger.info("Worker process tree pid=%s terminated with return code %s.", process.pid, process.poll())
            return

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=3)
        except Exception:
            alive_pids = self._kill_known_processes(known_processes)
            if alive_pids:
                logger.warning("Worker process tree still alive after psutil kill: %s", alive_pids)
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                pass
            try:
                process.wait(timeout=2)
            except Exception:
                pass
        else:
            alive_pids = self._kill_known_processes(known_processes, timeout=1.0)
            if alive_pids:
                logger.warning("Worker child process(es) remained after process-group termination: %s", alive_pids)
        logger.info("Worker process tree pid=%s terminated with return code %s.", process.pid, process.poll())

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
            raise RuntimeError(self._format_error(error_payload, handle.stderr_lines, return_code=return_code))
        if return_code != 0:
            raise RuntimeError(self._format_error({}, handle.stderr_lines, return_code=return_code))
        return repair_mojibake_obj(result_payload)

    @staticmethod
    def _format_error(
        error_payload: Dict[str, Any],
        stderr_lines: Deque[str],
        return_code: Optional[int] = None,
    ) -> str:
        message = error_payload.get("error") or "Worker process failed."
        if return_code not in (None, 0):
            message = f"{message} (worker return code: {return_code})"
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
        self._active_handles: Dict[int, WorkerHandle] = {}
        self._cancelled_pids: set[int] = set()
        self._local_inferencers: Dict[str, Any] = {}

        self.implementation_metadata = metadata.get("implementations") or {
            getattr(args, "whisper_type", WhisperImpl.FASTER_WHISPER.value): metadata
        }
        self.device = metadata["device"]
        self.available_models = metadata["available_models"]
        self.available_langs = metadata["available_langs"]
        self.available_compute_types = metadata["available_compute_types"]
        self.current_compute_type = metadata["current_compute_type"]
        self.gpu_total_memory_gb = metadata.get("gpu_total_memory_gb")
        self.gpu_name = metadata.get("gpu_name")
        self.music_separator = _MusicSeparatorProxy(self._client, metadata["music_separator"])
        self.diarizer = _DiarizerProxy(metadata["diarizer"])

    def _get_local_inferencer(self, whisper_type: Optional[str] = None):
        whisper_type = whisper_type or getattr(self._args, "whisper_type", WhisperImpl.FASTER_WHISPER.value)
        if whisper_type not in self._local_inferencers:
            from modules.whisper.whisper_factory import WhisperFactory
            from modules.utils.paths import CANARY_QWEN_MODELS_DIR

            self._local_inferencers[whisper_type] = WhisperFactory.create_whisper_inference(
                whisper_type=whisper_type,
                whisper_model_dir=self._args.whisper_model_dir,
                faster_whisper_model_dir=self._args.faster_whisper_model_dir,
                insanely_fast_whisper_model_dir=self._args.insanely_fast_whisper_model_dir,
                canary_qwen_model_dir=getattr(self._args, "canary_qwen_model_dir", None) or CANARY_QWEN_MODELS_DIR,
                diarization_model_dir=self._args.diarization_model_dir,
                uvr_model_dir=self._args.uvr_model_dir,
                output_dir=self._args.output_dir,
            )
        return self._local_inferencers[whisper_type]

    def _local_inferencer_for(self, whisper_type: Optional[str] = None):
        try:
            return self._get_local_inferencer(whisper_type)
        except TypeError:
            return self._get_local_inferencer()

    @staticmethod
    def _use_subprocess(pipeline_params) -> bool:
        params = TranscriptionPipelineParams.from_list(list(pipeline_params))
        return bool(getattr(params.whisper, "start_as_subprocess", True))

    def _selected_whisper_type(self, pipeline_params) -> str:
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            whisper_type = getattr(params.whisper, "whisper_type", None)
            if whisper_type:
                return whisper_type
        except Exception:
            pass
        return getattr(self._args, "whisper_type", WhisperImpl.FASTER_WHISPER.value)

    @staticmethod
    def _split_progress_and_pipeline_args(extra_args):
        if extra_args and isinstance(extra_args[0], gr.Progress):
            return extra_args[0], list(extra_args[1:])
        return gr.Progress(), list(extra_args)

    def cancel_active_generation(self) -> bool:
        with self._active_lock:
            self._prune_active_handles_locked()
            handles = dict(self._active_handles)
            if (
                self._active_handle is not None
                and self._active_handle.process.poll() is None
                and self._active_handle.process.pid not in handles
            ):
                handles[self._active_handle.process.pid] = self._active_handle

            if not handles:
                return False

            for handle in handles.values():
                self._cancelled_pids.add(handle.process.pid)

        logger.info("Cancellation requested for active worker pid(s): %s", sorted(handles))
        for handle in handles.values():
            self._client.terminate_worker(handle)
        return True

    def _prune_active_handles_locked(self) -> None:
        for pid, handle in list(self._active_handles.items()):
            if handle.process.poll() is not None:
                self._active_handles.pop(pid, None)

        if self._active_handle is not None and self._active_handle.process.poll() is not None:
            self._active_handle = None

    def _set_active_handle(self, handle: WorkerHandle) -> None:
        with self._active_lock:
            self._active_handle = handle
            self._active_handles[handle.process.pid] = handle
        logger.info("Registered active worker pid=%s.", handle.process.pid)

    def _clear_active_handle(self, handle: WorkerHandle) -> None:
        with self._active_lock:
            if self._active_handle is handle:
                self._active_handle = None
            self._active_handles.pop(handle.process.pid, None)
        logger.info("Cleared active worker pid=%s.", handle.process.pid)

    def _terminate_unfinished_worker(self, handle: WorkerHandle) -> None:
        if handle.process.poll() is not None:
            return

        with self._active_lock:
            self._cancelled_pids.add(handle.process.pid)

        self._client.terminate_worker(handle)

    def _wait_and_finalize_worker(self, handle: WorkerHandle) -> int:
        try:
            return_code = handle.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._terminate_unfinished_worker(handle)
            try:
                return_code = handle.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return_code = handle.process.poll()
                if return_code is None:
                    return_code = -1
        finally:
            self._clear_active_handle(handle)
            self._client.finalize_worker(handle)

        return return_code

    def _was_cancelled(self, handle: WorkerHandle) -> bool:
        with self._active_lock:
            if handle.process.pid in self._cancelled_pids:
                self._cancelled_pids.discard(handle.process.pid)
                return True
        return False

    def _call_transcription_worker(self, action: str, action_payload: Dict[str, Any]) -> Any:
        handle = self._client.start_worker(action, action_payload)
        result_payload = None
        error_payload = None
        stream_exhausted = False

        self._set_active_handle(handle)

        try:
            for event in self._client.iter_events(handle):
                if event["event"] == "result":
                    result_payload = event.get("payload")
                elif event["event"] == "error":
                    error_payload = event.get("payload") or {}
            stream_exhausted = True
        finally:
            if not stream_exhausted:
                self._terminate_unfinished_worker(handle)
            return_code = self._wait_and_finalize_worker(handle)

        if self._was_cancelled(handle):
            raise RuntimeError("Cancelled. Running subprocess was terminated.")
        if error_payload:
            raise RuntimeError(self._client._format_error(error_payload, handle.stderr_lines, return_code=return_code))
        if return_code != 0:
            raise RuntimeError(self._client._format_error({}, handle.stderr_lines, return_code=return_code))
        return repair_mojibake_obj(result_payload)

    def _stream_transcription_worker(self, action: str, action_payload: Dict[str, Any]):
        handle = self._client.start_worker(action, action_payload)
        last_live_output = ""
        last_result = ""
        last_paths = []
        worker_error = None
        stream_exhausted = False

        self._set_active_handle(handle)

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
            stream_exhausted = True
        finally:
            if not stream_exhausted:
                self._terminate_unfinished_worker(handle)
            return_code = self._wait_and_finalize_worker(handle)

        if self._was_cancelled(handle):
            yield last_live_output, "Cancelled. Running subprocess was terminated.", last_paths
            return

        if worker_error:
            raise RuntimeError(self._client._format_error(worker_error, handle.stderr_lines, return_code=return_code))
        if return_code != 0:
            raise RuntimeError(self._client._format_error({}, handle.stderr_lines, return_code=return_code))

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
        whisper_type = self._selected_whisper_type(pipeline_params)
        if not self._use_subprocess(pipeline_params):
            yield from self._local_inferencer_for(whisper_type).transcribe_file_with_live_output(
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
            "transcribe_file",
            {
                "files": files,
                "batch_mode": batch_mode,
                "input_folder_path": input_folder_path,
                "include_subdirectory": include_subdirectory,
                "overwrite_existing": overwrite_existing,
                "output_dir": output_dir,
                "file_formats": file_formats,
                "add_timestamp": add_timestamp,
                "whisper_type": whisper_type,
                "pipeline_params": list(pipeline_params),
            }
        )

    def transcribe_youtube(
        self,
        youtube_link: str,
        file_format="SRT",
        add_timestamp=True,
        mass_transcribe_channel=False,
        latest_video_count=100,
        *extra_args,
    ):
        progress, pipeline_params = self._split_progress_and_pipeline_args(extra_args)
        whisper_type = self._selected_whisper_type(pipeline_params)
        if not self._use_subprocess(pipeline_params):
            return self._local_inferencer_for(whisper_type).transcribe_youtube(
                youtube_link,
                file_format,
                add_timestamp,
                mass_transcribe_channel,
                latest_video_count,
                progress,
                *pipeline_params,
            )

        return self._call_transcription_worker(
            "transcribe_youtube",
            {
                "youtube_link": youtube_link,
                "file_format": file_format,
                "add_timestamp": add_timestamp,
                "mass_transcribe_channel": mass_transcribe_channel,
                "latest_video_count": latest_video_count,
                "whisper_type": whisper_type,
                "pipeline_params": list(pipeline_params),
            },
        )

    def transcribe_mic(
        self,
        mic_audio: str,
        file_format="SRT",
        add_timestamp=True,
        *extra_args,
    ):
        progress, pipeline_params = self._split_progress_and_pipeline_args(extra_args)
        whisper_type = self._selected_whisper_type(pipeline_params)
        if not self._use_subprocess(pipeline_params):
            return self._local_inferencer_for(whisper_type).transcribe_mic(
                mic_audio,
                file_format,
                add_timestamp,
                progress,
                *pipeline_params,
            )

        return self._call_transcription_worker(
            "transcribe_mic",
            {
                "mic_audio": mic_audio,
                "file_format": file_format,
                "add_timestamp": add_timestamp,
                "whisper_type": whisper_type,
                "pipeline_params": list(pipeline_params),
            },
        )

    def transcribe_mic_with_live_output(
        self,
        mic_audio,
        file_format="SRT",
        add_timestamp=True,
        *extra_args,
    ):
        progress, pipeline_params = self._split_progress_and_pipeline_args(extra_args)
        whisper_type = self._selected_whisper_type(pipeline_params)
        if not self._use_subprocess(pipeline_params):
            yield from self._local_inferencer_for(whisper_type).transcribe_mic_with_live_output(
                mic_audio,
                file_format,
                add_timestamp,
                progress,
                *pipeline_params,
            )
            return

        yield from self._stream_transcription_worker(
            "transcribe_mic_stream",
            {
                "mic_audio": mic_audio,
                "file_format": file_format,
                "add_timestamp": add_timestamp,
                "whisper_type": whisper_type,
                "pipeline_params": list(pipeline_params),
            },
        )

    def transcribe_live_preview(self, audio, *pipeline_params):
        whisper_type = self._selected_whisper_type(pipeline_params)
        return self._local_inferencer_for(whisper_type).transcribe_live_preview(
            audio,
            *pipeline_params,
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
