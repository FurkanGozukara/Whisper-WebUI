import os
import torch
from typing import List, Union, BinaryIO, Optional, Tuple
import numpy as np
import time
import logging
import gc

from modules.utils.paths import DIARIZATION_MODELS_DIR
from modules.diarize.audio_loader import load_audio
from modules.whisper.data_classes import *


class Diarizer:
    def __init__(self,
                 model_dir: str = DIARIZATION_MODELS_DIR
                 ):
        self.device = self.get_device()
        self.available_device = self.get_available_device()
        self.compute_type = "bfloat16"
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.pipe = None

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            transcribed_result: List[Segment],
            use_auth_token: str,
            device: Optional[str] = None
            ) -> Tuple[List[Segment], float]:
        """
        Diarize transcribed result as a post-processing

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio input. This can be file path or binary type.
        transcribed_result: List[Segment]
            transcribed result through whisper.
        use_auth_token: str
            Optional fallback token for advanced/manual diarization setups.
        device: Optional[str]
            Device for diarization.

        Returns
        ----------
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for running
        """
        start_time = time.time()

        if device is None:
            device = self.device

        if device != self.device or self.pipe is None:
            self.update_pipe(
                device=device,
                use_auth_token=use_auth_token
            )

        if self.pipe is None:
            raise RuntimeError("Diarization pipeline is unavailable.")

        audio = load_audio(audio)
        from modules.diarize.diarize_pipeline import assign_word_speakers

        diarization_segments = self.pipe(audio)
        diarized_result = assign_word_speakers(
            diarization_segments,
            {"segments": transcribed_result}
        )

        segments_result = []
        for segment in diarized_result["segments"]:
            speaker = "None"
            if "speaker" in segment:
                speaker = segment["speaker"]
            diarized_text = speaker + "|" + segment["text"].strip()
            segments_result.append(Segment(
                start=segment["start"],
                end=segment["end"],
                text=diarized_text
            ))

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_pipe(self,
                    use_auth_token: Optional[str] = None,
                    device: Optional[str] = None,
                    ):
        """
        Set pipeline for diarization

        Parameters
        ----------
        use_auth_token: str
            Optional fallback token for advanced/manual diarization setups.
        device: str
            Device for diarization.
        """
        if device is None:
            device = self.get_device()
        self.device = device

        os.makedirs(self.model_dir, exist_ok=True)

        logger = logging.getLogger("speechbrain.utils.train_logger")
        # Disable redundant torchvision warning message
        logger.disabled = True
        try:
            from modules.diarize.diarize_pipeline import DiarizationPipeline

            self.pipe = DiarizationPipeline(
                use_auth_token=use_auth_token,
                device=device,
                cache_dir=self.model_dir
            )
        except Exception as e:
            # Keep diarization optional: don't crash the whole app if diarization can't be initialized.
            print(
                "\nFailed to initialize diarization pipeline.\n"
                f"Error: {type(e).__name__}: {e}\n"
                "Tip: The installer normally downloads the offline diarization bundle automatically. "
                "Run DownloadModels.py or the installer again to repair "
                "models/Diarization/Speaker_Diarization_3_1.\n"
            )
            self.pipe = None
        logger.disabled = False

    def offload(self):
        """Offload the model and free up the memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        if self.device == "xpu":
            torch.xpu.empty_cache()
            torch.xpu.reset_accumulated_memory_stats()
            torch.xpu.reset_peak_memory_stats()
        gc.collect()

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.xpu.is_available():
            return "xpu"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def get_available_device():
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.xpu.is_available():
            devices.append("xpu")
        if torch.backends.mps.is_available():
            devices.append("mps")
        return devices
