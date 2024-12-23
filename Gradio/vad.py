import time
from typing import TYPE_CHECKING

import librosa
import numpy as np
import torch
from silero_vad import get_speech_timestamps

if TYPE_CHECKING:
    from app import AppState

START_THRESHOLD = 0.5
PAUSE_THRESHOLD = 0.95
SAMPLING_RATE = 16000


def determine_pause(vad, stream: np.ndarray, state: "AppState") -> bool:
    """
    Determines if there is a pause in the given audio stream using Voice Activity Detection (VAD).

    Args:
        vad: The VAD model used to detect speech activity.
        stream (np.ndarray): The audio stream to be analyzed.
        state (AppState): The application state containing the sampling rate and talking state.

    Returns:
        bool: True if a pause is detected, False otherwise.
    """
    _st = time.time()
    _stream = stream.copy()
    try:
        _stream = _stream.astype(np.float32) / 32768.0
        if state.sampling_rate != SAMPLING_RATE:
            _stream = librosa.resample(
                _stream, orig_sr=state.sampling_rate, target_sr=SAMPLING_RATE
            )

        _stream = torch.from_numpy(_stream).to(torch.float32)
        chunks = get_speech_timestamps(_stream, vad, sampling_rate=SAMPLING_RATE)
        dur_vad = np.sum([(c["end"] - c["start"]) / SAMPLING_RATE for c in chunks])
    except Exception:
        dur_vad = -1.0

    duration = len(stream) / state.sampling_rate
    if (dur_vad / duration >= START_THRESHOLD) and (not state.started_talking):
        return False
    return (duration - dur_vad) / duration >= PAUSE_THRESHOLD
