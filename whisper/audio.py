"""
Complete audio preprocessing implementation for tinygrad Whisper
Based on tinygrad/examples/whisper.py with full mel spectrogram generation
"""
import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union, List

import numpy as np
import librosa
from tinygrad import Tensor

from .utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input
N_MELS = 80

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token

# Constants from tinygrad implementation
SEGMENT_SECONDS = 30
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_SECONDS  # 480000
FRAMES_PER_SEGMENT = SAMPLES_PER_SEGMENT // HOP_LENGTH  # 3000


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    # Use librosa for more reliable audio loading
    try:
        waveform, _ = librosa.load(file, sr=sr)
        return waveform.astype(np.float32)
    except Exception:
        # Fallback to ffmpeg method
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]
        try:
            out = run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if isinstance(array, Tensor):
        if array.shape[axis] > length:
            # Use slicing for tinygrad tensors
            slices = [slice(None)] * array.ndim
            slices[axis] = slice(0, length)
            array = array[tuple(slices)]

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = array.pad(pad_widths)
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return Tensor(f[f"mel_{n_mels}"])


def prep_audio(waveforms: List[np.ndarray], batch_size: int, truncate=False) -> np.ndarray:
    """
    Complete audio preprocessing implementation from tinygrad/examples/whisper.py
    
    :param waveforms: A list of possibly variable length 16000Hz audio samples
    :param batch_size: The batch_size associated with the Whisper model being used to transcribe the audio.
                       Used to prevent JIT mismatch errors since the encoder does not accept symbolic shapes
    :param truncate: If true, truncates (or pads) audio to exactly 30s for a single encoder pass
    :return: mel spectrogram of the given waveforms
    """
    def pad_or_trim_audio(arr, target_len):
        curr_len = len(arr)
        if curr_len == target_len:
            return arr
        elif curr_len < target_len:
            return np.pad(arr, (0, target_len - curr_len), 'constant')
        else:
            return arr[:target_len]

    max_len = SAMPLES_PER_SEGMENT if truncate else max(len(wav) for wav in waveforms)
    if (r := max_len % SAMPLES_PER_SEGMENT) > 0: 
        max_len += SAMPLES_PER_SEGMENT - r
    waveforms = np.array(list(map(lambda w: pad_or_trim_audio(w, max_len), waveforms)))
    assert waveforms.shape[0] <= batch_size
    if waveforms.shape[0] < batch_size:
        # we could have a symbolic batch_size dim instead of manually padding here if conv/layernorm supported symbolic shapes
        waveforms = np.pad(waveforms, pad_width=((0, batch_size - waveforms.shape[0]), (0, 0)))

    stft = librosa.stft(waveforms, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', dtype=np.csingle)
    magnitudes = np.absolute(stft[..., :-1]) ** 2
    mel_spec = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS) @ magnitudes

    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max((1,2), keepdims=True) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[str] = None,
):
    """
    Compute the log-Mel spectrogram using tinygrad
    
    This implementation uses the same approach as tinygrad/examples/whisper.py

    Parameters
    ----------
    audio: Union[str, np.ndarray, Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[str]
        ignored for tinygrad

    Returns
    -------
    Tensor, shape = (n_mels, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if isinstance(audio, str):
        audio = load_audio(audio)
    elif isinstance(audio, Tensor):
        audio = audio.numpy()
    
    if padding > 0:
        audio = np.pad(audio, (0, padding))
    
    # Use the complete preprocessing function
    log_spec = prep_audio([audio], batch_size=1, truncate=True)
    
    return Tensor(log_spec[0])  # Return first (and only) item from batch


def load_file_waveform(filename):
    """Load audio file and return waveform array"""
    waveform, _ = librosa.load(filename, sr=SAMPLE_RATE)
    return waveform