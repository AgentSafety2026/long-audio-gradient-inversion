"""Shared helpers used by both main.py (single mode) and main_batch.py (batch mode)."""

import os

import numpy as np


AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')


def get_audio_files(folder_path):
    """Sorted list of supported audio files under folder_path (non-recursive)."""
    audio_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(AUDIO_EXTS)
    ]
    return sorted(audio_files)


def create_output_dirs():
    """Create the conventional output/ subdirectories used by both pipelines."""
    for d in ('output/chunks', 'output/reconstructed_chunks',
              'output/plots', 'output/reconstructed_audio'):
        os.makedirs(d, exist_ok=True)


def calculate_snr(audio_signal):
    """SNR estimate using the bottom 20% of amplitudes as the noise floor."""
    signal_power = np.mean(audio_signal ** 2)
    sorted_amplitudes = np.sort(np.abs(audio_signal))
    noise_threshold = sorted_amplitudes[int(len(sorted_amplitudes) * 0.2)]
    noise_samples = audio_signal[np.abs(audio_signal) <= noise_threshold]
    if len(noise_samples) > 0:
        noise_power = np.mean(noise_samples ** 2)
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
    return float('inf')


def calculate_cosine_similarity(original, reconstructed):
    """Cosine similarity between two arrays (flattened)."""
    a = original.flatten()
    b = reconstructed.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calculate_audio_cosine_similarity(original, reconstructed):
    """Cosine similarity between two waveforms (truncated to shorter length)."""
    n = min(len(original), len(reconstructed))
    a, b = original[:n], reconstructed[:n]
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calculate_audio_mse(original, reconstructed):
    """MSE between two waveforms (truncated to shorter length)."""
    n = min(len(original), len(reconstructed))
    return np.mean((original[:n] - reconstructed[:n]) ** 2)


def calculate_mel_mse(original, reconstructed):
    """MSE between two mel-spectrograms."""
    return np.mean((original - reconstructed) ** 2)


def calculate_statistics(values):
    """Mean and standard deviation of an iterable of floats."""
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr))
