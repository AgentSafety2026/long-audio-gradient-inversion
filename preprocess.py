import numpy as np
import librosa


def load_and_preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    return y, sr, S_db

def calculate_chunk_size(mel_spec_shape, target_chunk_size=(16, 30), width_range=(20, 35)):
    """Paper's *Calculate Segment Size* (§2.1).

    Picks (chunk_height, chunk_width) for splitting a mel-spectrogram. Height
    is capped to the spectrogram's actual height. Width is the largest value
    in width_range that divides the spectrogram's width; if no exact divisor
    exists in that range, the largest in-range candidate ≤ width is returned
    and adjust_mel_spec_shape() trims the spectrogram to fit.
    """
    height, width = mel_spec_shape
    chunk_height = min(target_chunk_size[0], height)
    min_w, max_w = width_range
    upper = min(max_w, width)
    if upper < min_w:
        return chunk_height, width
    for chunk_width in range(upper, min_w - 1, -1):
        if width % chunk_width == 0:
            return chunk_height, chunk_width
    return chunk_height, upper

def adjust_mel_spec_shape(shape, chunk_height, chunk_width):
    height, width = shape
    new_height = (height // chunk_height) * chunk_height
    new_width = (width // chunk_width) * chunk_width
    return new_height, new_width