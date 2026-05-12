import argparse
import os
import time
import multiprocessing as mp

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from preprocess import load_and_preprocess_audio, calculate_chunk_size, adjust_mel_spec_shape
from chunk_processing import split_mel_spectrogram, save_chunks, reconstruct_from_chunks
from chunk_reconstruction import process_all_chunks
from utils import (
    get_audio_files,
    create_output_dirs,
    calculate_snr,
    calculate_cosine_similarity,
    calculate_audio_cosine_similarity,
    calculate_audio_mse,
    calculate_mel_mse,
    calculate_statistics,
)


def prepare_chunks(audio_path):
    y, sr, S_db = load_and_preprocess_audio(audio_path)
    
    print(f"Processing {audio_path}")
    print(f"Original mel-spectrogram shape: {S_db.shape}")
    
    chunk_height, chunk_width = calculate_chunk_size(S_db.shape)
    adjusted_height, adjusted_width = adjust_mel_spec_shape(S_db.shape, chunk_height, chunk_width)
    S_db_adjusted = S_db[:adjusted_height, :adjusted_width]
    print(f"Adjusted mel-spectrogram shape: {S_db_adjusted.shape}")
    print(f"Chunk size: {chunk_height}x{chunk_width}")

    chunks = split_mel_spectrogram(S_db_adjusted, chunk_height, chunk_width)
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk shape: {chunks[0].shape}")

    # Save original chunks and S_db_adjusted
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    chunk_dir = f'output/chunks/{audio_name}'
    os.makedirs(chunk_dir, exist_ok=True)
    save_chunks(chunks, chunk_dir)
    np.save(f'{chunk_dir}/original_S_db.npy', S_db_adjusted)
    np.save(f'{chunk_dir}/original_audio.npy', y) 

    return audio_name, S_db_adjusted.shape, (chunk_height, chunk_width), sr

def process_single_audio(args):
    audio_name, original_shape, chunk_shape, sr = args
    chunk_dir = f'output/chunks/{audio_name}'
    
    start_time = time.time()

    # Load original audio and mel-spectrogram
    original_audio = np.load(f'{chunk_dir}/original_audio.npy')
    S_db_adjusted = np.load(f'{chunk_dir}/original_S_db.npy')
    
    # Run gradient inversion on each chunk. Chunks come back as (1, 3, h, w) tensors
    # in mel-spec [0, 1] normalized scale, already in librosa orientation.
    reconstructed_chunks, all_losses, all_mses = process_all_chunks(chunk_dir, chunk_shape)
    re_chunks = [c[0].mean(dim=0).cpu().numpy() for c in reconstructed_chunks]

    # Denormalize [0, 1] back to [-80, 0] dB — inverse of the (x + 80) / 80 mapping
    # applied in chunk_reconstruction.process_chunk.
    normalized_chunks = [np.clip(c, 0.0, 1.0) * 80.0 - 80.0 for c in re_chunks]

    reconstructed_S_db = reconstruct_from_chunks(normalized_chunks, original_shape)

    # Convert from dB scale back to power scale
    reconstructed_S = librosa.db_to_power(reconstructed_S_db)

    # Inverse mel spectrogram to get audio
    y_reconstructed = librosa.feature.inverse.mel_to_audio(
        reconstructed_S,
        sr=sr,
        n_iter=10
    )

    # Normalize the audio
    y_reconstructed = y_reconstructed / np.max(np.abs(y_reconstructed))

    # Apply a simple low-pass filter
    b, a = butter(4, 0.2, btype='low', analog=False)
    y_filtered = filtfilt(b, a, y_reconstructed)

    end_time = time.time()

    # Calculate all metrics
    snr = calculate_snr(y_filtered)
    mel_cos_sim = calculate_cosine_similarity(S_db_adjusted, reconstructed_S_db)
    mel_mse = calculate_mel_mse(S_db_adjusted, reconstructed_S_db)
    audio_cos_sim = calculate_audio_cosine_similarity(original_audio, y_filtered)
    audio_mse = calculate_audio_mse(original_audio, y_filtered)

    return (
        reconstructed_S_db,
        S_db_adjusted,
        y_filtered,
        sr,
        snr,
        mel_cos_sim,
        mel_mse,
        audio_cos_sim,
        audio_mse,
        (end_time - start_time)
    )

def main(folder_path, batch_size):
    create_output_dirs()

    audio_paths = get_audio_files(folder_path)
    if not audio_paths:
        print(f"No audio files found in {folder_path}")
        return
    
    print(f"Found {len(audio_paths)} audio files in {folder_path}")
    for path in audio_paths:
        print(f"  - {os.path.basename(path)}")
    print()
    
    # First, prepare all chunks
    with mp.Pool(processes=min(len(audio_paths), mp.cpu_count())) as pool:
        chunk_info = pool.map(prepare_chunks, audio_paths)

    total_files = len(audio_paths)
    print(f"\nProcessing {total_files} files in batches of {batch_size}")

    processed_files = 0

    all_snrs = []
    all_mel_cos_sims = []
    all_mel_mses = []
    all_audio_cos_sims = []
    all_audio_mses = []

    for i in range(0, len(chunk_info), batch_size):
        batch = chunk_info[i:i+batch_size]
        current_batch_size = len(batch)
        print(f"\nProcessing batch {i//batch_size + 1} ({processed_files + 1}-{processed_files + current_batch_size} of {total_files})")

        with mp.Pool(processes=min(batch_size, mp.cpu_count())) as pool:
            results = pool.map(process_single_audio, batch)
        
        for j, (reconstructed_S_db, S_db_adjusted, y_filtered, sr, snr, mel_cos_sim, mel_mse, 
               audio_cos_sim, audio_mse, processing_time) in enumerate(results):
            audio_name = batch[j][0]

            # Store all metrics
            all_snrs.append(snr)
            all_mel_cos_sims.append(mel_cos_sim)
            all_mel_mses.append(mel_mse)
            all_audio_cos_sims.append(audio_cos_sim)
            all_audio_mses.append(audio_mse)
            
            # Save reconstructed audio
            sf.write(f'output/reconstructed_audio/{audio_name}_reconstructed.wav', y_filtered, sr)
            
            # Save plots
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(reconstructed_S_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Reconstructed Mel-spectrogram - {audio_name}')
            plt.tight_layout()
            plt.savefig(f'output/plots/{audio_name}_reconstructed_mel_spectrogram.png')
            plt.close()

            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_db_adjusted, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Original Mel-spectrogram - {audio_name}')
            plt.tight_layout()
            plt.savefig(f'output/plots/{audio_name}_original_mel_spectrogram.png')
            plt.close()
            
            # Print results
            print(f"Processed {audio_name}:")
            print(f"  SNR: {snr:.4f} dB")
            print(f"  Mel-spectrogram Cosine Similarity: {mel_cos_sim:.4f}")
            print(f"  Mel-spectrogram MSE: {mel_mse:.8f}")
            print(f"  Audio Waveform Cosine Similarity: {audio_cos_sim:.4f}")
            print(f"  Audio Waveform MSE: {audio_mse:.8f}")
            print(f"  Processing time: {processing_time:.2f} seconds")
            print()
            
        processed_files += current_batch_size


    # Calculate and print overall statistics
    print("\nOverall Statistics:")
    metrics = [
        ("SNR (dB)", all_snrs),
        ("Mel-spectrogram Cosine Similarity", all_mel_cos_sims),
        ("Mel-spectrogram MSE", all_mel_mses),
        ("Audio Waveform Cosine Similarity", all_audio_cos_sims),
        ("Audio Waveform MSE", all_audio_mses)
    ]
    
    for metric_name, values in metrics:
        mean, std = calculate_statistics(values)
        print(f"{metric_name}: {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch audio reconstruction from mel-spectrogram")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing audio files")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="multiprocessing.Pool parallelism — number of audio files reconstructed concurrently (NOT the paper's batch reconstruction; see main_batch.py for that)")
    args = parser.parse_args()
    
    main(args.folder, args.batch_size)