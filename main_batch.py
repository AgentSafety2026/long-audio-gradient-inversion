import argparse
import os
import time

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from preprocess import load_and_preprocess_audio, calculate_chunk_size, adjust_mel_spec_shape
from chunk_processing import split_mel_spectrogram, save_chunks, reconstruct_from_chunks
from batch_reconstruction import process_batch_chunks
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


def process_batch(batch_paths):
    """Process a batch of audio files together"""
    batch_chunks = []
    batch_shapes = []
    batch_info = []
    original_audios = []  # Store original audio waveforms
    
    print("\nPreparing batch:")
    for audio_path in batch_paths:
        # Load and preprocess
        y, sr, S_db = load_and_preprocess_audio(audio_path)
        original_audios.append(y)  # Store original audio
        
        print(f"Processing {audio_path}")
        print(f"Original mel-spectrogram shape: {S_db.shape}")
        
        chunk_height, chunk_width = calculate_chunk_size(S_db.shape)
        adjusted_height, adjusted_width = adjust_mel_spec_shape(S_db.shape, chunk_height, chunk_width)
        S_db_adjusted = S_db[:adjusted_height, :adjusted_width]
        
        chunks = split_mel_spectrogram(S_db_adjusted, chunk_height, chunk_width)
        print(f"    Number of chunks: {len(chunks)}")
        print(f"    Chunk shape: {chunks[0].shape}")
        
        batch_chunks.append(chunks)
        batch_shapes.append((chunk_height, chunk_width))
        batch_info.append((os.path.basename(audio_path), S_db_adjusted, sr, (adjusted_height, adjusted_width)))
    
    print("\nProcessing batch together...")
    start_time = time.time()
    reconstructed_batch = process_batch_chunks(batch_chunks, batch_shapes)
    
    results = []
    for i, (audio_name, S_db_adjusted, sr, original_shape) in enumerate(batch_info):
        reconstructed_chunks = reconstructed_batch[i]
        
        # Reconstruct mel-spectrogram
        reconstructed_S_db = reconstruct_from_chunks(reconstructed_chunks, original_shape)
        
        # Convert to audio
        reconstructed_S = librosa.db_to_power(reconstructed_S_db)
        y_reconstructed = librosa.feature.inverse.mel_to_audio(
            reconstructed_S,
            sr=sr,
            n_iter=10
        )
        
        # Normalize and filter
        y_reconstructed = y_reconstructed / np.max(np.abs(y_reconstructed))
        b, a = butter(4, 0.2, btype='low', analog=False)
        y_filtered = filtfilt(b, a, y_reconstructed)
        
        # Calculate metrics
        snr = calculate_snr(y_filtered)
        
        # Mel-spectrogram similarities
        mel_cosine_sim = calculate_cosine_similarity(S_db_adjusted, reconstructed_S_db)
        mel_mse = calculate_mel_mse(S_db_adjusted, reconstructed_S_db)
        
        # Audio waveform similarities
        audio_cosine_sim = calculate_audio_cosine_similarity(original_audios[i], y_filtered)
        audio_mse = calculate_audio_mse(original_audios[i], y_filtered)
        
        results.append((
            reconstructed_S_db, 
            S_db_adjusted, 
            y_filtered, 
            sr, 
            snr, 
            mel_cosine_sim,
            mel_mse,
            audio_cosine_sim,
            audio_mse
        ))
    
    processing_time = time.time() - start_time
    return results, processing_time

def main(folder_path, batch_size):
    create_output_dirs()
    
    # Get all audio files from the folder
    audio_paths = get_audio_files(folder_path)
    if not audio_paths:
        print(f"No audio files found in {folder_path}")
        return
    
    print(f"Found {len(audio_paths)} audio files in {folder_path}")
    for path in audio_paths:
        print(f"  - {os.path.basename(path)}")
    print()
    
    # Initialize lists for all metrics
    all_snrs = []
    all_mel_cosine_sims = []
    all_mel_mses = []
    all_audio_cosine_sims = []
    all_audio_mses = []
    
    start_time = time.time()
    
    # Process files in batches
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(audio_paths)-1)//batch_size + 1}")
        
        results, processing_time = process_batch(batch_paths)
        
        # Save results for each audio in the batch
        for j, (reconstructed_S_db, S_db_adjusted, y_filtered, sr, snr, mel_cosine_sim, mel_mse, audio_cosine_sim, audio_mse) in enumerate(results):
            audio_name = os.path.splitext(os.path.basename(batch_paths[j]))[0]
            
            # Store metrics
            all_snrs.append(snr)
            all_mel_cosine_sims.append(mel_cosine_sim)
            all_mel_mses.append(mel_mse)
            all_audio_cosine_sims.append(audio_cosine_sim)
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
            
            print(f"Processed {audio_name}:")
            print(f"  SNR: {snr:.4f} dB")
            print(f"  Mel-spectrogram Cosine Similarity: {mel_cosine_sim:.4f}")
            print(f"  Mel-spectrogram MSE: {mel_mse:.4f}")
            print(f"  Audio Waveform Cosine Similarity: {audio_cosine_sim:.4f}")
            print(f"  Audio Waveform MSE: {audio_mse:.4f}")
            print(f"  Processing time: {processing_time/len(results):.2f} seconds")
            print()
    
    total_time = time.time() - start_time
    
    # Calculate and save statistics
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stats_file = f"output/reconstruction_statistics_{timestamp}.txt"
    
    with open(stats_file, 'w') as f:
        f.write("Reconstruction Statistics\n")
        f.write("=======================\n\n")
        f.write(f"Processing Details:\n")
        f.write(f"  Number of files processed: {len(audio_paths)}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Total processing time: {total_time:.2f} seconds\n")
        f.write(f"  Average time per file: {total_time/len(audio_paths):.2f} seconds\n\n")
        
        # Write overall statistics
        for metric_name, metric_values in [
            ("SNR (dB)", all_snrs),
            ("Mel-spectrogram Cosine Similarity", all_mel_cosine_sims),
            ("Mel-spectrogram MSE", all_mel_mses),
            ("Audio Waveform Cosine Similarity", all_audio_cosine_sims),
            ("Audio Waveform MSE", all_audio_mses)
        ]:
            mean, std = calculate_statistics(metric_values)
            f.write(f"{metric_name}: {mean:.4f} ± {std:.4f}\n")
        
        f.write("\nIndividual File Results:\n")
        f.write("=======================\n")
        
        for i, audio_path in enumerate(audio_paths):
            name = os.path.splitext(os.path.basename(audio_path))[0]
            f.write(f"\n{name}:\n")
            f.write(f"  SNR: {all_snrs[i]:.4f} dB\n")
            f.write(f"  Mel-spectrogram Cosine Similarity: {all_mel_cosine_sims[i]:.4f}\n")
            f.write(f"  Mel-spectrogram MSE: {all_mel_mses[i]:.4f}\n")
            f.write(f"  Audio Waveform Cosine Similarity: {all_audio_cosine_sims[i]:.4f}\n")
            f.write(f"  Audio Waveform MSE: {all_audio_mses[i]:.4f}\n")
    
    # Print summary to console
    print("\nOverall Statistics:")
    for metric_name, metric_values in [
        ("SNR (dB)", all_snrs),
        ("Mel-spectrogram Cosine Similarity", all_mel_cosine_sims),
        ("Mel-spectrogram MSE", all_mel_mses),
        ("Audio Waveform Cosine Similarity", all_audio_cosine_sims),
        ("Audio Waveform MSE", all_audio_mses)
    ]:
        mean, std = calculate_statistics(metric_values)
        print(f"{metric_name}: {mean:.4f} ± {std:.4f}")
    
    print(f"\nDetailed statistics saved to {stats_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch audio reconstruction from mel-spectrogram")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing audio files")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="paper's batch size — number of audios reconstructed jointly per gradient-matching call")
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist.")
        exit(1)
        
    main(args.folder, args.batch_size)