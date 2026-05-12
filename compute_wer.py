import os
import torch
import whisper # type: ignore
import numpy as np
from jiwer import wer # type: ignore
import pandas as pd
from tqdm import tqdm
import argparse

def get_base_filename(filename):
    """Get base filename without extension and '_reconstructed' suffix."""
    # Remove '_reconstructed' if present and get the name without extension
    return os.path.splitext(filename.replace('_reconstructed', ''))[0]

def find_matching_file(base_name, folder):
    """Find a matching audio file in the folder regardless of extension."""
    audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')
    for ext in audio_extensions:
        potential_file = base_name + ext
        if os.path.exists(os.path.join(folder, potential_file)):
            return potential_file
    return None

def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("large").to(device)
    return model

def transcribe_audio(model, audio_path):
    """Transcribe audio using Whisper model."""
    try:
        result = model.transcribe(audio_path)
        return result["text"].strip().lower()
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""

def compute_wer_for_pair(model, original_path, reconstructed_path):
    """Compute WER between original and reconstructed audio."""
    original_text = transcribe_audio(model, original_path)
    reconstructed_text = transcribe_audio(model, reconstructed_path)
    
    if not original_text or not reconstructed_text:
        return None
    
    return wer(original_text, reconstructed_text)

def evaluate_reconstructions(original_folder, reconstructed_folder):
    """Evaluate WER for all audio pairs."""
    print("Loading Whisper model...")
    model = load_whisper_model()
    
    # Get list of reconstructed files
    reconstructed_files = [f for f in os.listdir(reconstructed_folder) 
                         if f.endswith('_reconstructed.wav')]
    
    results = []
    
    print("\nProcessing audio pairs...")
    for reconstructed_file in tqdm(reconstructed_files):
        # Get base name and find matching original file
        base_name = get_base_filename(reconstructed_file)
        original_file = find_matching_file(base_name, original_folder)
        
        if original_file is None:
            print(f"Warning: No matching original file found for {reconstructed_file}")
            continue
        
        original_path = os.path.join(original_folder, original_file)
        reconstructed_path = os.path.join(reconstructed_folder, reconstructed_file)
        
        # Compute WER
        current_wer = compute_wer_for_pair(model, original_path, reconstructed_path)
        
        if current_wer is not None:
            results.append({
                'file': original_file,
                'reconstructed_file': reconstructed_file,
                'wer': current_wer
            })
    
    # Convert results to DataFrame for analysis
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("No valid pairs found for evaluation!")
        return None, None, None
    
    # Calculate statistics
    mean_wer = df['wer'].mean()
    std_wer = df['wer'].std()
    
    # Save detailed results
    df.to_csv('wer_results.csv', index=False)
    
    # Print summary
    print("\nDetailed Results:")
    print(df.to_string())
    print("\nSummary:")
    print(f"WER: {mean_wer:.4f} ± {std_wer:.4f}")
    
    return mean_wer, std_wer, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WER between original and reconstructed audio files")
    parser.add_argument("--original", type=str, default="short_wavs",
                      help="Path to folder containing original audio files")
    parser.add_argument("--reconstructed", type=str, default="output/reconstructed_audio",
                      help="Path to folder containing reconstructed audio files")
    
    args = parser.parse_args()
    
    # Check if folders exist
    if not os.path.exists(args.original):
        print(f"Error: Original audio folder '{args.original}' does not exist.")
        exit(1)
    if not os.path.exists(args.reconstructed):
        print(f"Error: Reconstructed audio folder '{args.reconstructed}' does not exist.")
        exit(1)
    
    # Run evaluation
    mean_wer, std_wer, results = evaluate_reconstructions(args.original, args.reconstructed)

# python compute_wer.py --original CV_16 --reconstructed output/reconstructed_audio