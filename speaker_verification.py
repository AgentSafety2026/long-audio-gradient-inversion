import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from speechbrain.pretrained import EncoderClassifier # type: ignore
from scipy.special import softmax

def get_base_filename(filename):
    """Get base filename without extension and '_reconstructed' suffix."""
    return os.path.splitext(filename.replace('_reconstructed', ''))[0]

def find_matching_file(base_name, folder):
    """Find a matching audio file in the folder regardless of extension."""
    audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')
    for ext in audio_extensions:
        potential_file = base_name + ext
        if os.path.exists(os.path.join(folder, potential_file)):
            return potential_file
    return None

def load_ecapa_model():
    """Load pretrained ECAPA-TDNN model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    return model

def get_embedding(model, audio_path):
    """Extract speaker embedding from audio file."""
    try:
        signal, fs = torchaudio.load(audio_path)
        
        # If stereo, convert to mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        # Resample if necessary (ECAPA-TDNN expects 16kHz)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
        
        embedding = model.encode_batch(signal)
        return embedding.squeeze().cpu().detach().numpy()
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def compute_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    scores = softmax([similarity, -similarity])
    return similarity, scores[0]

def evaluate_speaker_verification(original_folder, reconstructed_folder):
    """Evaluate speaker verification scores for all audio pairs."""
    print("Loading ECAPA-TDNN model...")
    model = load_ecapa_model()
    
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
        
        # Get embeddings
        original_embedding = get_embedding(model, original_path)
        reconstructed_embedding = get_embedding(model, reconstructed_path)
        
        if original_embedding is not None and reconstructed_embedding is not None:
            similarity, probability = compute_similarity(original_embedding, reconstructed_embedding)
            
            results.append({
                'file': original_file,
                'reconstructed_file': reconstructed_file,
                'similarity_score': similarity,
                'same_speaker_probability': probability
            })
    
    # Convert results to DataFrame for analysis
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("No valid pairs found for evaluation!")
        return None, None, None, None, None
    
    # Calculate statistics
    mean_similarity = df['similarity_score'].mean()
    std_similarity = df['similarity_score'].std()
    mean_probability = df['same_speaker_probability'].mean()
    std_probability = df['same_speaker_probability'].std()
    
    # Save detailed results
    df.to_csv('speaker_verification_results.csv', index=False)
    
    # Print summary
    print("\nDetailed Results:")
    print(df.to_string())
    print("\nSummary:")
    print(f"Similarity Score: {mean_similarity:.4f} ± {std_similarity:.4f}")
    print(f"Same Speaker Probability: {mean_probability:.4f} ± {std_probability:.4f}")
    
    return mean_similarity, std_similarity, mean_probability, std_probability, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate speaker verification between original and reconstructed audio files")
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
    mean_sim, std_sim, mean_prob, std_prob, results = evaluate_speaker_verification(
        args.original, args.reconstructed)
    
# python speaker_verification.py --original foldername --reconstructed output/reconstructed_audio