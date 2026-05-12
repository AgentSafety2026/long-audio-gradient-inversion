import numpy as np
import matplotlib.pyplot as plt


def split_mel_spectrogram(mel_spec, chunk_height, chunk_width):
    height, width = mel_spec.shape
    chunks = []
    for i in range(0, height, chunk_height):
        for j in range(0, width, chunk_width):
            chunk = mel_spec[i:i+chunk_height, j:j+chunk_width]
            chunks.append(chunk)
    return chunks

def save_chunks(chunks, output_dir):
    for i, chunk in enumerate(chunks):
        plt.figure(figsize=(5, 5))
        plt.imshow(chunk, aspect='auto', origin='lower')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(f'{output_dir}/chunk_{i+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

def reconstruct_from_chunks(chunks, original_shape):
    height, width = original_shape
    chunk_height, chunk_width = chunks[0].shape
    reconstructed = np.zeros(original_shape)
    chunk_idx = 0
    for i in range(0, height, chunk_height):
        for j in range(0, width, chunk_width):
            reconstructed[i:i+chunk_height, j:j+chunk_width] = chunks[chunk_idx]
            chunk_idx += 1
    return reconstructed