import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model_batch import LeNet, weights_init

def normalize_mel_chunk(chunk):
    """Normalize mel spectrogram chunk from [-80, 0] dB to [0, 1]"""
    return (chunk + 80) / 80  # Map [-80, 0] to [0, 1]

def denormalize_mel_chunk(chunk):
    """Denormalize from [0, 1] back to [-80, 0] dB"""
    return chunk * 80 - 80  # Map [0, 1] back to [-80, 0]

def spectral_loss(original, reconstructed):
    """Calculate loss in frequency domain"""
    orig_fft = torch.fft.fft2(original)
    recon_fft = torch.fft.fft2(reconstructed)
    return torch.mean(torch.abs(orig_fft - recon_fft))


def process_batch_chunks(batch_chunks, chunk_shapes, num_attempts=2, max_iterations=500):
    """Process multiple chunks from different audio files together with improved optimization"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find the maximum chunk dimensions in the batch
    max_height = max(shape[0] for shape in chunk_shapes)
    max_width = max(shape[1] for shape in chunk_shapes)
    target_shape = (max_height, max_width)
    
    # Prepare all chunks
    all_gt_data = []
    chunk_indices = []
    original_shapes = []
    
    for i, chunks in enumerate(batch_chunks):
        for chunk in chunks:
            # Normalize chunk from [-80, 0] dB to [0, 1]
            normalized_chunk = normalize_mel_chunk(chunk)
            
            # Convert to tensor and add channel dimension
            chunk_tensor = torch.FloatTensor(normalized_chunk).unsqueeze(0)
            
            # Resize if necessary
            if chunk.shape != target_shape:
                chunk_tensor = F.interpolate(
                    chunk_tensor.unsqueeze(0),
                    size=target_shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            all_gt_data.append(chunk_tensor)
            chunk_indices.append(i)
            original_shapes.append(chunk.shape)
    
    # Stack all chunks into a single batch
    all_gt_data = torch.stack(all_gt_data).to(device)
    all_gt_data = all_gt_data.repeat(1, 3, 1, 1)  # Convert to 3 channels
    
    print(f"Processing batch of shape: {all_gt_data.shape}")
    
    best_reconstructed = None
    best_loss = float('inf')
    
    # Try multiple attempts
    for attempt in range(num_attempts):
        print(f"\nAttempt {attempt + 1}/{num_attempts}")
        
        # Initialize network
        net = LeNet(target_shape).to(device)
        net.apply(weights_init)
        
        # Initialize dummy data with values in [0, 1] range
        dummy_data = torch.rand_like(all_gt_data)  # Use rand instead of randn
        dummy_data.requires_grad_(True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([dummy_data], lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        
        for iters in range(max_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = net(dummy_data)
            
            # Calculate losses
            content_loss = F.mse_loss(reconstructed, all_gt_data)
            spec_loss = spectral_loss(reconstructed[:, 0], all_gt_data[:, 0])
            # Smoothness loss over both axes (paper Eq. 6): |∇_y B̂|_1 + |∇_x B̂|_1.
            smooth_loss = (
                torch.mean(torch.abs(reconstructed[:, :, 1:, :] - reconstructed[:, :, :-1, :]))
                + torch.mean(torch.abs(reconstructed[:, :, :, 1:] - reconstructed[:, :, :, :-1]))
            )
            
            # Combined loss
            total_loss = content_loss + 0.1 * spec_loss + 0.01 * smooth_loss

            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Clamp values to [0, 1] range
            with torch.no_grad():
                dummy_data.data.clamp_(0, 1)
            
            # Update learning rate
            scheduler.step(total_loss)
            
            if iters % 50 == 0:
                print(f"Iteration {iters:4d}: Loss = {total_loss.item():.6f}")
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_reconstructed = reconstructed.detach()
                
            if total_loss.item() < 1e-6:
                break
    
    # Split reconstructed chunks back to individual audio files
    reconstructed_chunks = []
    chunk_counts = [len(chunks) for chunks in batch_chunks]
    start_idx = 0
    
    for i, num_chunks in enumerate(chunk_counts):
        end_idx = start_idx + num_chunks
        file_chunks = best_reconstructed[start_idx:end_idx]
        
        # Convert to numpy and resize to original shape
        file_reconstructed = []
        orig_shape = chunk_shapes[i]
        
        for chunk in file_chunks:
            # Take first channel and convert to numpy
            chunk_np = chunk[0].cpu().numpy()
            
            # Resize if necessary
            if chunk_np.shape != orig_shape:
                chunk_img = Image.fromarray(chunk_np)
                chunk_np = np.array(chunk_img.resize(orig_shape[::-1]))
            
            # Denormalize back to [-80, 0] dB range
            chunk_np = denormalize_mel_chunk(chunk_np)
            
            file_reconstructed.append(chunk_np)
        
        reconstructed_chunks.append(file_reconstructed)
        start_idx = end_idx
    
    return reconstructed_chunks