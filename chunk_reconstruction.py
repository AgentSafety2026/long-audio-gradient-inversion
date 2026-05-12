import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from chunk_processing import split_mel_spectrogram
from model_single import LeNet, weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_chunk_to_unit(chunk_np):
    """Map a mel-spec chunk from [-80, 0] dB to [0, 1]."""
    return np.clip((chunk_np + 80.0) / 80.0, 0.0, 1.0)


def process_chunk(args, num_attempts=3, max_iterations=300):
    """Recover one mel-spec chunk via gradient matching (paper §2.2, Eq. 1).

    args is a tuple of (chunk_np, chunk_number, chunk_shape, audio_name).
    chunk_np is the original mel-spec chunk (2D, dB scale, librosa orientation).
    """
    chunk_np, chunk_number, chunk_shape, audio_name = args

    chunk_tensor = torch.from_numpy(_normalize_chunk_to_unit(chunk_np).copy()).float()
    chunk_tensor = chunk_tensor.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).contiguous()
    gt_data = chunk_tensor.to(device)
    gt_label = torch.tensor([6]).to(device)

    net = LeNet(input_shape=chunk_shape).to(device)
    net.apply(weights_init)
    criterion = nn.CrossEntropyLoss()

    out = net(gt_data)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # iDLG label inference from the last fully connected layer's gradient.
    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,))

    best_dummy_data = None
    best_loss = float('inf')
    history = []
    losses = []
    mses = []

    for attempt in range(num_attempts):
        dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummy_data], lr=1)

        for iters in range(max_iterations):
            def closure():
                optimizer.zero_grad()
                pred = net(dummy_data)
                dummy_loss = criterion(pred, label_pred)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)
            current_loss = closure().item()
            current_mse = torch.mean((dummy_data - gt_data) ** 2).item()

            losses.append(current_loss)
            mses.append(current_mse)

            if iters % 10 == 0:
                print(f"Chunk {chunk_number}, Attempt {attempt+1}, Iter {iters}: loss = {current_loss:.8f}, mse = {current_mse:.8f}")
                history.append(transforms.ToPILImage()(dummy_data[0].cpu()))

            if current_loss < best_loss:
                best_loss = current_loss
                best_dummy_data = dummy_data.detach().clone()

            if current_loss < 0.000001:
                break

        if best_loss < 0.000001:
            break

    # Per-audio output dirs so concurrent file processing doesn't clobber each other.
    plots_dir = f'output/plots/{audio_name}'
    chunks_out_dir = f'output/reconstructed_chunks/{audio_name}'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(chunks_out_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 10, 1)
    plt.imshow(transforms.ToPILImage()(gt_data[0].cpu()))
    plt.title("Original")
    plt.axis('off')
    for i in range(min(len(history), 29)):
        plt.subplot(3, 10, i + 2)
        plt.imshow(history[i])
        plt.title(f'iter={i*10}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/reconstruction_progress_chunk_{chunk_number}.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(transforms.ToPILImage()(gt_data[0].cpu()))
    plt.title(f"Ground truth (Chunk {chunk_number})")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(transforms.ToPILImage()(best_dummy_data[0].cpu()))
    plt.title(f"Reconstructed (Chunk {chunk_number})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/comparison_chunk_{chunk_number}.png')
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.imshow(transforms.ToPILImage()(best_dummy_data[0].cpu()), aspect='auto', interpolation='nearest')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f'{chunks_out_dir}/reconstructed_chunk_{chunk_number}.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()

    return best_dummy_data, losses, mses


def process_all_chunks(chunk_dir, chunk_shape):
    """Run gradient inversion on every chunk of one audio file.

    chunk_dir is 'output/chunks/{audio_name}', expected to contain
    original_S_db.npy (written by main.prepare_chunks).
    """
    audio_name = os.path.basename(chunk_dir)
    S_db = np.load(os.path.join(chunk_dir, 'original_S_db.npy'))
    chunks = split_mel_spectrogram(S_db, chunk_shape[0], chunk_shape[1])

    all_losses, all_mses, reconstructed_chunks = [], [], []
    for i, chunk_np in enumerate(chunks):
        reconstructed_chunk, losses, mses = process_chunk(
            (chunk_np, i + 1, chunk_shape, audio_name)
        )
        reconstructed_chunks.append(reconstructed_chunk)
        all_losses.append(losses)
        all_mses.append(mses)

    plots_dir = f'output/plots/{audio_name}'
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i, losses in enumerate(all_losses):
        plt.plot(losses, label=f'Chunk {i+1}')
    plt.title('Loss over iterations for all chunks')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, mses in enumerate(all_mses):
        plt.plot(mses, label=f'Chunk {i+1}')
    plt.title('MSE over iterations for all chunks')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/overall_loss_mse_trends.png')
    plt.close()

    print(f"All chunks processed for {audio_name}.")
    return reconstructed_chunks, all_losses, all_mses
