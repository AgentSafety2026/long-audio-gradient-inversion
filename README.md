# Audio Gradient Inversion — Source Code

Code accompanying:

> Xijie Zeng and Frank Rudzicz. *How to Recover Long Audio Sequences Through Gradient Inversion Attack With Dynamic Segment-based Reconstruction.* Interspeech 2025. 

## File map

| File | Purpose | Paper |
|---|---|---|
| `main.py` | Single-mode entry point. `--batch_size` is multiprocessing parallelism over files, **not** the paper's batch reconstruction. | §2.2 |
| `main_batch.py` | Batch-mode entry point. `--batch_size` **is** the paper's batch size. | §2.2 |
| `model_single.py` | CNN for single mode (4 × Conv2d, 12 ch, 5×5, Sigmoid). | §3.2 |
| `model_batch.py` | Encoder-decoder for batch mode (Conv2d + BN + LeakyReLU, ConvTranspose2d decoder, residuals). | §3.2 |
| `preprocess.py` | librosa mel-spec prep; `calculate_chunk_size` = paper's *Calculate Segment Size*. | §2.1, §3.3 |
| `chunk_processing.py` | Split / save-as-PNG / reassemble segments. | §2.1 |
| `chunk_reconstruction.py` | Single-mode gradient-matching loop (L-BFGS, iDLG label trick). | §2.2, Eq. 1 |
| `batch_reconstruction.py` | Batch-mode reconstruction loop. | §2.2, Eq. 4–6 |
| `compute_wer.py` | Whisper-Large transcription + WER. | §4.2 |
| `speaker_verification.py` | ECAPA-TDNN speaker-embedding similarity. | §4.2 |
| `utils.py` | Shared helpers (metrics, audio listing, output dir setup). | — |

Sample audio for each experiment lives in folders named `<dataset>_<batchsize>` (batch mode) or `<dataset>_single` (single mode).

## Setup

Python 3.10 recommended.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The evaluation scripts need `openai-whisper`, `jiwer`, `speechbrain`, `torchaudio`, `pandas`, `tqdm` — uncomment them in `requirements.txt` if you want WER / speaker-verification numbers.

## Running

### Single mode

```bash
python main.py --folder AudioMN_single --batch_size 1
```

### Batch mode

```bash
python main_batch.py --folder AudioMN_8 --batch_size 4
```

### Evaluation

```bash
python compute_wer.py --original AudioMN_8 --reconstructed output/reconstructed_audio
python speaker_verification.py --original AudioMN_8 --reconstructed output/reconstructed_audio
```

(`--original short_wavs` is a stale default; always pass `--original` explicitly.)

### Output layout

- `output/reconstructed_audio/<name>_reconstructed.wav` — final audio
- `output/plots/<name>_*.png` and `output/plots/<name>/` — full-spectrogram and per-chunk plots
- `output/chunks/<name>/` — saved originals (mel-spec npy + chunk visualizations)
- `output/reconstructed_chunks/<name>/` — per-chunk gradient-inversion outputs