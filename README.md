# Speech Transcription with Speaker Diarization

Combines NVIDIA NeMo ASR with pyannote.audio speaker diarization for multi-speaker audio transcription. **Optimized for long audio files** with intelligent chunking and GPU memory management.

## Features

- **Long Audio Support**: Handles audio files of any length via intelligent chunking
- **Memory Optimized**: GPU cache clearing after each chunk prevents OOM errors
- **Smart Chunking**: Splits audio at silence boundaries for clean transcriptions
- **Diarization Chunking**: Long audio diarization with 60s overlap for speaker continuity
- **Speaker Merging**: Automatic speaker remapping across chunks using temporal patterns
- **Parallel Processing**: Configurable worker threads for batch processing

## Key Improvements (vs Original)

| Feature | Original | This Version |
|---------|----------|--------------|
| Max Chunk Duration | 24 min | 10 min (configurable) |
| Batch Size | 8 | 4 (memory safe) |
| Diarization | Single pass | Chunked with merging |
| GPU Memory | No management | Auto-cleared per chunk |
| Split Algorithm | Could hang | Guaranteed termination |

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA 11.8+ (recommended)
- HuggingFace account with access to:
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/segmentation-3.0

## Installation

### Step 1: Create Environment

**Option A: Using Conda (recommended)**
```bash
conda create -n speech_transcription python=3.11
conda activate speech_transcription
```

**Option B: Using venv**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

### Step 2: Install PyTorch with CUDA

Install PyTorch with GPU support **before** other dependencies:

```bash
# For CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Check your CUDA version with `nvidia-smi` and choose accordingly. See [pytorch.org](https://pytorch.org/get-started/locally/) for other options.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Expected output: `PyTorch: 2.x.x+cu124, CUDA: True`

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your settings
```

Key settings:
```bash
HUGGINGFACE_TOKEN=hf_your_token_here  # Required for pyannote
ASR_BATCH_SIZE=4                       # Reduce if OOM errors
CHUNK_DURATION_SECONDS=600             # 10 minutes (reduce for less GPU memory)
NUM_WORKERS=1                          # Increase for parallel file processing
```

## Configuration for Setting Number of Speakers

Adjust speaker detection in your `.env` file based on your use case:

**For interviews (2 people):**
```bash
DIARIZATION_MIN_SPEAKERS=2
DIARIZATION_MAX_SPEAKERS=2
```

**For meetings (multiple people):**
```bash
DIARIZATION_MIN_SPEAKERS=2
DIARIZATION_MAX_SPEAKERS=10
```

**For unknown number of speakers:**
```bash
DIARIZATION_MIN_SPEAKERS=1
DIARIZATION_MAX_SPEAKERS=10
```

| Use Case | MIN | MAX | Notes |
|----------|-----|-----|-------|
| 1-on-1 Interview | 2 | 2 | Most accurate for known 2-person conversations |
| Small Meeting | 2 | 6 | Typical team meetings |
| Large Meeting | 2 | 10 | Conferences, group discussions |
| Unknown | 1 | 10 | Let pyannote decide automatically |

**Tip:** Setting `MAX_SPEAKERS` slightly higher than expected is better than too low - pyannote will only detect speakers it actually hears.

## Usage

```bash
# Place audio files in data/audio/
python main.py
```

Supported formats: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.webm`

## Output

Results saved to `results/` directory:
- `{filename}_complete.json` - Full metadata, transcription, diarization, speaker stats
- `{filename}_transcript.txt` - Human-readable transcript with speaker labels
- `{filename}_statistics.json` - Processing metrics and speaker statistics

## Architecture

```
Audio File
    |
    v
[Duration Check]
    |
    +---> Short (<10 min): Direct transcription
    |
    +---> Long (>10 min): Intelligent Chunking
              |
              v
         [Chunk at silence boundaries]
              |
              v
         [Transcribe each chunk]
              |
              v
         [Stitch transcriptions]
    |
    v
[Speaker Diarization]
    |
    +---> Short (<20 min): Direct diarization
    |
    +---> Long (>20 min): Chunked with 60s overlap
              |
              v
         [Merge & remap speakers]
    |
    v
[Align text with speakers]
    |
    v
[Save results]
```

## Troubleshooting

**HuggingFace 403 Error:** Request access to both pyannote models on HuggingFace

**CUDA OOM:**
- Reduce `ASR_BATCH_SIZE` to 2
- Reduce `CHUNK_DURATION_SECONDS` to 300 (5 min)
- Set `NUM_WORKERS=1`

**Infinite loop in chunking:** Fixed in this version with max iteration limit

## Credits

Based on [maaz328/speech-diarization-transcription](https://github.com/maaz328/speech-diarization-transcription) with significant modifications for long audio support and memory optimization.

## License

MIT
