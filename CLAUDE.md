# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech transcription system combining NVIDIA NeMo ASR with pyannote.audio speaker diarization for multi-speaker audio processing. Production-ready with chunking for long audio files.

## Key Commands

```bash
# Run full pipeline on all audio files
python main.py

# Test single file processing
python test_single_file.py

# Test diarization only
python test_diarization_only.py
```

## Architecture

### Processing Pipeline
```
Audio → Chunking Decision → Transcription → Diarization → Alignment → Output
```

### Core Services

**ASR Service (`src/asr_service.py`)**
- Handles NeMo model loading and transcription
- Manages chunking for files >24 minutes via `ChunkingManager`
- Returns `TranscriptionResult` with text as string (extracts from Hypothesis object)
- Critical: NeMo returns `Hypothesis` objects, must extract `.text` attribute

**Diarization Service (`src/diarization_service.py`)**
- Uses pyannote.audio 3.1 pipeline
- ALWAYS runs on full audio (not chunks) for speaker consistency
- Returns `DiarizationResult` with speaker segments
- Critical: For pyannote 3.x, access `diarization.speaker_diarization` attribute for iteration

**Chunking Utils (`src/chunking_utils.py`)**
- `IntelligentChunker`: Detects silence boundaries for optimal splits
- `TranscriptionStitcher`: Removes duplicates at chunk boundaries
- Only used for audio >24 minutes (configured via CHUNK_DURATION_SECONDS)

### Main Entry Point (`main.py`)
- `AudioProcessor` class handles complete pipeline per file
- Worker-based parallel processing (configurable via NUM_WORKERS)
- Saves 3 output files per audio: complete.json, transcript.txt, statistics.json

## Critical Implementation Details

### NeMo Transcription Output
```python
# NeMo returns Hypothesis objects, not strings
if hasattr(transcripts[0], 'text'):
    text = transcripts[0].text
```

### Pyannote 3.x API Compatibility
```python
# DiarizeOutput is not directly iterable
if hasattr(diarization, 'speaker_diarization'):
    annotation = diarization.speaker_diarization
for segment, _, speaker in annotation.itertracks(yield_label=True):
    # Process segments
```

### Configuration (.env)
```bash
HUGGINGFACE_TOKEN=hf_xxx  # Required for pyannote
CHUNK_DURATION_SECONDS=1440  # 24 minutes
NUM_WORKERS=2  # Parallel processing
ASR_DEVICE=cuda  # or cpu
```

## Required Model Access
Users must request access on HuggingFace:
- pyannote/speaker-diarization-3.1
- pyannote/segmentation-3.0
