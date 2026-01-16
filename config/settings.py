from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import torch
import os

class TranscriptionConfig(BaseSettings):
    # ASR Configuration
    asr_model_name: str = Field(default="nvidia/parakeet-tdt-0.6b-v2", description="NeMo ASR model name")
    asr_device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu", description="Processing device")
    asr_batch_size: int = Field(default=4, description="ASR batch size - REDUCED from 8 to prevent OOM")
    asr_gpu_memory_fraction: float = Field(default=0.6, description="GPU memory fraction for ASR")
    
    # Diarization Configuration
    diarization_model: str = Field(default="pyannote/speaker-diarization-3.1", description="Pyannote diarization model")
    diarization_min_speakers: int = Field(default=2, description="Minimum number of speakers")
    diarization_max_speakers: int = Field(default=4, description="Maximum number of speakers")
    
    # HuggingFace Token
    huggingface_token: Optional[str] = Field(default=None, description="HuggingFace API token")
    
    # Processing Configuration
    chunk_duration_seconds: float = Field(
        default=600,  # CHANGED: 10 minutes (600s) instead of 24 minutes (1440s)
        description="Audio chunk duration in seconds - reduced to prevent GPU OOM"
    )
    min_audio_duration_seconds: float = Field(default=5.0, description="Minimum audio duration to process")
    num_workers: int = Field(
        default=1,  # CHANGED: 1 worker instead of 2 to reduce GPU memory pressure
        description="Number of processing workers - reduced to prevent GPU contention"
    )
    
    # Paths
    audio_input_path: str = Field(default="./data/audio/", description="Input audio directory")
    results_output_path: str = Field(default="./results/", description="Output results directory")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

# Global settings instance
_settings = None

def get_settings() -> TranscriptionConfig:
    global _settings
    if _settings is None:
        _settings = TranscriptionConfig()
        
        # Log the configuration on first load
        import logging
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("Transcription Configuration Loaded:")
        logger.info(f"  Chunk Duration: {_settings.chunk_duration_seconds/60:.1f} minutes")
        logger.info(f"  ASR Batch Size: {_settings.asr_batch_size}")
        logger.info(f"  Workers: {_settings.num_workers}")
        logger.info(f"  Device: {_settings.asr_device}")
        logger.info("=" * 60)
        
    return _settings