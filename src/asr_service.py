#!/usr/bin/env python3
"""
Standalone ASR Service using NVIDIA NeMo Parakeet models - MEMORY OPTIMIZED
"""

import torch
from typing import Optional, List, Dict, Any, Tuple
import librosa
import soundfile as sf
import tempfile
import os
import numpy as np
from loguru import logger
from dataclasses import dataclass
import time
try:
    from chunking_utils import ChunkingManager, AudioChunk
except ImportError:
    from .chunking_utils import ChunkingManager, AudioChunk

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
    logger.info("NeMo toolkit available")
except ImportError:
    NEMO_AVAILABLE = False
    logger.warning("NeMo toolkit not available")

@dataclass
class TranscriptionResult:
    text: str
    duration: float
    rtf: float
    confidence: Optional[float] = None
    chunks: int = 1
    audio_quality: Optional[Dict[str, Any]] = None

class NeMoASRService:
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
        device: str = "cuda",
        batch_size: int = 4,  # REDUCED from 8 to prevent OOM
        max_chunk_duration: float = 10 * 60,  # REDUCED to 10 minutes from 24
        min_audio_duration: float = 5.0
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_chunk_duration = max_chunk_duration
        self.min_audio_duration = min_audio_duration
        self.model = None

        # Initialize chunking manager with safer chunk size
        self.chunking_manager = ChunkingManager(
            max_chunk_duration=max_chunk_duration,
            overlap_duration=5.0
        )

        logger.info(f"Initializing ASR service with model: {model_name}")
        logger.info(f"Device: {device}, Batch size: {batch_size}")
        logger.info(f"Max chunk duration: {max_chunk_duration/60:.1f} minutes")

        if not NEMO_AVAILABLE:
            raise ImportError("NeMo toolkit is required. Install with: pip install nemo_toolkit[asr]")

        self._load_nemo_model()
        
    def _load_nemo_model(self):
        """Load NeMo Parakeet model"""
        try:
            logger.info(f"Loading NeMo Parakeet model: {self.model_name}")
            
            # Load the model
            self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(self.model_name)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Enable mixed precision for H100/A100 GPUs
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.half()
                logger.info("Enabled FP16 precision for GPU")
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"NeMo Parakeet model {self.model_name} loaded successfully")
            
            # Warmup
            self._warmup_model()
            
        except Exception as e:
            logger.error(f"Failed to load NeMo model: {e}")
            raise
    
    def _warmup_model(self):
        """Warmup the model with dummy audio"""
        logger.info("Warming up NeMo model...")
        try:
            # Create dummy audio (1 second of silence)
            dummy_audio = torch.zeros(16000, dtype=torch.float32)
            
            with torch.no_grad():
                if self.device == "cuda":
                    dummy_audio = dummy_audio.cuda()
                    if hasattr(self.model, 'half'):
                        dummy_audio = dummy_audio.half()
                
                # Run inference
                _ = self.model.transcribe([dummy_audio.cpu().numpy()])
            
            logger.info("NeMo model warmup completed")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def transcribe_chunk(self, chunk: AudioChunk) -> Dict:
        """Transcribe a single audio chunk"""
        try:
            # Save chunk to temporary file for NeMo
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, chunk.audio_data, 16000)
                temp_path = tmp_file.name

            # Transcribe with NeMo
            with torch.no_grad():
                transcriptions = self.model.transcribe([temp_path])
                # Extract text from transcription result
                if transcriptions:
                    # Handle Hypothesis object or direct text
                    if hasattr(transcriptions[0], 'text'):
                        text = transcriptions[0].text
                    elif isinstance(transcriptions[0], list):
                        text = ' '.join(transcriptions[0]) if transcriptions[0] else ""
                    else:
                        text = str(transcriptions[0])
                else:
                    text = ""

            # Clean up temp file
            os.unlink(temp_path)
            
            # Clear GPU memory after each chunk
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return {
                "text": text,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "duration": chunk.duration,
                "chunk_id": chunk.chunk_id
            }
        except Exception as e:
            logger.error(f"Error transcribing chunk {chunk.chunk_id}: {e}")
            # Clear GPU memory on error too
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return {
                "text": "",
                "error": str(e),
                "chunk_id": chunk.chunk_id
            }

    def transcribe_chunks_batch(self, chunks: List[AudioChunk]) -> List[Dict]:
        """Transcribe multiple chunks with batching and memory management"""
        results = []

        # Process ONE chunk at a time to avoid OOM
        # Even though we have batch_size, we process individually for safety
        for chunk in chunks:
            logger.info(f"Transcribing chunk {chunk.chunk_id}/{len(chunks)}: "
                       f"{chunk.start_time:.1f}s-{chunk.end_time:.1f}s ({chunk.duration:.1f}s)")
            
            try:
                result = self.transcribe_chunk(chunk)
                results.append(result)
                
                # Log progress
                if result.get('text'):
                    logger.info(f"  ✓ Chunk {chunk.chunk_id} transcribed: {len(result['text'])} chars")
                else:
                    logger.warning(f"  ⚠ Chunk {chunk.chunk_id} produced empty text")
                    
            except Exception as e:
                logger.error(f"Failed to transcribe chunk {chunk.chunk_id}: {e}")
                results.append({
                    "text": "",
                    "error": str(e),
                    "chunk_id": chunk.chunk_id,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "duration": chunk.duration
                })
            
            # Force GPU memory cleanup after each chunk
            if self.device == "cuda":
                torch.cuda.empty_cache()

        logger.info(f"Completed transcription of {len(results)} chunks")
        return results

    async def transcribe_audio(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio file with intelligent chunking for long files"""
        start_time = time.time()

        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(audio) / sr

            logger.info(f"Loaded audio: duration={duration:.2f}s, sample_rate={sr}Hz")

            # Check minimum duration
            if duration < self.min_audio_duration:
                raise ValueError(f"Audio too short: {duration:.2f}s < {self.min_audio_duration}s")

            # Check if chunking is needed
            if duration > self.max_chunk_duration:
                logger.info(f"Long audio ({duration/60:.1f} min), using intelligent chunking")

                # Process with chunking
                chunks, stitch_function = self.chunking_manager.process_long_audio(audio, sr)
                logger.info(f"Created {len(chunks)} chunks")

                # Transcribe chunks ONE AT A TIME
                chunk_results = self.transcribe_chunks_batch(chunks)

                # Stitch results
                stitched_result = stitch_function(chunk_results)
                text = stitched_result.get('text', '')
                chunks_processed = stitched_result.get('chunks_processed', len(chunks))
                
                if not text:
                    logger.error("Stitching produced empty text!")
                    logger.error(f"Chunk results: {[len(r.get('text', '')) for r in chunk_results]}")
                
            else:
                # Single chunk processing for short audio
                logger.info(f"Short audio ({duration/60:.1f} min), single chunk processing")

                # Transcribe directly
                with torch.no_grad():
                    transcripts = self.model.transcribe([audio_path])
                    # Extract text from transcription result
                    if transcripts:
                        # Handle Hypothesis object or direct text
                        if hasattr(transcripts[0], 'text'):
                            text = transcripts[0].text
                        elif isinstance(transcripts[0], list):
                            text = ' '.join(transcripts[0]) if transcripts[0] else ""
                        else:
                            text = str(transcripts[0])
                    else:
                        text = ""
                chunks_processed = 1
                
                # Clear GPU memory
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            processing_time = time.time() - start_time
            rtf = processing_time / duration

            logger.info(f"Transcription completed: duration={duration:.2f}s, RTF={rtf:.3f}, chunks={chunks_processed}")
            logger.info(f"Text length: {len(text)} characters")

            return TranscriptionResult(
                text=text,
                duration=duration,
                rtf=rtf,
                confidence=None,  # NeMo doesn't provide confidence by default
                chunks=chunks_processed,
                audio_quality={"sample_rate": sr}
            )

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise