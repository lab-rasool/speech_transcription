#!/usr/bin/env python3
"""
Speaker Diarization Service using pyannote.audio
"""

#!/usr/bin/env python3
"""
Speaker Diarization Service using pyannote.audio - FIXED for long audio files
"""

#!/usr/bin/env python3
"""
Speaker Diarization Service using pyannote.audio - FIXED for long audio files
"""

import os
from typing import List, Dict, Any, Optional
import torch
from loguru import logger
from dataclasses import dataclass
import time
import librosa
import numpy as np
import tempfile
import soundfile as sf

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Annotation
    PYANNOTE_AVAILABLE = True
    logger.info("pyannote.audio available")
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("pyannote.audio not available")

@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float
    duration: float

@dataclass
class DiarizationResult:
    segments: List[SpeakerSegment]
    total_speakers: int
    total_duration: float
    processing_time: float

class SpeakerDiarizationService:
    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        huggingface_token: Optional[str] = None,
        min_speakers: int = 2,
        max_speakers: int = 4,
        device: str = "cuda",
        max_duration: float = 20 * 60  # 20 minutes max per chunk
    ):
        self.model_name = model_name
        self.huggingface_token = huggingface_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.device = device
        self.max_duration = max_duration
        self.pipeline = None
        
        logger.info(f"Initializing diarization service with model: {model_name}")
        logger.info(f"Max chunk duration: {max_duration/60:.1f} minutes")
        
        if not PYANNOTE_AVAILABLE:
            raise ImportError("pyannote.audio is required. Install with: pip install pyannote.audio")
        
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load pyannote diarization pipeline"""
        try:
            if self.huggingface_token:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = self.huggingface_token
                logger.info("HuggingFace token set")
            
            logger.info(f"Loading pyannote pipeline: {self.model_name}")
            
            try:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    token=self.huggingface_token
                )
            except Exception as e:
                logger.warning(f"Failed with token, trying without: {e}")
                self.pipeline = Pipeline.from_pretrained(self.model_name)
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
                logger.info("Moved diarization pipeline to GPU")
            
            logger.info("Pyannote diarization pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            if "authentication" in str(e).lower() or "token" in str(e).lower():
                logger.error("This might be due to missing HuggingFace token.")
                logger.error("Get your token from: https://huggingface.co/settings/tokens")
                logger.error("Then set HUGGINGFACE_TOKEN in your .env file")
            raise
    
    def _chunk_audio_for_diarization(self, audio_path: str) -> List[Dict]:
        """Split long audio into overlapping chunks for diarization"""
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / sr
        
        if duration <= self.max_duration:
            return [{"path": audio_path, "start": 0, "end": duration, "is_chunk": False}]
        
        # Create overlapping chunks
        chunks = []
        overlap = 60.0  # Increased to 60 seconds for better speaker continuity
        chunk_duration = self.max_duration
        start = 0
        chunk_id = 0
        
        while start < duration:
            end = min(start + chunk_duration, duration)
            
            # Extract chunk
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            chunk_audio = audio[start_sample:end_sample]
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav', 
                delete=False,
                prefix=f'diarization_chunk_{chunk_id}_'
            )
            sf.write(temp_file.name, chunk_audio, sr)
            temp_file.close()
            
            chunks.append({
                "path": temp_file.name,
                "start": start,
                "end": end,
                "is_chunk": True,
                "chunk_id": chunk_id
            })
            
            chunk_id += 1
            start = end - overlap  # Move forward with overlap
            
            if end >= duration:
                break
        
        logger.info(f"Split {duration/60:.1f}min audio into {len(chunks)} diarization chunks")
        return chunks
    
    def _merge_diarization_chunks(self, chunk_results: List[Dict]) -> List[SpeakerSegment]:
        """Merge diarization results from multiple chunks"""
        if len(chunk_results) == 1:
            return chunk_results[0]["segments"]
        
        # Collect all segments with time offsets
        all_segments = []
        speaker_mapping = {}  # Map chunk-local speakers to global speakers
        global_speaker_id = 0
        
        for chunk_result in chunk_results:
            chunk_start = chunk_result["chunk_start"]
            
            for segment in chunk_result["segments"]:
                # Adjust timing
                adjusted_segment = SpeakerSegment(
                    speaker=segment.speaker,  # Will remap later
                    start=segment.start + chunk_start,
                    end=segment.end + chunk_start,
                    duration=segment.duration
                )
                all_segments.append(adjusted_segment)
        
        # Sort by start time
        all_segments.sort(key=lambda x: x.start)
        
        # Simple speaker remapping based on temporal consistency
        # For 2 speakers, this should work well
        merged_segments = self._remap_speakers(all_segments)
        
        return merged_segments
    
    def _remap_speakers(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Remap speaker labels for consistency across chunks - IMPROVED"""
        if not segments:
            return segments
        
        # Strategy: Use temporal patterns to identify consistent speakers
        # Assumption: In interviews, one speaker (interviewer) speaks first and more frequently
        
        # Group segments by original speaker label
        speaker_segments = {}
        for seg in segments:
            if seg.speaker not in speaker_segments:
                speaker_segments[seg.speaker] = []
            speaker_segments[seg.speaker].append(seg)
        
        # Sort by who speaks first (likely the interviewer)
        speaker_order = []
        for speaker, segs in speaker_segments.items():
            first_appearance = min(seg.start for seg in segs)
            total_time = sum(seg.duration for seg in segs)
            segment_count = len(segs)
            
            speaker_order.append({
                'label': speaker,
                'first_appearance': first_appearance,
                'total_time': total_time,
                'segment_count': segment_count
            })
        
        # Sort by first appearance time (who spoke first)
        speaker_order.sort(key=lambda x: x['first_appearance'])
        
        # Create mapping: first speaker -> SPEAKER_00, second -> SPEAKER_01
        speaker_map = {}
        for idx, speaker_info in enumerate(speaker_order[:2]):  # Only take first 2 speakers
            speaker_map[speaker_info['label']] = f"SPEAKER_{idx:02d}"
        
        logger.info(f"Speaker remapping: {len(speaker_map)} speakers identified")
        for old_label, new_label in speaker_map.items():
            logger.info(f"  {old_label} -> {new_label}")
        
        # Apply mapping
        remapped = []
        for seg in segments:
            new_label = speaker_map.get(seg.speaker, seg.speaker)
            remapped.append(SpeakerSegment(
                speaker=new_label,
                start=seg.start,
                end=seg.end,
                duration=seg.duration
            ))
        
        return remapped
    
    def diarize_audio(self, audio_path: str) -> DiarizationResult:
        """Perform speaker diarization on audio file (handles long files)"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting diarization for: {audio_path}")
            
            # Check duration and chunk if needed
            chunks = self._chunk_audio_for_diarization(audio_path)
            
            chunk_results = []
            temp_files = []
            
            for chunk in chunks:
                logger.info(f"Processing diarization chunk {chunk.get('chunk_id', 0)}: "
                          f"{chunk['start']/60:.1f}-{chunk['end']/60:.1f} min")
                
                # Run diarization on chunk with timeout protection
                try:
                    diarization = self.pipeline(
                        chunk['path'],
                        min_speakers=self.min_speakers,
                        max_speakers=self.max_speakers
                    )
                    
                    # Extract segments
                    segments = []
                    if hasattr(diarization, 'speaker_diarization'):
                        annotation = diarization.speaker_diarization
                    elif hasattr(diarization, 'itertracks'):
                        annotation = diarization
                    else:
                        raise ValueError(f"Unknown diarization output format: {type(diarization)}")
                    
                    for segment, _, speaker in annotation.itertracks(yield_label=True):
                        segments.append(SpeakerSegment(
                            speaker=speaker,
                            start=segment.start,
                            end=segment.end,
                            duration=segment.duration
                        ))
                    
                    chunk_results.append({
                        "segments": segments,
                        "chunk_start": chunk['start'],
                        "chunk_end": chunk['end']
                    })
                    
                    # Clean up GPU memory
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Chunk diarization failed: {e}")
                    raise
                
                # Track temp files for cleanup
                if chunk['is_chunk']:
                    temp_files.append(chunk['path'])
            
            # Merge chunks if multiple
            if len(chunk_results) > 1:
                logger.info("Merging diarization chunks...")
                segments = self._merge_diarization_chunks(chunk_results)
            else:
                segments = chunk_results[0]["segments"]
            
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            processing_time = time.time() - start_time
            total_duration = max(seg.end for seg in segments) if segments else 0
            total_speakers = len(set(seg.speaker for seg in segments))
            
            logger.info(f"Diarization completed: {total_speakers} speakers, {len(segments)} segments")
            logger.info(f"Processing time: {processing_time:.2f}s")
            
            return DiarizationResult(
                segments=segments,
                total_speakers=total_speakers,
                total_duration=total_duration,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Diarization failed for {audio_path}: {e}")
            raise
    
    def align_text_with_speakers(
        self,
        transcription: str,
        diarization_result: DiarizationResult,
        audio_duration: float
    ) -> List[Dict[str, Any]]:
        """Align transcription text with speaker segments"""
        
        if isinstance(transcription, list):
            transcription = ' '.join(transcription) if transcription else ''
        
        import re
        sentences = re.split(r'[.!?]+', transcription)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        chars_per_second = len(transcription) / audio_duration
        
        aligned_segments = []
        current_time = 0.0
        
        for sentence in sentences:
            sentence_duration = len(sentence) / chars_per_second
            sentence_end = current_time + sentence_duration
            
            best_speaker = "SPEAKER_00"
            max_overlap = 0.0
            
            for segment in diarization_result.segments:
                overlap_start = max(current_time, segment.start)
                overlap_end = min(sentence_end, segment.end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = segment.speaker
            
            aligned_segments.append({
                "speaker": best_speaker,
                "start": round(current_time, 2),
                "end": round(sentence_end, 2),
                "duration": round(sentence_duration, 2),
                "text": sentence
            })
            
            current_time = sentence_end
        
        return aligned_segments