import librosa
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from loguru import logger
import re

@dataclass
class AudioChunk:
    audio_data: np.ndarray
    start_time: float
    end_time: float
    duration: float
    chunk_id: int
    overlap_start: float = 0.0
    overlap_end: float = 0.0
    speaker_embedding: Optional[np.ndarray] = None

@dataclass
class TranscriptionChunk:
    text: str
    start_time: float
    end_time: float
    chunk_id: int
    confidence: Optional[float] = None
    speaker_id: Optional[int] = None
    word_timestamps: Optional[List[Dict]] = None

class IntelligentChunker:
    def __init__(self,
                 max_chunk_duration: float = 24 * 60,  # 24 minutes
                 overlap_duration: float = 5.0,  # 5 seconds overlap
                 min_chunk_duration: float = 30.0,  # 30 seconds minimum
                 voice_activity_threshold: float = 0.5):

        self.max_chunk_duration = max_chunk_duration
        self.overlap_duration = overlap_duration
        self.min_chunk_duration = min_chunk_duration
        self.voice_activity_threshold = voice_activity_threshold

    def detect_speech_boundaries(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Detect speech segments using voice activity detection and energy analysis"""

        # Use librosa to detect non-silent intervals
        intervals = librosa.effects.split(audio,
                                         top_db=20,
                                         frame_length=2048,
                                         hop_length=512)

        if len(intervals) == 0:
            return [(0, len(audio) / sr)]

        # Convert sample indices to time
        speech_segments = []
        for start_sample, end_sample in intervals:
            start_time = start_sample / sr
            end_time = end_sample / sr

            # Merge nearby segments (within 0.5 seconds)
            if speech_segments and start_time - speech_segments[-1][1] < 0.5:
                # Extend the previous segment
                speech_segments[-1] = (speech_segments[-1][0], end_time)
            else:
                speech_segments.append((start_time, end_time))

        return speech_segments

    def find_optimal_split_points(self, audio: np.ndarray, sr: int,
                                 speech_segments: List[Tuple[float, float]]) -> List[float]:
        """Find optimal points to split audio based on silence and speech patterns - FIXED"""

        total_duration = len(audio) / sr
        if total_duration <= self.max_chunk_duration:
            return [total_duration]

        split_points = []
        current_time = 0
        
        logger.info(f"Finding split points for {total_duration:.1f}s audio with {len(speech_segments)} segments")

        iteration = 0
        max_iterations = 100  # Safety limit to prevent infinite loops
        
        while current_time < total_duration and iteration < max_iterations:
            iteration += 1
            
            target_end_time = min(current_time + self.max_chunk_duration, total_duration)
            
            # Find the best split point within the target window
            best_split = self._find_best_split_in_window(
                speech_segments, current_time, target_end_time
            )

            if best_split is None or best_split <= current_time:
                # Fallback: use target end time if no good split found
                best_split = target_end_time
                logger.debug(f"No optimal split found, using target end time: {best_split:.1f}s")

            split_points.append(best_split)
            
            # Move forward with overlap, ensuring we make progress
            next_start = best_split - self.overlap_duration
            
            # CRITICAL FIX: Ensure we always move forward
            if next_start <= current_time:
                next_start = best_split  # Skip overlap if it would cause us to go backwards
                logger.debug(f"Skipping overlap to ensure forward progress")
            
            current_time = next_start
            
            logger.debug(f"Iteration {iteration}: Split at {best_split:.1f}s, next start: {current_time:.1f}s")
            
            # If we're within one chunk duration of the end, just add the final point
            if total_duration - current_time <= self.max_chunk_duration:
                if split_points[-1] < total_duration:  # Don't add duplicate
                    split_points.append(total_duration)
                break
        
        if iteration >= max_iterations:
            logger.warning(f"Hit max iterations ({max_iterations}), forcing final split")
            if split_points[-1] < total_duration:
                split_points.append(total_duration)

        logger.info(f"Created {len(split_points)} split points")
        return split_points

    def _find_best_split_in_window(self, speech_segments: List[Tuple[float, float]],
                                  start_time: float, end_time: float) -> Optional[float]:
        """Find the best split point within a time window - OPTIMIZED"""

        # Look for silence gaps near the end of the window
        # Search in last 2 minutes of window for efficiency
        search_start = max(start_time, end_time - 120)

        best_split = None
        longest_gap = 0

        # Only search through relevant segments (optimization for 770+ segments)
        relevant_segments = [
            seg for seg in speech_segments 
            if seg[0] <= end_time and seg[1] >= search_start
        ]

        for i in range(len(relevant_segments) - 1):
            segment_end = relevant_segments[i][1]
            next_segment_start = relevant_segments[i + 1][0]

            # Check if this gap is in our search window
            if search_start <= segment_end <= end_time:
                gap_duration = next_segment_start - segment_end

                if gap_duration > longest_gap and gap_duration >= 0.5:  # At least 0.5s silence
                    longest_gap = gap_duration
                    # Split in the middle of the silence
                    best_split = segment_end + gap_duration / 2

        # If no good silence gap found, split at 90% of the window
        if best_split is None:
            best_split = start_time + (end_time - start_time) * 0.9

        return best_split

    def create_chunks(self, audio: np.ndarray, sr: int) -> List[AudioChunk]:
        """Create intelligent audio chunks with optimal split points"""

        total_duration = len(audio) / sr

        if total_duration <= self.max_chunk_duration:
            logger.info(f"Audio duration {total_duration:.1f}s is within max chunk size, no chunking needed")
            return [AudioChunk(
                audio_data=audio,
                start_time=0,
                end_time=total_duration,
                duration=total_duration,
                chunk_id=0
            )]

        logger.info(f"Chunking {total_duration:.1f}s audio (max chunk: {self.max_chunk_duration:.1f}s)")

        # Detect speech segments
        speech_segments = self.detect_speech_boundaries(audio, sr)
        logger.debug(f"Detected {len(speech_segments)} speech segments")

        # Find optimal split points
        split_points = self.find_optimal_split_points(audio, sr, speech_segments)

        # Create chunks with overlaps
        chunks = []
        start_time = 0

        for i, end_time in enumerate(split_points):
            # Calculate actual start/end with overlaps
            actual_start = max(0, start_time - (self.overlap_duration if i > 0 else 0))
            actual_end = min(total_duration, end_time + (self.overlap_duration if i < len(split_points) - 1 else 0))

            start_sample = int(actual_start * sr)
            end_sample = int(actual_end * sr)

            chunk_audio = audio[start_sample:end_sample]
            chunk_duration = len(chunk_audio) / sr

            # Only add chunk if it meets minimum duration
            if chunk_duration >= self.min_chunk_duration or i == len(split_points) - 1:
                chunk = AudioChunk(
                    audio_data=chunk_audio,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    chunk_id=i,
                    overlap_start=start_time - actual_start if i > 0 else 0,
                    overlap_end=actual_end - end_time if i < len(split_points) - 1 else 0
                )

                chunks.append(chunk)
                logger.debug(f"Created chunk {i}: {start_time:.1f}s - {end_time:.1f}s (duration: {chunk_duration:.1f}s)")
            else:
                logger.debug(f"Skipping chunk {i} (too short: {chunk_duration:.1f}s)")

            start_time = end_time

        logger.info(f"Created {len(chunks)} chunks with overlaps")
        
        # Log chunk details
        for chunk in chunks:
            logger.info(f"  Chunk {chunk.chunk_id}: {chunk.start_time:.1f}s-{chunk.end_time:.1f}s ({chunk.duration:.1f}s)")
        
        return chunks

class TranscriptionStitcher:
    def __init__(self, overlap_threshold: float = 0.7):
        self.overlap_threshold = overlap_threshold

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text segments"""

        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def find_overlap_boundary(self, prev_text: str, curr_text: str,
                            overlap_duration: float) -> Tuple[str, str]:
        """Find the best boundary to remove overlapped content"""

        prev_words = prev_text.split()
        curr_words = curr_text.split()

        if len(prev_words) == 0 or len(curr_words) == 0:
            return prev_text, curr_text

        # Estimate words per second (rough approximation)
        words_per_second = 3.0  # Average speaking rate
        overlap_words = int(overlap_duration * words_per_second)

        # Look for the best match in the overlap region
        best_similarity = 0
        best_split_prev = len(prev_words)
        best_split_curr = 0

        # Search in the last part of previous text and first part of current text
        search_window = min(overlap_words * 2, len(prev_words), len(curr_words))

        for i in range(max(0, len(prev_words) - search_window), len(prev_words)):
            for j in range(min(search_window, len(curr_words))):
                prev_segment = " ".join(prev_words[i:])
                curr_segment = " ".join(curr_words[:j+1])

                similarity = self.calculate_text_similarity(prev_segment, curr_segment)

                if similarity > best_similarity and similarity > self.overlap_threshold:
                    best_similarity = similarity
                    best_split_prev = i
                    best_split_curr = j + 1

        # Apply the best split found
        if best_similarity > self.overlap_threshold:
            trimmed_prev = " ".join(prev_words[:best_split_prev])
            trimmed_curr = " ".join(curr_words[best_split_curr:])
        else:
            # Fallback: simple time-based trimming
            trim_words = min(overlap_words // 2, len(prev_words) // 4, len(curr_words) // 4)
            trimmed_prev = " ".join(prev_words[:-trim_words]) if trim_words > 0 else prev_text
            trimmed_curr = " ".join(curr_words[trim_words:]) if trim_words > 0 else curr_text

        return trimmed_prev, trimmed_curr

    def stitch_transcriptions(self, chunks: List[AudioChunk],
                            transcription_results: List[Dict]) -> Dict:
        """Stitch transcribed chunks into coherent text"""

        if not chunks or not transcription_results:
            return {"text": "", "confidence": 0, "chunks_processed": 0}

        if len(chunks) != len(transcription_results):
            logger.warning(f"Chunk count mismatch: {len(chunks)} chunks, {len(transcription_results)} results")

        transcription_chunks = []

        # Convert results to TranscriptionChunk objects
        for i, (chunk, result) in enumerate(zip(chunks, transcription_results)):
            if result.get('error'):
                logger.warning(f"Skipping chunk {i} due to error: {result['error']}")
                continue

            trans_chunk = TranscriptionChunk(
                text=result.get('text', '').strip(),
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                chunk_id=chunk.chunk_id,
                confidence=result.get('confidence')
            )
            transcription_chunks.append(trans_chunk)

        if not transcription_chunks:
            return {"text": "", "confidence": 0, "chunks_processed": 0}

        # Sort chunks by start time
        transcription_chunks.sort(key=lambda x: x.start_time)

        # Stitch chunks together
        stitched_text = transcription_chunks[0].text
        confidences = [transcription_chunks[0].confidence] if transcription_chunks[0].confidence else []

        for i in range(1, len(transcription_chunks)):
            curr_chunk = transcription_chunks[i]
            prev_chunk = transcription_chunks[i-1]

            # Find corresponding audio chunks to get overlap info
            curr_audio_chunk = next((c for c in chunks if c.chunk_id == curr_chunk.chunk_id), None)

            if curr_audio_chunk and curr_audio_chunk.overlap_start > 0:
                # Handle overlap
                trimmed_prev, trimmed_curr = self.find_overlap_boundary(
                    stitched_text, curr_chunk.text, curr_audio_chunk.overlap_start
                )

                # Replace the previous text with trimmed version and add current
                if trimmed_prev != stitched_text:  # Only if we actually trimmed something
                    # Find the last occurrence of the original ending to replace it
                    original_words = stitched_text.split()
                    trimmed_words = trimmed_prev.split()

                    if len(trimmed_words) < len(original_words):
                        stitched_text = trimmed_prev

                if trimmed_curr:
                    stitched_text += " " + trimmed_curr

            else:
                # No overlap, simple concatenation
                if curr_chunk.text:
                    stitched_text += " " + curr_chunk.text

            if curr_chunk.confidence:
                confidences.append(curr_chunk.confidence)

        # Clean up the text
        stitched_text = self._clean_stitched_text(stitched_text)

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        result = {
            "text": stitched_text,
            "confidence": avg_confidence,
            "chunks_processed": len(transcription_chunks),
            "total_chunks": len(chunks),
            "chunk_details": [
                {
                    "chunk_id": chunk.chunk_id,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "text_length": len(chunk.text),
                    "confidence": chunk.confidence
                }
                for chunk in transcription_chunks
            ]
        }

        logger.info(f"Stitched {len(transcription_chunks)} chunks into {len(stitched_text)} characters")
        return result

    def _clean_stitched_text(self, text: str) -> str:
        """Clean up the stitched text"""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove duplicate punctuation
        text = re.sub(r'([.!?])\s*\1+', r'\1', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)

        # Remove duplicate words that might occur at boundaries
        words = text.split()
        cleaned_words = []

        for i, word in enumerate(words):
            # Check for duplicate with previous word
            if i == 0 or word.lower() != words[i-1].lower():
                cleaned_words.append(word)

        text = " ".join(cleaned_words)

        return text.strip()

class ChunkingManager:
    def __init__(self, max_chunk_duration: float = 24 * 60, overlap_duration: float = 5.0):
        self.chunker = IntelligentChunker(
            max_chunk_duration=max_chunk_duration,
            overlap_duration=overlap_duration
        )
        self.stitcher = TranscriptionStitcher()

    def process_long_audio(self, audio: np.ndarray, sr: int) -> Tuple[List[AudioChunk], callable]:
        """Process long audio into chunks and return stitching function"""

        chunks = self.chunker.create_chunks(audio, sr)

        def stitch_results(transcription_results: List[Dict]) -> Dict:
            return self.stitcher.stitch_transcriptions(chunks, transcription_results)

        return chunks, stitch_results