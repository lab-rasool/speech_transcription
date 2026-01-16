#!/usr/bin/env python3
"""
Main entry point for batch audio transcription with speaker diarization.
Processes audio files with intelligent chunking for long recordings.

Architecture:
- Workers process audio files in parallel
- Each worker handles one file completely:
  - Short audio (<24 min): Direct transcription
  - Long audio (>24 min): Intelligent chunking → batch transcription → stitching
- Diarization always runs on full audio (not chunks) for speaker consistency
- Results combine full transcription with speaker-aligned segments
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import time
import librosa
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from asr_service import NeMoASRService
from diarization_service import SpeakerDiarizationService

# Add parent directory to path for config
sys.path.append(os.path.dirname(__file__))
from config.settings import get_settings


class AudioProcessor:
    """Process a single audio file with transcription and diarization"""

    def __init__(self, settings):
        self.settings = settings
        self.asr_service = None
        self.diarization_service = None

    def initialize_services(self):
        """Initialize ASR and diarization services (called in worker thread)"""
        if not self.asr_service:
            logger.info(f"Initializing ASR service...")
            self.asr_service = NeMoASRService(
                model_name=self.settings.asr_model_name,
                device=self.settings.asr_device,
                batch_size=self.settings.asr_batch_size,
                max_chunk_duration=self.settings.chunk_duration_seconds
            )

        if not self.diarization_service:
            logger.info(f"Initializing diarization service...")
            self.diarization_service = SpeakerDiarizationService(
                model_name=self.settings.diarization_model,
                huggingface_token=self.settings.huggingface_token,
                min_speakers=self.settings.diarization_min_speakers,
                max_speakers=self.settings.diarization_max_speakers,
                device=self.settings.asr_device
            )

    async def process_audio_file(self, audio_path: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Process a single audio file:
        1. Load and check duration
        2. Transcribe (with chunking if needed)
        3. Diarize the full audio
        4. Align transcription with speakers
        """

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {audio_path.name}")
        logger.info(f"{'='*60}")

        file_start_time = time.time()

        try:
            # Ensure services are initialized
            self.initialize_services()

            # Load audio once to check duration
            audio_data, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            duration = len(audio_data) / sr
            duration_minutes = duration / 60

            logger.info(f"Audio duration: {duration_minutes:.1f} minutes")

            # Step 1: Transcription (with automatic chunking if needed)
            if duration > self.settings.chunk_duration_seconds:
                logger.info(f"→ Long audio detected ({duration_minutes:.1f} min > {self.settings.chunk_duration_seconds/60:.1f} min)")
                logger.info(f"→ Using intelligent chunking...")

                # The ASR service handles chunking internally
                transcription_result = await self.asr_service.transcribe_audio(str(audio_path))

                logger.info(f"→ Transcribed {transcription_result.chunks} chunks successfully")
            else:
                logger.info(f"→ Short audio ({duration_minutes:.1f} min), single-pass transcription")
                transcription_result = await self.asr_service.transcribe_audio(str(audio_path))

            # Step 2: Speaker Diarization (always on full audio)
            logger.info(f"→ Running speaker diarization on full audio...")
            diarization_result = self.diarization_service.diarize_audio(str(audio_path))
            logger.info(f"→ Identified {diarization_result.total_speakers} speakers")

            # Step 3: Align transcription with speakers
            logger.info(f"→ Aligning transcription with speaker segments...")
            aligned_segments = self.diarization_service.align_text_with_speakers(
                transcription_result.text,
                diarization_result,
                transcription_result.duration
            )

            # Calculate speaker statistics
            speaker_stats = self._calculate_speaker_stats(diarization_result.segments)

            # Calculate processing metrics
            file_processing_time = time.time() - file_start_time

            # Prepare complete results
            result_data = {
                "file_name": audio_path.name,
                "file_path": str(audio_path),
                "processed_at": datetime.now().isoformat(),
                "audio_info": {
                    "duration_seconds": round(duration, 2),
                    "duration_minutes": round(duration_minutes, 2),
                    "chunks_processed": transcription_result.chunks,
                    "rtf": round(transcription_result.rtf, 3),
                    "processing_time": round(file_processing_time, 2)
                },
                "transcription": {
                    "full_text": transcription_result.text,
                    "word_count": len(transcription_result.text.split()),
                    "character_count": len(transcription_result.text)
                },
                "diarization": {
                    "total_speakers": diarization_result.total_speakers,
                    "total_segments": len(diarization_result.segments),
                    "speaker_statistics": speaker_stats
                },
                "speaker_segments": aligned_segments
            }            

            # Save results
            output_files = self._save_results(audio_path, result_data, output_dir)

            logger.info(f"✅ Processing completed in {file_processing_time:.1f}s")
            logger.info(f"   → Transcription: {transcription_result.chunks} chunks, RTF: {transcription_result.rtf:.3f}")
            logger.info(f"   → Diarization: {diarization_result.total_speakers} speakers")
            logger.info(f"   → Output: {', '.join([f.name for f in output_files])}")

            return {
                "status": "success",
                "file": audio_path.name,
                "duration_minutes": round(duration_minutes, 2),
                "chunks": transcription_result.chunks,
                "speakers": diarization_result.total_speakers,
                "rtf": round(transcription_result.rtf, 3),
                "processing_time": round(file_processing_time, 2),
                "output_files": [str(f) for f in output_files]
            }

        except Exception as e:
            logger.error(f"❌ Failed to process {audio_path.name}: {e}")
            import traceback
            traceback.print_exc()

            return {
                "status": "failed",
                "file": audio_path.name,
                "error": str(e)
            }
        finally:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _calculate_speaker_stats(self, segments) -> Dict:
        """Calculate speaking time statistics for each speaker"""
        stats = {}
        total_time = sum(seg.duration for seg in segments)

        for segment in segments:
            if segment.speaker not in stats:
                stats[segment.speaker] = {
                    "total_time": 0,
                    "segment_count": 0,
                    "average_segment_duration": 0
                }
            stats[segment.speaker]["total_time"] += segment.duration
            stats[segment.speaker]["segment_count"] += 1

        # Calculate percentages and averages
        for speaker in stats:
            stats[speaker]["percentage"] = round(
                (stats[speaker]["total_time"] / total_time * 100) if total_time > 0 else 0, 2
            )
            stats[speaker]["average_segment_duration"] = round(
                stats[speaker]["total_time"] / stats[speaker]["segment_count"], 2
            )
            stats[speaker]["total_time"] = round(stats[speaker]["total_time"], 2)

        return stats

    def _save_results(self, audio_path: Path, result_data: Dict, output_dir: Path) -> List[Path]:
        """Save results in multiple formats"""
        output_files = []

        # Save JSON (complete data)
        json_file = output_dir / f"{audio_path.stem}_complete.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        output_files.append(json_file)

        # Save plain text transcript
        text_file = output_dir / f"{audio_path.stem}_transcript.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"FILE: {audio_path.name}\n")
            f.write(f"DATE: {result_data['processed_at']}\n")
            f.write(f"DURATION: {result_data['audio_info']['duration_minutes']:.1f} minutes\n")
            f.write(f"SPEAKERS: {result_data['diarization']['total_speakers']}\n")
            f.write(f"CHUNKS PROCESSED: {result_data['audio_info']['chunks_processed']}\n")
            f.write("=" * 80 + "\n\n")
            f.write("FULL TRANSCRIPTION:\n")
            f.write("-" * 80 + "\n")
            f.write(result_data['transcription']['full_text'])
            f.write("\n\n" + "=" * 80 + "\n\n")
            f.write("SPEAKER-SEGMENTED TRANSCRIPTION:\n")
            f.write("-" * 80 + "\n\n")

            for segment in result_data['speaker_segments']:
                f.write(f"[{segment['speaker']} @ {segment['start']:.1f}s - {segment['end']:.1f}s]\n")
                f.write(f"{segment['text']}\n\n")
        output_files.append(text_file)

        # Save speaker statistics
        stats_file = output_dir / f"{audio_path.stem}_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "file": audio_path.name,
                "audio_duration_minutes": result_data['audio_info']['duration_minutes'],
                "processing_time_seconds": result_data['audio_info']['processing_time'],
                "rtf": result_data['audio_info']['rtf'],
                "chunks_processed": result_data['audio_info']['chunks_processed'],
                "speaker_statistics": result_data['diarization']['speaker_statistics']
            }, f, indent=2)
        output_files.append(stats_file)

        return output_files


async def process_with_workers(audio_files: List[Path], num_workers: int, settings):
    """
    Process audio files using worker pool.
    Each worker processes one file at a time completely.
    """

    output_dir = Path(settings.results_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create processor instance (will be shared)
    processor = AudioProcessor(settings)

    # Process files in batches based on number of workers
    results = []
    total_files = len(audio_files)

    for i in range(0, total_files, num_workers):
        batch = audio_files[i:i + num_workers]
        batch_num = (i // num_workers) + 1
        total_batches = (total_files + num_workers - 1) // num_workers

        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH {batch_num}/{total_batches} - Processing {len(batch)} file(s)")
        logger.info(f"{'='*80}")

        # Create tasks for this batch
        tasks = [
            processor.process_audio_file(audio_file, output_dir)
            for audio_file in batch
        ]

        # Run batch in parallel
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

    return results


async def main():
    """Main entry point"""

    # Load settings
    settings = get_settings()
    num_workers = settings.num_workers

    # Setup directories
    audio_dir = Path(settings.audio_input_path)
    output_dir = Path(settings.results_output_path)

    # Print configuration
    logger.info("=" * 80)
    logger.info("BATCH AUDIO TRANSCRIPTION WITH SPEAKER DIARIZATION")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  → Input directory: {audio_dir}")
    logger.info(f"  → Output directory: {output_dir}")
    logger.info(f"  → Parallel workers: {num_workers}")
    logger.info(f"  → Max chunk duration: {settings.chunk_duration_seconds/60:.1f} minutes")
    logger.info(f"  → Device: {settings.asr_device}")
    logger.info(f"  → ASR model: {settings.asr_model_name}")
    logger.info(f"  → Diarization model: {settings.diarization_model}")
    logger.info("=" * 80)

    # Find audio files
    audio_files = []
    extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg', '*.webm']
    for ext in extensions:
        audio_files.extend(audio_dir.glob(ext))

    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio file(s) to process")

    # Start processing
    start_time = time.time()

    # Process files with workers
    results = await process_with_workers(audio_files, num_workers, settings)

    # Calculate summary
    total_time = time.time() - start_time
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Total processing time: {total_time/60:.1f} minutes")

    if successful:
        total_audio_minutes = sum(r.get("duration_minutes", 0) for r in successful)
        total_chunks = sum(r.get("chunks", 1) for r in successful)
        avg_rtf = sum(r.get("rtf", 0) for r in successful) / len(successful) if successful else 0

        logger.info(f"\nProcessing Statistics:")
        logger.info(f"  → Total audio processed: {total_audio_minutes:.1f} minutes")
        logger.info(f"  → Total chunks processed: {total_chunks}")
        logger.info(f"  → Average RTF: {avg_rtf:.3f}")
        logger.info(f"  → Processing speed: {total_audio_minutes/(total_time/60):.1f}x realtime")

        logger.info(f"\nSuccessfully processed files:")
        for r in successful:
            chunks_info = f" ({r['chunks']} chunks)" if r.get('chunks', 1) > 1 else ""
            logger.info(f"  ✅ {r['file']}")
            logger.info(f"     → Duration: {r['duration_minutes']:.1f} min{chunks_info}")
            logger.info(f"     → Speakers: {r['speakers']}, RTF: {r['rtf']:.3f}")

    if failed:
        logger.info(f"\nFailed files:")
        for r in failed:
            logger.info(f"  ❌ {r['file']}: {r['error']}")

    logger.info(f"\n{'='*80}")
    logger.info(f"All results saved to: {output_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    logger.add("logs/transcription_{time}.log", rotation="1 day", level="DEBUG")

    # Run the main async function
    asyncio.run(main())