"""Speaker embedding extraction using WeSpeaker ResNet34-LM (ONNX).

Extracts 256-dimensional speaker embeddings from audio segments detected
by VAD.  Long segments are split with a sliding window so that each
window produces its own embedding, improving clustering granularity.
"""
from __future__ import annotations

import os
import io
import tempfile
from pathlib import Path
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
import soundfile as sf

from diarize.utils import SpeechSegment, SubSegment, logger

# ---- CONFIG ----
NUM_WORKERS = 4
BATCH_SIZE = 8

MIN_SEGMENT_DURATION = 0.5
EMBEDDING_WINDOW = 1.5
EMBEDDING_STEP = 0.75


# ---- GLOBALS (per worker) ----
_model = None
_audio_data = None
_sr = None


# ---- INITIALIZER (runs once per worker) ----
def _init_worker(audio_data, sr):
    global _model, _audio_data, _sr
    import wespeakerruntime as wespeaker_rt

    _model = wespeaker_rt.Speaker(lang="en")
    _audio_data = audio_data
    _sr = sr
    

# ---- WORKER FUNCTION ----
def _process_batch(batch):
    """Process a batch tasks"""
    """Each task processes a window → extract embedding."""
    global _model, _audio_data, _sr
    results = []

    for win_start, win_end, parent_idx in batch:
        # Each process loads its own model (safe for multiprocessing)
        # model = wespeaker_rt.Speaker(lang="en")

        try:
            start_sample = int(win_start * _sr)
            end_sample = int(win_end * _sr)
            segment_audio = _audio_data[start_sample:end_sample]

            import tempfile
            tmp_path = None
            # Write temp wav (required by wespeaker runtime)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, segment_audio, _sr)
            # buf = io.BytesIO()
            # sf.write(buf, segment_audio, sr, format="WAV")
            # buf.seek(0)

            emb = _model.extract_embedding(tmp_path)

            if emb is not None:
                if emb.ndim == 2:
                    emb = emb[0]

                results.append((
                    emb,
                    SubSegment(
                        start=win_start,
                        end=win_end,
                        parent_idx=parent_idx,
                    )
                ))
                #
                # return emb, SubSegment(
                #     start=win_start,
                #     end=win_end,
                #     parent_idx=parent_idx,
                # )

        except Exception:
            continue

    return results


# ---- HELPER: CHUNK TASKS ----
def _chunkify(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


# ---- MAIN FUNCTION ----
def extract_embeddings(
    audio_path: str | Path,
    speech_segments: List[SpeechSegment],
) -> Tuple[np.ndarray, List[SubSegment]]:
    """Extract 256-dim speaker embeddings using multiprocessing."""

    logger.info("Extracting speaker embeddings (multi-core enabled)...")

    # 🔥 IMPORTANT: prevent oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Load full audio
    audio_data, sr = sf.read(str(audio_path))
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # ---- PREPARE TASKS ----
    tasks = []

    for idx, seg in enumerate(speech_segments):
        seg_duration = seg.duration

        if seg_duration < MIN_SEGMENT_DURATION:
            continue

        # Create windows
        if seg_duration <= EMBEDDING_WINDOW * 1.5:
            windows = [(seg.start, seg.end)]
        else:
            windows = []
            win_start = seg.start
            while win_start + MIN_SEGMENT_DURATION < seg.end:
                win_end = min(win_start + EMBEDDING_WINDOW, seg.end)
                windows.append((win_start, win_end))
                win_start += EMBEDDING_STEP

        for win_start, win_end in windows:
            tasks.append((win_start, win_end, idx))

    if not tasks:
        return np.empty((0, 256), dtype=np.float32), []

    # ---- BATCH TASKS ----
    batched_tasks = _chunkify(tasks, BATCH_SIZE)

    logger.info(
        "Processing %d windows in %d batches using %d workers...",
        len(tasks), len(batched_tasks), NUM_WORKERS
    )
    # logger.info("Processing %d windows using %d workers...", len(tasks), NUM_WORKERS)

    # ---- MULTIPROCESSING ----
    with Pool(
            NUM_WORKERS,
            initializer=_init_worker,
            initargs=(audio_data,sr)
    ) as pool:
        results = pool.imap_unordered(_process_batch, batched_tasks, chunksize=1)

        # ---- COLLECT RESULTS ----
        embeddings: List[np.ndarray] = []
        subsegments: List[SubSegment] = []

        for batch in results:
            for emb, subseg in batch:
                embeddings.append(emb)
                subsegments.append(subseg)

    if not embeddings:
        return np.empty((0, 256), dtype=np.float32), []

    X = np.stack(embeddings)

    logger.info("Extracted %d embeddings (dim=%d)", X.shape[0], X.shape[1])

    return X, subsegments
