"""
models.py — Standardized dataclasses and models for the Agentic Multi-Video Research Assistant.
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class VideoMetadata:
    """Metadata of a video, including duration, authorship, views, and source URL."""
    video_id: str
    title: str
    author: str
    duration: int  # In seconds
    upload_date: str  # YYYY-MM-DD
    view_count: int
    url: str
    language: str = "English"
    transcript_source: str = "manual"
    detected_language: str = "en"

@dataclass
class TranscriptSegment:
    """A segment of transcript text mapping directly to a specific timestamp duration."""
    text: str
    start: float  # Start time in seconds
    duration: float  # Duration in seconds

@dataclass
class VideoData:
    """The complete representation of an ingested video: its metadata and full timestamped segments."""
    metadata: VideoMetadata
    segments: List[TranscriptSegment]

@dataclass
class RetrievalChunk:
    """A segment chunk parsed for indexing and search retrieval, tracking text and timestamp bounds."""
    chunk_id: str
    video_id: str
    text: str
    start_time: float
    end_time: float
    score: float
