# -*- coding: utf-8 -*-
"""
YouTube Ingestion Service for Trading Strategy Pipeline (Track A).

This module handles extracting YouTube video IDs from various URL formats,
fetching transcript/captions using the modern instance-based `youtube-transcript-api`
library, and assembling a clean, whitespace-normalized output.

Design Decisions:
1. Unified Exception Interface: Every failure (disabled transcripts, network issue, 
   unavailable video, unavailable language, etc.) is caught and wrapped inside 
   `IngestionError`. This simplifies client integration so that they only need 
   to handle a single exception type.
2. Modern Instance-Based Api: We adhere strictly to the newer instance-based 
   API `YouTubeTranscriptApi().list(video_id)` introduced in version 1.2.4+, 
   rather than deprecated static class methods.
3. Fallback Selection Strategy: We inspect the returned `TranscriptList` to prefer 
   manually uploaded English transcripts, fall back to auto-generated English, 
   and if English is entirely missing, fallback to any available manual or generated transcript.
"""

import re
from dataclasses import dataclass
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    FailedToCreateConsentCookie
)

class IngestionError(Exception):
    """Custom exception representing any failure in the YouTube transcript ingestion process."""
    pass


@dataclass(frozen=True)
class VideoTranscript:
    """Dataclass holding structured outcome of YouTube ingestion."""
    video_id: str
    url: str
    transcript: str
    transcript_language: str


def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character video ID from varied YouTube URL formats.
    
    Supports:
        - Raw ID: 'dQw4w9WgXcQ'
        - Standard watch: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
        - Shortened youtu.be: 'https://youtu.be/dQw4w9WgXcQ'
        - Embed: 'https://www.youtube.com/embed/dQw4w9WgXcQ'
        - Shorts: 'https://www.youtube.com/shorts/dQw4w9WgXcQ'
        
    Raises:
        IngestionError: If a valid 11-character video ID cannot be determined.
    """
    cleaned = url.strip()
    
    # 1. Check if the input is already a raw 11-character ID
    if len(cleaned) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", cleaned):
        return cleaned

    # 2. Comprehensive pattern search for common YouTube URL structures
    patterns = [
        r"(?:v=|\/v\/|embed\/|shorts\/|youtu\.be\/|\/embed\/|\/watch\?v=|\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)
            
    raise IngestionError(f"Format unrecognized. Could not extract a valid 11-character YouTube video ID from: {url}")


class YouTubeIngestionService:
    """Service class for fetching transcripts from YouTube."""

    def __init__(self):
        """Initializes the YouTube Ingestion Service using the modern class instance."""
        self._api = YouTubeTranscriptApi()

    def fetch_transcript(self, url_or_id: str) -> VideoTranscript:
        """
        Fetches the transcript for the specified video, falling back to auto-generated if needed.
        
        Args:
            url_or_id: Full YouTube URL or direct 11-character video ID.
            
        Returns:
            VideoTranscript: Structured container for parsed results.
            
        Raises:
            IngestionError: Wraps any API exceptions into a single consistent error.
        """
        try:
            video_id = extract_video_id(url_or_id)
        except IngestionError:
            raise
        except Exception as e:
            raise IngestionError(f"Unexpected error resolving video ID: {str(e)}") from e

        video_url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            # Multi-stage fallback search (defensively supports list_transcripts or list)
            if hasattr(self._api, "list_transcripts"):
                transcript_list = self._api.list_transcripts(video_id)
            elif hasattr(self._api, "list"):
                transcript_list = self._api.list(video_id)
            else:
                # Direct dynamic invoke if metadata attributes are hidden
                try:
                    transcript_list = self._api.list_transcripts(video_id)
                except AttributeError:
                    transcript_list = self._api.list(video_id)
            
            # 1. Look for English (prefers manual over auto-generated English)
            try:
                transcript_obj = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
            except NoTranscriptFound:
                # 2. If English is not available, gather list of all available languages (manual first, then auto-extracted)
                manual_langs = [t.language_code for t in transcript_list if not t.is_generated]
                gen_langs = [t.language_code for t in transcript_list if t.is_generated]
                all_langs = manual_langs + gen_langs
                
                if not all_langs:
                    raise IngestionError(f"No transcripts/captions available in any language for video {video_id}.")
                
                # Fetch the best available language transcript
                transcript_obj = transcript_list.find_transcript(all_langs)

            # 3. Fetch the content blocks
            data_blocks = transcript_obj.fetch()
            
            # whitespace normalize the full transcript text
            text_slices = []
            for block in data_blocks:
                if isinstance(block, dict):
                    text_slices.append(block.get("text", "").strip())
                else:
                    text_slices.append(getattr(block, "text", "").strip())
            full_text = " ".join(text_slices)
            normalized_text = " ".join(full_text.split())
            
            return VideoTranscript(
                video_id=video_id,
                url=video_url,
                transcript=normalized_text,
                transcript_language=transcript_obj.language_code
            )

        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, FailedToCreateConsentCookie) as api_err:
            raise IngestionError(
                f"Failed to ingest transcript for video {video_id}. "
                f"Reason: {type(api_err).__name__} - {str(api_err)}"
            ) from api_err
        except IngestionError:
            raise
        except Exception as general_err:
            raise IngestionError(
                f"An unexpected system exception occurred during YouTube transcript retrieval: {str(general_err)}"
            ) from general_err
