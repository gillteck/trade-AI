# -*- coding: utf-8 -*-
"""
Strategy Extraction Service using Gemini AI (Track A).

This module manages interfacing with the Google Gemini API using the modern
`google-genai` SDK to carry out structured strategy extraction. It leverages
Pydantic schemas to align Gemini's output guarantee perfectly with the downstream
and Track B expectation.

Design Decisions:
1. Strict Typings via Pydantic: We define the structure using standard Pydantic models.
   This automatically generates a schema that is sent directly to Gemini as response_schema,
   guaranteeing a parser-safe JSON structure.
2. Anti-AI-Slop & Precision Instructions: The prompt explicitly tells Gemini to *never*
   invent indicators, numbers, or rules, preserving transcript vagueness ("wait for confirmation")
   accurately and penalizing lack of detail in `extraction_confidence`.
3. Extended StrategyObject Wrapping: The final extraction output wraps the strategy data
   and couples it with workflow state fields (UUID, status, timestamp, metrics placeholders) 
   suitable for sqlite/postgres downstream pipelines.
"""

import os
from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, condecimal

# Try importing the modern Google GenAI SDK. 
# Downstream users will run: pip install google-genai
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
except ImportError:
    # Included a fallback check warning so if ran locally without config it clarifies setup
    pass


class RiskManagement(BaseModel):
    """Encapsulates risk control rules described in the trading source."""
    stop_loss: Optional[str] = Field(
        None, 
        description="Stop loss rules/parameters (e.g., '1.5 ATR', 'below recent pivot swing low'). Store null if not explicitly mentioned."
    )
    take_profit: Optional[str] = Field(
        None, 
        description="Take profit level or logic (e.g., '2R risk-reward ratio', 'opposite Bollinger band'). Store null if not explicitly mentioned."
    )
    position_sizing: Optional[str] = Field(
        None, 
        description="Position sizing details (e.g., '1% account risk per trade', 'fixed lot'). Store null if not explicitly mentioned."
    )
    max_concurrent_positions: Optional[int] = Field(
        None, 
        description="Maximum concurrent open positions allowed by the strategy. Store null if not explicitly mentioned."
    )


class TradingStrategy(BaseModel):
    """The raw trading plan extracted by Gemini from the captions transcript."""
    name: str = Field(
        ..., 
        description="A concise, descriptive name for the strategy (e.g., '5-Minute EMA Scalping Strategy')."
    )
    asset_class: Literal["forex", "stocks", "crypto", "unknown"] = Field(
        ..., 
        description="The primary financial asset class. Must be forex, stocks, crypto, or unknown if unavailable."
    )
    timeframe: str = Field(
        ..., 
        description="The chart interval used (e.g., '5m', '1h', 'daily', 'flexible')."
    )
    indicators: List[str] = Field(
        ..., 
        description="List of core indicators required (e.g., ['50 EMA', '200 EMA', 'RSI (14)']). Return empty list if no indicators."
    )
    entry_rules: str = Field(
        ..., 
        description="Exact concrete buy/sell rules explicitly detailed in the transcript. Preserve vagueness if applicable."
    )
    exit_rules: str = Field(
        ..., 
        description="Exact concrete trigger rules for closing positions (not SL/TP, but strategy close conditions)."
    )
    risk_management: RiskManagement = Field(
        ..., 
        description="Comprehensive details on risk management."
    )
    extraction_confidence: float = Field(
        ..., 
        description="Self-evaluated precision of strategy extraction from 0.0 (extremely vague/contradictory) to 1.0 (crystal clear)."
    )
    extraction_notes: str = Field(
        ..., 
        description="Critical logs marking missing stop loss rules, contradictory indicators, or general gaps in the transcript."
    )


class StrategyObject(BaseModel):
    """
    Complete Track A persistent state wrapper containing the strategy data
    along with orchestration status and backtest/paper metric hooks.
    """
    id: UUID = Field(default_factory=uuid4)
    status: Literal["extracted", "backtested", "paper_trading", "live", "retired"] = "extracted"
    source_video_url: str
    source_video_title: str
    source_channel: str
    extracted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    
    # Flattened extracted fields for clean DB storage mapping & simpler downstream lookup:
    name: str
    asset_class: Literal["forex", "stocks", "crypto", "unknown"]
    timeframe: str
    indicators: List[str]
    entry_rules: str
    exit_rules: str
    risk_management: RiskManagement
    extraction_confidence: float
    extraction_notes: str
    
    # Backtesting & operational metrics placeholders (filled in by Track B)
    backtest_metrics: Optional[dict] = None
    paper_trading_metrics: Optional[dict] = None


class StrategyExtractionService:
    """Service utilizing Gemini's structured generation capabilities to extract trading strategies."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the service.
        
        Args:
            api_key: Optional Google Gemini API key. If omitted, the SDK will 
                     automatically look up the 'GEMINI_API_KEY' environment variable.
        """
        # Read API key from environment if not passed explicitly
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        # Instantiate google-genai Client
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            # Create a mock/lazy initialization warning if dependencies aren't built yet
            self.client = None
            self._init_error = e

    def extract_strategy(
        self, 
        transcript_text: str, 
        video_url: str, 
        video_title: str = "Unknown Video", 
        channel_name: str = "Unknown Channel"
    ) -> StrategyObject:
        """
        Analyzes transcript text using Gemini and extracts a structured trading strategy.
        
        Args:
            transcript_text: The normalized YouTube video transcript text.
            video_url: Origin YouTube URL.
            video_title: Title of the video.
            channel_name: Channel name of the publisher.
            
        Returns:
            StrategyObject: Persistent strategy record with initial status "extracted".
            
        Raises:
            RuntimeError: If Gemini API credentials are absent or if structured call fails.
        """
        if self.client is None:
            err_msg = getattr(self, "_init_error", "Gemini Client not initialized")
            raise RuntimeError(
                f"Gemini Client is missing. Please ensure 'google-genai' is installed "
                f"and 'GEMINI_API_KEY' is set in your environment. Detailed error: {err_msg}"
            )

        # 1. Standard instruction guiding Gemini's extraction behavior
        system_instruction = (
            "You are a professional quantitative financial analyst and algorithmic trading system compiler. "
            "Your objective is to extract a highly structured trading strategy from a video transcript.\n\n"
            "CRITICAL INSTRUCTION: Do NOT invent, assume, extrapolate, or inject any parameters, indicators, "
            "or rules that are not explicitly stated in the transcript text.\n"
            "If the video states something vague like 'wait for confirmation', 'look for a rejection candle', "
            "or 'use momentum' without detailing exactly what indicator or formula achieves that, you MUST "
            "preserve that exact vague wording (e.g. 'wait for confirmation') in 'entry_rules' or 'exit_rules'. "
            "Do NOT invent indicators (e.g. do not assume momentum means RSI) to fill in these blanks.\n\n"
            "Flag all vagueness, inconsistencies, missing definitions, or incomplete exit rules in 'extraction_notes'.\n"
            "Rate your self-evaluated precision as a float in 'extraction_confidence' on a scale of 0.0 (extremely vague, "
            "contradictory, or missing crucial buy/sell rules) to 1.0 (pristine, step-by-step clear strategy details)."
        )

        prompt_body = (
            f"Here is the transcript of a YouTube video about a trading strategy.\n"
            f"Analyze it carefully and map it into the requested JSON schema.\n\n"
            f"--- TRANSCRIPT START ---\n"
            f"{transcript_text}\n"
            f"--- TRANSCRIPT END ---\n"
        )

        try:
            # 2. Config structured response mirroring the TradingStrategy pydantic schema
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TradingStrategy,
                system_instruction=system_instruction,
                temperature=0.1,  # Low temperature preserves high determinism & minimizes creative extrapolation
            )

            # 3. Call model
            response = self.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt_body,
                config=config
            )

            # 4. Extract and validate structuring
            # Structured schema parses cleanly through Pydantic
            raw_strategy = TradingStrategy.model_validate_json(response.text)

            # 5. Assemble and wrap in StrategyObject
            return StrategyObject(
                source_video_url=video_url,
                source_video_title=video_title,
                source_channel=channel_name,
                
                # Unpack TradingStrategy fields flatly:
                name=raw_strategy.name,
                asset_class=raw_strategy.asset_class,
                timeframe=raw_strategy.timeframe,
                indicators=raw_strategy.indicators,
                entry_rules=raw_strategy.entry_rules,
                exit_rules=raw_strategy.exit_rules,
                risk_management=raw_strategy.risk_management,
                extraction_confidence=raw_strategy.extraction_confidence,
                extraction_notes=raw_strategy.extraction_notes
            )

        except APIError as api_err:
            raise RuntimeError(f"Gemini API invocation failed: {str(api_err)}") from api_err
        except Exception as pydantic_or_unexpected:
            raise RuntimeError(
                f"Failed to extract structured trading strategy due to validation/system failure: "
                f"{str(pydantic_or_unexpected)}"
            ) from pydantic_or_unexpected
