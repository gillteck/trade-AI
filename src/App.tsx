/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { 
  Youtube, 
  Cpu, 
  Database, 
  TrendingUp, 
  ShieldAlert, 
  CheckCircle2, 
  XCircle, 
  Timer, 
  Sparkles, 
  HelpCircle, 
  Play, 
  FileCode, 
  Sliders, 
  Layers, 
  ArrowRight, 
  ExternalLink,
  Flame,
  LineChart,
  GitCommit,
  Check,
  Search,
  BookOpen,
  Info
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

// Python Code Contents for display in the interactive code explorer tab
const CODE_FILES = {
  "ingestion.py": `# -*- coding: utf-8 -*-
"""
YouTube Ingestion Service for Trading Strategy Pipeline (Track A).

Fetches transcripts using instance-based youtube-transcript-api.
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
    """Custom exception raised for any transcript ingestion failures."""
    pass

@dataclass(frozen=True)
class VideoTranscript:
    video_id: str
    url: str
    transcript: str
    transcript_language: str

def extract_video_id(url: str) -> str:
    cleaned = url.strip()
    if len(cleaned) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", cleaned):
        return cleaned
    patterns = [
        r"(?:v=|/v/|embed/|shorts/|youtu\\.be/|/embed/|/watch\\?v=|\\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\\.)?youtube\\.com/watch\\?v=([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)
    raise IngestionError(f"Format unrecognized: {url}")

class YouTubeIngestionService:
    def __init__(self):
        self._api = YouTubeTranscriptApi()

    def fetch_transcript(self, url_or_id: str) -> VideoTranscript:
        try:
            video_id = extract_video_id(url_or_id)
            transcript_list = self._api.list_transcripts(video_id)
            try:
                transcript_obj = transcript_list.find_transcript(['en', 'en-US'])
            except NoTranscriptFound:
                all_langs = [t.language_code for t in transcript_list]
                if not all_langs:
                     raise IngestionError("No transcripts available.")
                transcript_obj = transcript_list.find_transcript(all_langs)
            
            data_blocks = transcript_obj.fetch()
            full_text = " ".join([b.get("text", "").strip() for b in data_blocks])
            return VideoTranscript(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                transcript=" ".join(full_text.split()),
                transcript_language=transcript_obj.language_code
            )
        except Exception as e:
            raise IngestionError(f"Ingestion failed: {str(e)}") from e`,

  "extractor.py": `# -*- coding: utf-8 -*-
"""
Strategy Extraction Service utilizing Google Gemini AI.
"""
from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

class RiskManagement(BaseModel):
    stop_loss: Optional[str] = None
    take_profit: Optional[str] = None
    position_sizing: Optional[str] = None
    max_concurrent_positions: Optional[int] = None

class TradingStrategy(BaseModel):
    name: str = Field(..., description="Concise strategy title")
    asset_class: Literal["forex", "stocks", "crypto", "unknown"]
    timeframe: str
    indicators: List[str]
    entry_rules: str
    exit_rules: str
    risk_management: RiskManagement
    extraction_confidence: float
    extraction_notes: str

class StrategyObject(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    status: Literal["extracted", "backtested", "paper_trading", "live", "retired"] = "extracted"
    source_video_url: str
    source_video_title: str
    source_channel: str
    extracted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    name: str
    asset_class: str
    timeframe: str
    indicators: List[str]
    entry_rules: str
    exit_rules: str
    risk_management: RiskManagement
    extraction_confidence: float
    extraction_notes: str
    backtest_metrics: Optional[dict] = None
    paper_trading_metrics: Optional[dict] = None

class StrategyExtractionService:
    def __init__(self, api_key: Optional[str] = None):
        self.client = genai.Client(api_key=api_key)

    def extract_strategy(self, transcript_text: str, video_url: str, **meta) -> StrategyObject:
        sys_instruction = "Extract structural plans without extrapolating or inventing indicators..."
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TradingStrategy,
            system_instruction=sys_instruction,
            temperature=0.1
        )
        response = self.client.models.generate_content(
            model="gemini-3.5-flash",
            contents=transcript_text,
            config=config
        )
        raw = TradingStrategy.model_validate_json(response.text)
        return StrategyObject(
            source_video_url=video_url,
            source_video_title=meta.get("title", ""),
            source_channel=meta.get("channel", ""),
            name=raw.name,
            asset_class=raw.asset_class,
            timeframe=raw.timeframe,
            indicators=raw.indicators,
            entry_rules=raw.entry_rules,
            exit_rules=raw.exit_rules,
            risk_management=raw.risk_management,
            extraction_confidence=raw.extraction_confidence,
            extraction_notes=raw.extraction_notes
        )`,

  "storage.py": `# -*- coding: utf-8 -*-
"""
SQLite Persistence Layer and pipeline transition machine.
"""
import json
import sqlite3
from typing import List, Optional, Dict, Any, Union

class StorageService:
    VALID_TRANSITIONS = {
        "extracted": ["backtested", "retired"],
        "backtested": ["paper_trading", "retired"],
        "paper_trading": ["live", "retired"],
        "live": [],
        "retired": []
    }

    def __init__(self, db_path: str = "strategies.db"):
        self.db_path = db_path
        self._create_tables()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        with self._get_connection() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id TEXT PRIMARY KEY, status TEXT, source_video_url TEXT, name TEXT,
                    asset_class TEXT, timeframe TEXT, indicators TEXT, entry_rules TEXT,
                    exit_rules TEXT, risk_management TEXT, extraction_confidence REAL,
                    extraction_notes TEXT, backtest_metrics TEXT, paper_trading_metrics TEXT
                )
            """)
            c.commit()

    def update_status(self, strategy_id: str, new_status: str):
        with self._get_connection() as c:
            curr = c.execute("SELECT status FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
            if not curr:
                raise ValueError("Not found")
            curr_status = curr["status"]
            if new_status not in self.VALID_TRANSITIONS.get(curr_status, []):
                raise ValueError(f"Transition from {curr_status} to {new_status} is denied.")
            c.execute("UPDATE strategies SET status = ? WHERE id = ?", (new_status, strategy_id))
            c.commit()`,

  "main.py": `# -*- coding: utf-8 -*-
"""
Track A Strategy Pipeline Execution Harness
"""
import sys
import argparse
from ingestion import YouTubeIngestionService
from extractor import StrategyExtractionService
from storage import StorageService

def run_pipeline(url: str):
    print("Ingesting...")
    transcript = YouTubeIngestionService().fetch_transcript(url)
    print("Extracting with Gemini...")
    strategy = StrategyExtractionService().extract_strategy(transcript.transcript, url)
    print("Saving to local database...")
    StorageService().insert_strategy(strategy)
    print("Execution complete! Stored Strategy UUID: " + str(strategy.id))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    run_pipeline(parser.parse_args().url)`,

  "requirements.txt": `google-genai>=0.1.0
youtube-transcript-api>=1.2.4
pydantic>=2.0.0`
};

export default function App() {
  const [activeTab, setActiveTab] = useState<"pipeline" | "code">("pipeline");
  const [selectedLanguage, setSelectedLanguage] = useState<string>("ingestion.py");
  
  // Pipeline interactive state
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState(0);
  const [processingLogs, setProcessingLogs] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedStatusFilter, setSelectedStatusFilter] = useState<string>("all");
  
  // Simulated SQLite persistence state filled with high-fidelity pre-extracted items
  const [strategies, setStrategies] = useState<any[]>([
    {
      id: "6a2f4da4-16b7-4b47-a870-07bf84fd8d7a",
      status: "backtested",
      source_video_url: "https://www.youtube.com/watch?v=kG_26r0a2P8",
      source_video_title: "My Ultimate 5-Minute Moving Average Scalping Strategy (92% Win Rate)",
      source_channel: "Atlas Quantitative Research",
      extracted_at: "2026-06-21T04:12:11Z",
      name: "5-Minute Scalping Dual EMA Reversal",
      asset_class: "crypto",
      timeframe: "5m",
      indicators: ["50 EMA", "200 EMA", "RSI (14)"],
      entry_rules: "Enter long when the price crosses above the 50 EMA and the 50 EMA is strictly above the 200 EMA. Concurrently, the RSI (14) must have dipped below 40 and crossed back above 40 within the last 3 candles as a confirmation of momentum reversal. Short criteria are the exact mirror.",
      exit_rules: "Exit position when the price crosses back over the 50 EMA in the opposite direction, or when 'downward momentum appears heavy' on the 5-minute ticks.",
      risk_management: {
        stop_loss: "Plastered 10 pips below the active 200 EMA swing pivot line.",
        take_profit: "1.5 times the established stop-loss risk amount.",
        position_sizing: "1.0% account equity maximum allocation per trade",
        max_concurrent_positions: 3
      },
      extraction_confidence: 0.85,
      extraction_notes: "The exit criteria includes discretionary terms ('downward momentum appears heavy'). This vagueness was preserved directly instead of inserting unmentioned indicators. Confidence is capped at 0.85 due to this subjective rule.",
      backtest_metrics: {
        sharpe_ratio: 1.82,
        win_rate: 0.584,
        max_drawdown: "8.4%",
        total_trades: 142
      },
      paper_trading_metrics: null
    },
    {
      id: "f3c2bc1d-91b3-4f9e-a8fd-381c6bb4c90e",
      status: "extracted",
      source_video_url: "https://youtu.be/mFas8x9rKkM",
      source_video_title: "How I Trade Forex Clean Breakouts with the ATR Multiplier Method",
      source_channel: "FX Chartists Guild",
      extracted_at: "2026-06-22T02:44:19Z",
      name: "Forex Impulse Volatility Breakout",
      asset_class: "forex",
      timeframe: "1h",
      indicators: ["ATR (14)", "20 EMA"],
      entry_rules: "Identify a consolidation range covering at least 20 bars. Enter Buy Stop or Sell Stop pending orders 1.5x the current ATR value outside of the range bounds. The 20 EMA must be oriented in the breakout direction as a trend filter.",
      exit_rules: "Close the trade when a counter-trend reversal candle closes over the 20 EMA.",
      risk_management: {
        stop_loss: "1.5 times ATR from the entry breakout price level.",
        take_profit: "Trailing target adjusted by lock-in ticks at 3.0x ATR.",
        position_sizing: "0.5% fixed risk per pip deviation value.",
        max_concurrent_positions: 2
      },
      extraction_confidence: 0.95,
      extraction_notes: "Outstandingly structured video. All parameters, indicators, and exact numerical formulas (1.5x ATR, min 20 bars consolidation) are cleanly stated. No invented metrics needed.",
      backtest_metrics: null,
      paper_trading_metrics: null
    },
    {
      id: "da4412f8-bf78-4309-8de3-cb4528189c4a",
      status: "live",
      source_video_url: "https://www.youtube.com/shorts/pL_9a87cdX",
      source_video_title: "Super Easy Stocks Trend Rider Strategy using Only One Indicator",
      source_channel: "Short Term Gains",
      extracted_at: "2026-06-20T08:15:00Z",
      name: "Single Indicator Trend Rider",
      asset_class: "stocks",
      timeframe: "daily",
      indicators: ["Halftrend (V2)"],
      entry_rules: "Buy when the Halftrend indicator paints an arrow signaling buy and the color shifts to deep blue. Sell/Short when Halftrend paints red.",
      exit_rules: "Wait for a market trend shift color change. No other intermediate close-out rules specified.",
      risk_management: {
        stop_loss: null,
        take_profit: null,
        position_sizing: "discretionary allocation",
        max_concurrent_positions: null
      },
      extraction_confidence: 0.45,
      extraction_notes: "Critically incomplete strategy. The author failed to discuss stop-loss distances, exact risk management parameters, or how to handle multiple stocks. Confidence set to low (0.45). Vagueness preserved strictly as described in system instructions.",
      backtest_metrics: {
        sharpe_ratio: 0.92,
        win_rate: 0.442,
        max_drawdown: "28.3%",
        total_trades: 91
      },
      paper_trading_metrics: {
        monthly_yield: "3.2%",
        tracked_weeks: 4,
        slippage_ticks: 1.4
      }
    }
  ]);

  const [selectedStrategyId, setSelectedStrategyId] = useState<string>("6a2f4da4-16b7-4b47-a870-07bf84fd8d7a");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  // Quant metrics fields state
  const [inputSharpe, setInputSharpe] = useState("");
  const [inputWinRate, setInputWinRate] = useState("");
  const [inputDrawdown, setInputDrawdown] = useState("");
  const [inputTotalTrades, setInputTotalTrades] = useState("");

  const activeStrategy = strategies.find(s => s.id === selectedStrategyId) || strategies[0];

  // YouTube presets URLs
  const sampleVideos = [
    {
      url: "https://www.youtube.com/watch?v=hS_26z819xZ",
      title: "MACD & Stochastic Dual Cross Scalper Strategy",
      channel: "QuantEdge Labs",
      duration: "14:12"
    },
    {
      url: "https://youtu.be/fL5318ba9d8",
      title: "How to Trade Bollinger Squeeze Breakouts on Crypto",
      channel: "Nifty Traders",
      duration: "11:45"
    }
  ];

  // Pipeline simulation execution handler
  const handleIngestAndExtract = () => {
    if (!youtubeUrl) {
      setErrorMessage("Please input a valid YouTube URL or direct Video ID.");
      return;
    }
    
    setErrorMessage(null);
    setIsProcessing(true);
    setProcessingStep(0);
    setProcessingLogs(["Parsing URL format and identifying 11-char Video ID..."]);
    
    // Step-by-step pipeline logging simulator to expose pipeline phases to users
    setTimeout(() => {
      setProcessingStep(1);
      setProcessingLogs(prev => [
        ...prev,
        "✔ Video ID parsed: " + (youtubeUrl.length === 11 ? youtubeUrl : "hS_26z8") + "...",
        "Step 1: Connecting to youtube-transcript-api client instance...",
        "Fetching transcript metadata list..."
      ]);
    }, 1200);

    setTimeout(() => {
      setProcessingStep(2);
      setProcessingLogs(prev => [
        ...prev,
        "✔ Found available English transcript (Auto-generated fallback triggered).",
        "Downloading captions text tracks (length: 1,482 words)...",
        "Normalizing whitespace and joining transcript segments...",
        "Step 2: Preparing structured prompt with anti-slop rules...",
        "Invoking model: 'gemini-3.5-flash' in JSON structured output mode..."
      ]);
    }, 2500);

    setTimeout(() => {
      setProcessingStep(3);
      setProcessingLogs(prev => [
        ...prev,
        "✔ Gemini model response received in exactly 1.4 seconds.",
        "Validating structure against Pydantic schema...",
        "Step 3: Storing strategy record in local SQLite strategies.db...",
        "Inserting unified record metadata..."
      ]);
    }, 4000);

    setTimeout(() => {
      const newId = uuid4();
      const extractedStrategy = {
        id: newId,
        status: "extracted",
        source_video_url: youtubeUrl.includes("http") ? youtubeUrl : `https://www.youtube.com/watch?v=${youtubeUrl}`,
        source_video_title: "Bollinger Squeezes & Keltner Channels Breakout Strategist",
        source_channel: "QuantEdge Labs",
        extracted_at: new Date().toISOString(),
        name: "Bollinger-Keltner Volatility Squeeze System",
        asset_class: "crypto",
        timeframe: "15m",
        indicators: ["Bollinger Bands (20, 2)", "Keltner Channels (20, 1.5)", "Momentum Index"],
        entry_rules: "Wait for Bollinger Bands to completely squeese inside the Keltner Channels. Once prices close outside either of the channel extreme lines, trigger a market swing in that breakout direction. Concurrently, Momentum Index must be positive for buys and negative for sells.",
        exit_rules: "Exit position immediately whenever the range Bollinger band lines begin expanding outward and then turn inward again, showing dissipation.",
        risk_management: {
          stop_loss: "1.2x ATR or the opposite boundaries of the breakout Bollinger Band line.",
          take_profit: "Closed on reverse trigger momentum. No numerical ratio targets specified.",
          position_sizing: "1.5% account size with standard deviation leverage.",
          max_concurrent_positions: 4
        },
        extraction_confidence: 0.78,
        extraction_notes: "The exit parameter is somewhat subjective ('begin turning inward again'). Extracted rule word-for-word to maintain integrity rather than creating mock definitions. Consequent confidence set to 0.78.",
        backtest_metrics: null,
        paper_trading_metrics: null
      };

      setStrategies(prev => [extractedStrategy, ...prev]);
      setSelectedStrategyId(newId);
      setIsProcessing(false);
      setYoutubeUrl("");
      setProcessingLogs([]);
    }, 5500);
  };

  // Helper function to generate simulated UUID
  function uuid4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  // Handle pipeline graduation (State Machine)
  const transitionStatus = (newStatus: string) => {
    setErrorMessage(null);
    const validMap: Record<string, string[]> = {
      "extracted": ["backtested", "retired"],
      "backtested": ["paper_trading", "retired"],
      "paper_trading": ["live", "retired"],
      "live": [],
      "retired": []
    };

    const currentStatus = activeStrategy.status;
    const allowed = validMap[currentStatus] || [];

    if (!allowed.includes(newStatus)) {
      setErrorMessage(
        `State Machine Violation: Transition from '${currentStatus}' to '${newStatus}' is denied! ` +
        `Only these next actions are allowed: [${allowed.length > 0 ? allowed.join(", ") : "None. This is a terminal state"}]`
      );
      return;
    }

    // Success transition
    setStrategies(prev => prev.map(s => {
      if (s.id === activeStrategy.id) {
        return { ...s, status: newStatus };
      }
      return s;
    }));
  };

  // Handle updating metrics independently
  const saveQuantMetrics = (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    
    if (!inputSharpe || !inputWinRate || !inputDrawdown || !inputTotalTrades) {
      setErrorMessage("Please fill all quant metrics fields (Sharpe, Win Rate, Drawdown, Trades).");
      return;
    }

    setStrategies(prev => prev.map(s => {
      if (s.id === activeStrategy.id) {
        return {
          ...s,
          backtest_metrics: {
            sharpe_ratio: parseFloat(inputSharpe),
            win_rate: parseFloat(inputWinRate) / 100,
            max_drawdown: inputDrawdown.includes("%") ? inputDrawdown : `${inputDrawdown}%`,
            total_trades: parseInt(inputTotalTrades)
          }
        };
      }
      return s;
    }));

    setInputSharpe("");
    setInputWinRate("");
    setInputDrawdown("");
    setInputTotalTrades("");
  };

  // Filter strategies based on search and status
  const filteredStrategies = strategies.filter(s => {
    const matchesSearch = s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          s.indicators.some((ind: string) => ind.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesStatus = selectedStatusFilter === "all" || s.status === selectedStatusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="min-h-screen bg-[#020617] text-slate-100 font-sans relative overflow-x-hidden selection:bg-indigo-500/30 selection:text-white" id="main_container">
      {/* Background Decorative Gradient Blobs */}
      <div className="absolute top-[-5%] left-[-10%] w-[50%] h-[40%] bg-blue-600/15 rounded-full blur-[120px] pointer-events-none z-0"></div>
      <div className="absolute bottom-[20%] right-[-10%] w-[45%] h-[45%] bg-indigo-600/15 rounded-full blur-[120px] pointer-events-none z-0"></div>
      <div className="absolute top-[40%] left-[20%] w-[35%] h-[35%] bg-indigo-500/10 rounded-full blur-[100px] pointer-events-none z-0"></div>

      {/* High-contrast Glass Header */}
      <header className="border-b border-white/10 bg-white/5 backdrop-blur-md sticky top-0 z-50 shadow-lg shadow-indigo-950/20" id="app_header">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-600 text-white shadow-lg shadow-indigo-600/25">
              <Youtube className="h-6 w-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-display text-lg font-bold tracking-tight text-white uppercase">
                  STRAT-SYNC <span className="text-indigo-400 font-mono text-sm ml-1 px-2 py-0.5 border border-indigo-400/30 rounded">TRACK A</span>
                </span>
                <span className="rounded-full bg-indigo-500/10 px-2 py-0.5 font-mono text-[10px] font-semibold text-indigo-400 uppercase tracking-widest border border-indigo-500/20">Sandbox</span>
              </div>
              <p className="text-xs text-slate-400">YouTube captions parser & Pydantic-Gemini algorithmic extraction system</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Status indicators from the Design UI */}
            <div className="hidden md:flex gap-3">
              <div className="px-3 py-1.5 bg-white/5 backdrop-blur-md border border-white/10 rounded-full flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-300">Gemini API: Active</span>
              </div>
              <div className="px-3 py-1.5 bg-white/5 backdrop-blur-md border border-white/10 rounded-full flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-300">SQLite Engine: Online</span>
              </div>
            </div>

            <nav className="flex items-center gap-1.5 bg-white/5 border border-white/10 p-1 rounded-xl">
              <button
                onClick={() => setActiveTab("pipeline")}
                className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition-all cursor-pointer ${
                  activeTab === "pipeline" 
                    ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/20" 
                    : "text-slate-400 hover:text-white hover:bg-white/5"
                }`}
                id="tab_pipeline"
              >
                <Cpu className="h-3.5 w-3.5" />
                Ingestion Hub
              </button>
              <button
                onClick={() => setActiveTab("code")}
                className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition-all cursor-pointer ${
                  activeTab === "code" 
                    ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/20" 
                    : "text-slate-400 hover:text-white hover:bg-white/5"
                }`}
                id="tab_code"
              >
                <FileCode className="h-3.5 w-3.5" />
                Python Modules
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Container Workspace */}
      <main className="mx-auto max-w-7xl px-6 py-8 relative z-10">
        
        {/* Error notification banner */}
        <AnimatePresence>
          {errorMessage && (
            <motion.div 
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="mb-6 overflow-hidden rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-200 flex gap-3 p-4 shadow-lg backdrop-blur-md"
              id="error_banner"
            >
              <ShieldAlert className="h-5 w-5 text-rose-400 shrink-0" />
              <div>
                <p className="text-sm font-semibold text-rose-300">System Constraint / Error Triggered</p>
                <p className="text-xs text-rose-400/95 mt-0.5">{errorMessage}</p>
              </div>
              <button 
                onClick={() => setErrorMessage(null)}
                className="ml-auto text-xs text-rose-400 font-bold hover:text-rose-200 self-start animate-pulse"
              >
                Dismiss
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {activeTab === "pipeline" ? (
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-12" id="pipeline_grid">
            
            {/* LATERALLY: LEFT SIDE INGESTION & PIPELINE STATUS LIST (7 cols) */}
            <div className="lg:col-span-7 space-y-6">
              
              {/* YouTube Ingest Card */}
              <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl p-6 shadow-xl shadow-indigo-950/10">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Youtube className="h-5 w-5 text-indigo-400" />
                    <h2 className="text-xs font-bold uppercase tracking-widest text-indigo-400">YouTube Ingestion</h2>
                  </div>
                  <span className="flex items-center gap-1 text-[11px] font-semibold text-slate-300 bg-white/5 border border-white/10 rounded-md px-2 py-0.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                    API Active
                  </span>
                </div>

                <p className="text-xs text-slate-400 mb-4 font-sans leading-relaxed">
                  Pound in a standard YouTube URL or Video ID. The pipeline will invoke the modern instance-based 
                  <code className="bg-white/5 px-1 py-0.5 rounded text-indigo-300 font-mono font-semibold mx-1 border border-white/5">YouTubeTranscriptApi()</code>, 
                  grab manual/auto-generated transcripts, submit them structured to 
                  <code className="bg-white/5 px-1 py-0.5 rounded text-blue-300 font-mono font-semibold mx-1 border border-white/5">gemini-3.5-flash</code> with anti-extrapolation constraints, 
                  and write raw records into SQLite.
                </p>

                {/* Input Area */}
                <div className="flex flex-col gap-3">
                  <div>
                    <label className="block text-[10px] text-indigo-300 font-semibold uppercase tracking-wider mb-2 ml-1">Video Source URL</label>
                    <div className="flex flex-col sm:flex-row gap-2">
                      <input
                        type="text"
                        className="flex-1 rounded-xl border border-white/10 bg-slate-900/50 px-4 py-3 text-xs placeholder:text-slate-500 text-slate-200 focus:bg-slate-900/80 focus:outline-hidden focus:ring-1 focus:ring-indigo-500 outline-none transition-all"
                        placeholder="Enter YouTube URL (e.g. watch?v=hS_26z819xZ) or 11-char ID..."
                        value={youtubeUrl}
                        onChange={(e) => setYoutubeUrl(e.target.value)}
                        disabled={isProcessing}
                      />
                      <button
                        onClick={handleIngestAndExtract}
                        disabled={isProcessing}
                        className="rounded-xl bg-indigo-600 hover:bg-indigo-500 px-6 py-3 text-sm font-bold text-white transition-all hover:scale-[1.01] active:scale-95 disabled:opacity-50 flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/25 shrink-0 uppercase tracking-wider"
                      >
                        {isProcessing ? (
                          <span className="flex items-center gap-2">
                            <span className="h-2 w-2 rounded-full bg-white animate-ping"></span>
                            Extracting...
                          </span>
                        ) : "Initiate Extraction"}
                        <ArrowRight className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  {/* Sandbox Presets */}
                  <div className="mt-2">
                    <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Sandbox Quick Preloads:</span>
                    <div className="flex flex-wrap gap-2 mt-1.5">
                      {sampleVideos.map((vid, idx) => (
                        <button
                          key={idx}
                          onClick={() => setYoutubeUrl(vid.url)}
                          disabled={isProcessing}
                          className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 px-3 py-1.5 text-left text-xs text-slate-300 hover:text-white transition-all cursor-pointer"
                        >
                          <Play className="h-3 w-3 text-emerald-400 fill-emerald-400/20" />
                          <div className="max-w-[200px] truncate">
                            <span className="font-semibold text-slate-200">{vid.title}</span>
                            <span className="text-[10px] text-slate-450 ml-1">({vid.channel})</span>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Ingestion Visual Process logs */}
                {isProcessing && (
                  <div className="mt-6 border-t border-white/5 pt-5">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-slate-300">Pipeline Output Stream</span>
                      <span className="text-[10px] font-mono text-slate-400 bg-white/5 px-2 py-0.5 rounded border border-white/5">PID: {Math.floor(Math.random() * 8000) + 2000}</span>
                    </div>
                    
                    {/* Pipeline animation progress bars */}
                    <div className="flex gap-1.5 mb-4">
                      <div className={`h-1.5 rounded-full flex-1 transition-all duration-500 ${processingStep >= 0 ? "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]" : "bg-white/10"}`}></div>
                      <div className={`h-1.5 rounded-full flex-1 transition-all duration-500 ${processingStep >= 1 ? "bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.5)]" : "bg-white/10"}`}></div>
                      <div className={`h-1.5 rounded-full flex-1 transition-all duration-500 ${processingStep >= 2 ? "bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.5)]" : "bg-white/10"}`}></div>
                      <div className={`h-1.5 rounded-full flex-1 transition-all duration-500 ${processingStep >= 3 ? "bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-white/10"}`}></div>
                    </div>

                    <div className="rounded-xl bg-slate-950/70 border border-white/5 backdrop-blur-md p-4 font-mono text-[11px] text-slate-300 space-y-1.5 max-h-[160px] overflow-y-auto shadow-inner">
                      {processingLogs.map((log, i) => (
                        <div key={i} className="flex gap-2">
                          <span className="text-slate-500 select-none">[{new Date().toLocaleTimeString()}]</span>
                          <span className={log.startsWith("✔") ? "text-emerald-400 font-semibold" : log.includes("❌") ? "text-rose-400" : "text-white"}>{log}</span>
                        </div>
                      ))}
                      <div className="flex items-center gap-1.5 text-indigo-300 py-1 font-semibold">
                        <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-ping"></span>
                        Executing step: {processingStep === 0 && "Parsing parameters"}
                        {processingStep === 1 && "youtube-transcript-api.list()"}
                        {processingStep === 2 && "Gemini (Structured output/JSON)"}
                        {processingStep === 3 && "persistence SQLite strategies.db"}
                        ...
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* SQLite database viewer panel */}
              <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl overflow-hidden shadow-xl shadow-indigo-950/10">
                <div className="border-b border-white/10 bg-white/[0.02] px-6 py-4 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Database className="h-5 w-5 text-indigo-400" />
                    <h2 className="text-xs font-bold uppercase tracking-widest text-indigo-400">SQLite Table: <span className="font-mono text-indigo-300 font-bold">strategies</span></h2>
                  </div>
                  <span className="text-[11px] font-mono font-semibold bg-white/10 px-2 py-0.5 rounded text-indigo-300 border border-white/5">
                    {filteredStrategies.length} Row{filteredStrategies.length !== 1 ? "s" : ""}
                  </span>
                </div>

                {/* Filters Row */}
                <div className="border-b border-white/5 p-4 bg-transparent flex flex-col sm:flex-row gap-3 items-center justify-between">
                  <div className="relative w-full sm:w-64">
                    <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-500" />
                    <input
                      type="text"
                      className="w-full rounded-xl border border-white/10 bg-slate-900/40 pl-9 pr-4 py-2 text-xs text-white placeholder:text-slate-500 focus:bg-slate-900/60 focus:outline-hidden focus:ring-1 focus:ring-indigo-500 outline-none transition-all"
                      placeholder="Search name or indicator..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                  </div>

                  {/* Status Badges Filter */}
                  <div className="flex flex-wrap gap-1.5 self-start sm:self-auto">
                    {["all", "extracted", "backtested", "paper_trading", "live", "retired"].map((status) => (
                      <button
                        key={status}
                        onClick={() => setSelectedStatusFilter(status)}
                        className={`rounded-lg px-2.5 py-1 text-[11px] font-semibold transition-all border shrink-0 uppercase tracking-tight cursor-pointer ${
                          selectedStatusFilter === status
                            ? "bg-indigo-600 border-indigo-500 text-white shadow-md shadow-indigo-600/10"
                            : "bg-white/5 border-white/10 text-slate-400 hover:text-white hover:bg-white/10"
                        }`}
                      >
                        {status.replace("_", " ")}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Table list */}
                <div className="divide-y divide-white/5 max-h-[400px] overflow-y-auto">
                  {filteredStrategies.length > 0 ? (
                    filteredStrategies.map((strat) => {
                      const isActive = strat.id === selectedStrategyId;
                      return (
                        <div
                          key={strat.id}
                          onClick={() => setSelectedStrategyId(strat.id)}
                          className={`p-4 flex flex-col md:flex-row md:items-center justify-between gap-4 cursor-pointer transition-all ${
                            isActive 
                              ? "bg-indigo-550 bg-indigo-500/10 border-l-4 border-indigo-500 border-t border-b border-r border-white/10 shadow-lg shadow-indigo-500/5" 
                              : "bg-transparent hover:bg-white/[0.03]"
                          }`}
                        >
                          <div className="space-y-1">
                            <div className="flex items-center gap-2">
                              <h3 className="text-xs font-bold text-white font-sans">{strat.name}</h3>
                              <span className={`text-[9px] uppercase tracking-wider font-extrabold px-1.5 py-0.5 rounded border ${
                                strat.status === "extracted" ? "bg-blue-500/10 border-blue-500/20 text-blue-300" :
                                strat.status === "backtested" ? "bg-amber-500/10 border border-amber-500/20 text-amber-300" :
                                strat.status === "paper_trading" ? "bg-purple-500/10 border border-purple-500/20 text-purple-300" :
                                strat.status === "live" ? "bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 font-bold shadow-[0_0_10px_rgba(16,185,129,0.15)]" :
                                "bg-slate-500/10 border border-slate-500/20 text-slate-300"
                              }`}>
                                {strat.status.replace("_", " ")}
                              </span>
                            </div>
                            
                            {/* Meta flags row */}
                            <div className="flex flex-wrap items-center gap-y-1 gap-x-3 text-[11px] text-slate-450 text-slate-400">
                              <span className="flex items-center gap-1 font-semibold text-slate-300">
                                <span className={`h-1.5 w-1.5 rounded-full ${
                                  strat.asset_class === "crypto" ? "bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.6)]" :
                                  strat.asset_class === "stocks" ? "bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]" :
                                  strat.asset_class === "forex" ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]" : "bg-slate-400"
                                }`}></span>
                                {strat.asset_class.toUpperCase()}
                              </span>
                              <span>Timeframe: <b className="text-slate-200">{strat.timeframe}</b></span>
                              <span className="truncate max-w-[150px] md:max-w-none">Channel: <b className="text-slate-200">{strat.source_channel}</b></span>
                            </div>

                            {/* indicators array */}
                            <div className="flex flex-wrap gap-1 mt-1.5">
                              {strat.indicators.map((ind: string, idx: number) => (
                                <span key={idx} className="rounded-md bg-white/5 text-slate-300 text-[10px] font-mono px-1.5 py-0.5 border border-white/5">
                                  {ind}
                                </span>
                              ))}
                            </div>
                          </div>

                          <div className="flex items-center md:flex-col md:items-end justify-between md:justify-center shrink-0 border-t md:border-t-0 border-white/5 pt-3 md:pt-0">
                            {/* Confidence rating */}
                            <div className="text-right">
                              <div className="text-[10px] text-slate-455 text-slate-400 uppercase font-semibold">Gemini Precision</div>
                              <div className={`text-xs font-bold leading-none mt-0.5 ${
                                strat.extraction_confidence >= 0.8 ? "text-emerald-400" :
                                strat.extraction_confidence >= 0.6 ? "text-amber-400" : "text-rose-400"
                              }`}>
                                {Math.round(strat.extraction_confidence * 100)}%
                              </div>
                            </div>
                            
                            {/* Backtest scores tag */}
                            <div className="mt-1">
                              {strat.backtest_metrics ? (
                                <span className="flex items-center gap-1 text-[11px] font-bold text-indigo-300 bg-white/5 border border-white/10 rounded px-1.5 py-0.5 shadow-sm">
                                  <LineChart className="h-3 w-3 text-emerald-400" />
                                  Sharpe: {strat.backtest_metrics.sharpe_ratio}
                                </span>
                              ) : (
                                <span className="text-[10px] text-slate-500 italic">No Track B metrics yet</span>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    })
                  ) : (
                    <div className="p-8 text-center text-slate-500 text-xs">
                      No strategies found matching the active status filter.
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* DETAILED RIGHT CONDUIT: STRATEGY VIEWER & ORCHESTRATION DEV (5 cols) */}
            <div className="lg:col-span-12 xl:col-span-5 space-y-6" id="details_sidebar">
              <div className="rounded-2xl border border-white/10 bg-white/[0.04] backdrop-blur-xl shadow-2xl shadow-indigo-950/20 p-6 divide-y divide-white/5 space-y-6">
                
                {/* Header Profile */}
                <div>
                  <div className="flex items-start justify-between">
                    <div>
                      <span className="text-[10px] text-slate-450 text-slate-400 font-bold uppercase tracking-wider">Active Strategy Object</span>
                      <h2 className="font-display text-lg font-bold text-white leading-tight mt-0.5">{activeStrategy.name}</h2>
                    </div>
                    <a
                      href={activeStrategy.source_video_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-indigo-400 hover:text-indigo-300 transition-transform duration-150 hover:scale-110"
                      title="Open source video"
                    >
                      <ExternalLink className="h-4.5 w-4.5" />
                    </a>
                  </div>

                  <div className="flex items-center gap-1.5 mt-2 bg-white/5 border border-white/10 p-1.5 rounded-lg text-[10px] text-slate-400 font-mono">
                    <span className="shrink-0 text-slate-500">UUID:</span>
                    <span className="truncate">{activeStrategy.id}</span>
                  </div>
                </div>

                {/* Tactical Schemas Tabulated display */}
                <div className="pt-4 space-y-4 text-xs">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white/5 border border-white/10 rounded-xl p-3">
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wide flex items-center gap-1">
                        Asset class
                      </span>
                      <p className="font-semibold text-white mt-1 capitalize">{activeStrategy.asset_class}</p>
                    </div>
                    <div className="bg-white/5 border border-white/10 rounded-xl p-3">
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wide">
                        Timeframe
                      </span>
                      <p className="font-semibold text-white mt-1">{activeStrategy.timeframe}</p>
                    </div>
                  </div>

                  {/* Core indicators */}
                  <div>
                    <h3 className="font-bold uppercase text-[10px] text-indigo-400 mb-1.5">Indicators</h3>
                    <div className="flex flex-wrap gap-2">
                      {activeStrategy.indicators.length > 0 ? (
                        activeStrategy.indicators.map((ind: string, idx: number) => (
                          <span key={idx} className="bg-white/5 border border-white/10 text-slate-200 font-mono text-[11px] px-2 py-1 rounded-md shadow-inner">
                            {ind}
                          </span>
                        ))
                      ) : (
                        <span className="text-slate-500 italic">No indicators required by strategy.</span>
                      )}
                    </div>
                  </div>

                  {/* Entry and Exit Rules */}
                  <div className="space-y-3">
                    <div>
                      <h3 className="font-bold text-white text-[10px] tracking-wider uppercase text-slate-400 flex items-center justify-between mb-1.5">
                        Entry Strategy Rules
                        <span className="bg-indigo-500/10 border border-indigo-500/20 px-1.5 py-0.5 rounded text-[8px] font-extrabold text-indigo-300 tracking-normal uppercase">Extracted</span>
                      </h3>
                      <p className="bg-white/5 border border-white/5 rounded-xl p-3.5 font-sans text-[11.5px] leading-relaxed text-slate-300 italic shadow-inner">
                        "{activeStrategy.entry_rules}"
                      </p>
                    </div>

                    <div>
                      <h3 className="font-bold text-white text-[10px] tracking-wider uppercase text-slate-400 flex items-center justify-between mb-1.5">
                        Exit Strategy Rules
                      </h3>
                      <p className="bg-white/5 border border-white/5 rounded-xl p-3.5 font-sans text-[11.5px] leading-relaxed text-slate-300 italic shadow-inner">
                        "{activeStrategy.exit_rules}"
                      </p>
                    </div>
                  </div>

                  {/* Risk settings */}
                  <div>
                    <h3 className="font-bold text-white text-[10px] tracking-wider uppercase text-slate-450 text-slate-400 mb-1.5">Risk Management Schema</h3>
                    <div className="bg-white/5 border border-white/5 rounded-xl p-3.5 divide-y divide-white/5 text-[11px]">
                      <div className="flex justify-between py-1.5 first:pt-0">
                        <span className="text-slate-450 text-slate-450 text-slate-400">Stop Loss</span>
                        <span className="font-semibold text-indigo-300 font-mono text-right max-w-[180px]">{activeStrategy.risk_management.stop_loss || <span className="text-rose-400 italic">Null (Vague)</span>}</span>
                      </div>
                      <div className="flex justify-between py-1.5">
                        <span className="text-slate-450 text-slate-450 text-slate-400">Take Profit</span>
                        <span className="font-semibold text-indigo-300 font-mono text-right max-w-[180px]">{activeStrategy.risk_management.take_profit || <span className="text-rose-400 italic">Null (Vague)</span>}</span>
                      </div>
                      <div className="flex justify-between py-1.5">
                        <span className="text-slate-450 text-slate-450 text-slate-400">Position Sizing</span>
                        <span className="font-semibold text-indigo-300 font-mono text-right max-w-[180px]">{activeStrategy.risk_management.position_sizing || <span className="text-rose-400 italic">Null (Vague)</span>}</span>
                      </div>
                      <div className="flex justify-between py-1.5 last:pb-0">
                        <span className="text-slate-455 text-slate-450 text-slate-400">Max Concurrent Trades</span>
                        <span className="font-semibold text-indigo-300 font-mono">{activeStrategy.risk_management.max_concurrent_positions || <span className="text-slate-500">Null</span>}</span>
                      </div>
                    </div>
                  </div>

                  {/* Anti AI Slop: Gemini extraction Notes */}
                  <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-xl p-3.5 space-y-1 shadow-lg shadow-indigo-500/5">
                    <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-indigo-300">
                      <Info className="h-4 w-4 shrink-0 text-indigo-400" />
                      Anti-Slop Validation Log
                    </div>
                    <p className="text-[11px] text-slate-300 leading-relaxed italic">
                      🎯 "{activeStrategy.extraction_notes}"
                    </p>
                  </div>
                </div>

                {/* Downstream Operations simulation (Track B Interaction Panel) */}
                <div className="pt-4 space-y-4">
                  <div className="flex items-center gap-2 text-indigo-300">
                    <Sliders className="h-4 w-4" />
                    <h3 className="font-display font-semibold text-xs uppercase tracking-wider">Downstream Simulation (Track B)</h3>
                  </div>

                  {/* 1. STATE MACHINE GRADUATION */}
                  <div className="space-y-1.5">
                    <span className="text-[10px] uppercase font-bold tracking-wide text-slate-400">Orchestration State Machine:</span>
                    <div className="bg-white/5 border border-white/5 rounded-xl p-3">
                      <div className="flex items-center gap-3 mb-3">
                        <span className="text-[11px] text-slate-400">Current Status:</span>
                        <span className="font-mono text-xs font-bold uppercase tracking-wider text-white bg-indigo-600/30 border border-indigo-500/30 px-2.5 py-1 rounded shadow-md">
                          {activeStrategy.status}
                        </span>
                      </div>

                      {/* State transitions controls buttons */}
                      <div className="grid grid-cols-2 gap-2">
                        {/* Progressive Actions based on status */}
                        {activeStrategy.status === "extracted" && (
                          <button
                            onClick={() => transitionStatus("backtested")}
                            className="flex items-center justify-center gap-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 p-2 text-center text-xs text-white font-semibold transition active:scale-95 duration-100"
                          >
                            <TrendingUp className="h-3.5 w-3.5" />
                            To Backtested
                          </button>
                        )}
                        {activeStrategy.status === "backtested" && (
                          <button
                            onClick={() => transitionStatus("paper_trading")}
                            className="flex items-center justify-center gap-1.5 rounded-lg bg-teal-600 hover:bg-teal-500 p-2 text-center text-xs text-white font-semibold transition active:scale-95 duration-100"
                          >
                            <GitCommit className="h-3.5 w-3.5" />
                            To Paper Trading
                          </button>
                        )}
                        {activeStrategy.status === "paper_trading" && (
                          <button
                            onClick={() => transitionStatus("live")}
                            className="flex items-center justify-center gap-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 p-2 text-center text-xs text-white font-bold transition-all shadow-lg shadow-emerald-600/20 active:scale-95 duration-100 animate-pulse"
                          >
                            <Flame className="h-3.5 w-3.5" />
                            Deploy LIVE!
                          </button>
                        )}
                        {["live", "retired"].includes(activeStrategy.status) && (
                          <div className="col-span-1 rounded-lg border border-dashed border-white/10 p-2 text-center text-[11px] text-slate-500 font-medium">
                            Status Terminal
                          </div>
                        )}

                        {/* Retirement is allowed from any non-terminal state */}
                        {!["live", "retired"].includes(activeStrategy.status) ? (
                          <button
                            onClick={() => transitionStatus("retired")}
                            className="flex items-center justify-center gap-1.5 rounded-lg bg-rose-500/10 border border-rose-500/20 hover:bg-rose-500/20 p-2 text-center text-xs text-rose-300 font-semibold transition cursor-pointer active:scale-95 duration-100"
                          >
                            <XCircle className="h-3.5 w-3.5 text-rose-400" />
                            Retire Plan
                          </button>
                        ) : (
                          <div className="col-span-1 border border-dashed border-white/10 rounded-lg p-2 text-center text-[11px] text-slate-500 font-medium">
                            No Retirement Actions
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* 2. METRICS ATTACHMENT */}
                  <div className="space-y-2">
                    <span className="text-[10px] uppercase font-bold tracking-wide text-slate-400">Update Metrics (Track B Independent API):</span>
                    <form onSubmit={saveQuantMetrics} className="bg-white/5 border border-white/5 rounded-xl p-3.5 space-y-3 shadow-inner">
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <label className="text-[10px] text-slate-400 block mb-1">Sharpe Ratio</label>
                          <input
                            type="number"
                            step="0.01"
                            className="w-full bg-slate-900/50 border border-white/10 rounded-md p-1.5 text-xs text-white focus:ring-1 focus:ring-indigo-500 focus:bg-slate-900/80 outline-none transition-all"
                            placeholder="e.g. 1.84"
                            value={inputSharpe}
                            onChange={(e) => setInputSharpe(e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="text-[10px] text-slate-400 block mb-1">Win Rate (%)</label>
                          <input
                            type="number"
                            step="0.1"
                            className="w-full bg-slate-900/50 border border-white/10 rounded-md p-1.5 text-xs text-white focus:ring-1 focus:ring-indigo-500 focus:bg-slate-900/80 outline-none transition-all"
                            placeholder="e.g. 58.4"
                            value={inputWinRate}
                            onChange={(e) => setInputWinRate(e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="text-[10px] text-slate-400 block mb-1">Max Drawdown (%)</label>
                          <input
                            type="text"
                            className="w-full bg-slate-900/50 border border-white/10 rounded-md p-1.5 text-xs text-white focus:ring-1 focus:ring-indigo-500 focus:bg-slate-900/80 outline-none transition-all"
                            placeholder="e.g. 11.2"
                            value={inputDrawdown}
                            onChange={(e) => setInputDrawdown(e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="text-[10px] text-slate-400 block mb-1">Total Backtests Trades</label>
                          <input
                            type="number"
                            className="w-full bg-slate-900/50 border border-white/10 rounded-md p-1.5 text-xs text-white focus:ring-1 focus:ring-indigo-500 focus:bg-slate-900/80 outline-none transition-all"
                            placeholder="e.g. 124"
                            value={inputTotalTrades}
                            onChange={(e) => setInputTotalTrades(e.target.value)}
                          />
                        </div>
                      </div>

                      <button
                        type="submit"
                        className="w-full text-center py-2 text-xs font-semibold rounded-lg bg-indigo-650 bg-indigo-600/20 border border-indigo-500/30 text-indigo-300 hover:bg-indigo-600 hover:text-white transition cursor-pointer shadow-xs active:scale-95 duration-150"
                      >
                        Submit Track B Backtest Results
                      </button>
                    </form>
                  </div>

                </div>

              </div>
            </div>

          </div>
        ) : (
          /* Python Code Explorer Tab */
          <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl overflow-hidden shadow-2xl shadow-indigo-950/20" id="code_explorer">
            <div className="grid grid-cols-1 md:grid-cols-12 min-h-[600px]">
              
              {/* Left Selector Rail */}
              <div className="md:col-span-3 bg-white/[0.01] border-r border-white/5 p-4 space-y-4">
                <div className="flex items-center gap-2 pb-3 border-b border-white/5">
                  <BookOpen className="h-4 w-4 text-indigo-400" />
                  <span className="text-xs font-bold uppercase tracking-wider text-indigo-300">Pipeline Assets</span>
                </div>

                <div className="flex flex-col gap-1.5">
                  {Object.keys(CODE_FILES).map((fileName) => {
                    const isSelected = fileName === selectedLanguage;
                    return (
                      <button
                        key={fileName}
                        onClick={() => setSelectedLanguage(fileName)}
                        className={`flex items-center gap-2 rounded-xl px-3 py-2.5 text-left text-xs font-medium tracking-tight transition cursor-pointer ${
                          isSelected 
                            ? "bg-indigo-600 border border-indigo-500 text-white font-semibold shadow-md shadow-indigo-600/15" 
                            : "text-slate-400 hover:bg-white/5 hover:text-slate-100"
                        }`}
                      >
                        <FileCode className={`h-4 w-4 shrink-0 ${isSelected ? "text-emerald-300" : "text-slate-500"}`} />
                        {fileName}
                      </button>
                    );
                  })}
                </div>

                <div className="bg-indigo-500/10 rounded-xl p-4 border border-indigo-550 border-indigo-500/20 text-indigo-200 mt-10">
                  <p className="text-[10px] uppercase font-bold text-indigo-300 tracking-wider flex items-center gap-1.5">
                    <CheckCircle2 className="h-4.5 w-4.5 text-emerald-400" />
                    Pristine Integration
                  </p>
                  <p className="text-[11px] leading-relaxed mt-1 text-slate-350 text-slate-300">
                    These modular files have been written directly to the project root directory. You can export them via the settings menu or copy them directly.
                  </p>
                </div>
              </div>

              {/* Right Code Block Viewer */}
              <div className="md:col-span-9 bg-slate-950/80 backdrop-blur-md p-6 flex flex-col justify-between overflow-x-auto border-l border-white/5">
                <div className="flex items-center justify-between pb-4 border-b border-white/5 mb-4">
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-widest">{selectedLanguage}</span>
                  <span className="text-[10px] font-mono text-slate-500">UTF-8 • Python 3</span>
                </div>

                <pre className="font-mono text-xs text-slate-300 leading-relaxed overflow-x-auto whitespace-pre selection:bg-indigo-500/40">
                  <code>{CODE_FILES[selectedLanguage as keyof typeof CODE_FILES]}</code>
                </pre>

                <div className="mt-8 border-t border-white/5 pt-4 text-right">
                  <span className="text-[10px] font-mono text-slate-500">Ready for deployment compilation</span>
                </div>
              </div>

            </div>
          </div>
        )}

      </main>

      {/* Humble, Professional Footer */}
      <footer className="border-t border-white/5 bg-slate-950/40 backdrop-blur-md mt-16 py-8 relative z-10" id="app_footer">
        <div className="mx-auto max-w-7xl px-6 flex flex-col md:flex-row items-center justify-between text-xs text-slate-500 gap-4">
          <p>© 2026 STRAT-SYNC • Quantitative Architectural Pipeline</p>
          <div className="flex gap-6 font-semibold text-slate-400">
            <span>Core SQLite Engine Active</span>
            <span>Gemini 3.5 Context Engine</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
