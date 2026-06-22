# -*- coding: utf-8 -*-
"""
SQLite Storage Layer for Trading Strategy Pipeline (Track A).

This module handles persistent SQLite storage of StrategyObjects. The database schema 
and serialization techniques are designed to translate cleanly to PostgreSQL later 
(using JSON/JSONB types in Postgres).

Design Decisions:
1. Postgres Compatibility: Since SQLite does not have native JSON or list structures, 
   nested objects (like `indicators`, `risk_management`, `backtest_metrics`, and 
   `paper_trading_metrics`) are stored as serialized JSON text strings. This enables 
   an immediate, zero-friction upgrade to Postgres standard JSONB tables later.
2. State Machine Enforcement: The service enforces a strict, robust status-transition machine:
   - Progress: extracted -> backtested -> paper_trading -> live
   - Retirement: any non-terminal state (extracted, backtested, paper_trading) can move to "retired"
   - Terminality: "live" and "retired" are terminal; any transition out of them is blocked.
3. Decoupled Metrics Updates: Backtest metrics and paper trading metrics can be updated 
   separately and independently of the pipeline state.
"""

import json
import sqlite3
import os
from uuid import UUID
from typing import List, Optional, Dict, Any, Union
from extractor import StrategyObject, RiskManagement


class StateMachineError(ValueError):
    """Raised when an invalid status transition is requested."""
    pass


class StorageService:
    """Persistent SQLite Storage service managing strategy collection lifecycle."""

    # Flow progression table mapping current status to valid destinations
    VALID_TRANSITIONS = {
        "extracted": ["backtested", "retired"],
        "backtested": ["paper_trading", "retired"],
        "paper_trading": ["live", "retired"],
        "live": [],        # Terminal state
        "retired": []      # Terminal state
    }

    def __init__(self, db_path: str = "strategies.db"):
        """
        Initializes the database connection and creates tables if missing.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._create_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Helper to construct a thread-safe connection with dictionary-like row factories."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        """Initializes the database schema matching StrategyObject fields flatly."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    source_video_url TEXT NOT NULL,
                    source_video_title TEXT NOT NULL,
                    source_channel TEXT NOT NULL,
                    extracted_at TEXT NOT NULL,
                    
                    -- Extracted Strategy attributes
                    name TEXT NOT NULL,
                    asset_class TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    indicators TEXT NOT NULL,          -- Serialized JSON array
                    entry_rules TEXT NOT NULL,
                    exit_rules TEXT NOT NULL,
                    risk_management TEXT NOT NULL,     -- Serialized RiskManagement JSON dict
                    extraction_confidence REAL NOT NULL,
                    extraction_notes TEXT NOT NULL,
                    
                    -- Quant metrics placeholders (written by Track B)
                    backtest_metrics TEXT,             -- Serialized JSON dict or Null
                    paper_trading_metrics TEXT         -- Serialized JSON dict or Null
                )
            """)
            conn.commit()

    def _row_to_strategy(self, row: sqlite3.Row) -> StrategyObject:
        """Helper to convert a sqlite3 Row back into a fully typed StrategyObject."""
        # Deserialize JSON fields
        indicators_list = json.loads(row["indicators"])
        risk_mgt_dict = json.loads(row["risk_management"])
        
        backtest_metrics = None
        if row["backtest_metrics"] is not None:
            backtest_metrics = json.loads(row["backtest_metrics"])
            
        paper_trading_metrics = None
        if row["paper_trading_metrics"] is not None:
            paper_trading_metrics = json.loads(row["paper_trading_metrics"])

        return StrategyObject(
            id=UUID(row["id"]),
            status=row["status"],
            source_video_url=row["source_video_url"],
            source_video_title=row["source_video_title"],
            source_channel=row["source_channel"],
            extracted_at=row["extracted_at"],
            name=row["name"],
            asset_class=row["asset_class"],
            timeframe=row["timeframe"],
            indicators=indicators_list,
            entry_rules=row["entry_rules"],
            exit_rules=row["exit_rules"],
            risk_management=RiskManagement(**risk_mgt_dict),
            extraction_confidence=row["extraction_confidence"],
            extraction_notes=row["extraction_notes"],
            backtest_metrics=backtest_metrics,
            paper_trading_metrics=paper_trading_metrics
        )

    def insert_strategy(self, strategy: StrategyObject) -> None:
        """
        Inserts a new StrategyObject into the SQLite database.
        
        Args:
            strategy: Fully typed StrategyObject state.
        """
        # Serialize fields for standard DB safe storage
        indicators = json.dumps(strategy.indicators)
        risk_management = json.dumps(strategy.risk_management.model_dump())
        backtest = json.dumps(strategy.backtest_metrics) if strategy.backtest_metrics is not None else None
        paper_trading = json.dumps(strategy.paper_trading_metrics) if strategy.paper_trading_metrics is not None else None

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO strategies (
                    id, status, source_video_url, source_video_title, source_channel, extracted_at,
                    name, asset_class, timeframe, indicators, entry_rules, exit_rules, risk_management,
                    extraction_confidence, extraction_notes, backtest_metrics, paper_trading_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(strategy.id),
                    strategy.status,
                    strategy.source_video_url,
                    strategy.source_video_title,
                    strategy.source_channel,
                    strategy.extracted_at,
                    strategy.name,
                    strategy.asset_class,
                    strategy.timeframe,
                    indicators,
                    strategy.entry_rules,
                    strategy.exit_rules,
                    risk_management,
                    strategy.extraction_confidence,
                    strategy.extraction_notes,
                    backtest,
                    paper_trading
                )
            )
            conn.commit()

    def get_strategy(self, strategy_id: Union[UUID, str]) -> Optional[StrategyObject]:
        """
        Retrieves a StrategyObject by its ID.
        
        Args:
            strategy_id: UUID or string representing the unique ID of the strategy.
            
        Returns:
            Optional[StrategyObject]: The retrieved strategy or None if missing.
        """
        id_str = str(strategy_id)
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM strategies WHERE id = ?", (id_str,))
            row = cursor.fetchone()
            if row is not None:
                return self._row_to_strategy(row)
        return None

    def list_strategies(self, status: Optional[str] = None) -> List[StrategyObject]:
        """
        Retrieves a list of saved strategies, optionally filtered by status.
        
        Args:
            status: Optional string filter ('extracted', 'backtested', etc.)
            
        Returns:
            List[StrategyObject]: List of matched StrategyObjects.
        """
        with self._get_connection() as conn:
            if status:
                cursor = conn.execute("SELECT * FROM strategies WHERE status = ? ORDER BY extracted_at DESC", (status,))
            else:
                cursor = conn.execute("SELECT * FROM strategies ORDER BY extracted_at DESC")
            rows = cursor.fetchall()
            return [self._row_to_strategy(row) for row in rows]

    def update_status(self, strategy_id: Union[UUID, str], new_status: str) -> None:
        """
        Updates the progression status of a trading strategy, enforcing the state machine.
        
        Valid Transitions:
            - Progress: extracted -> backtested -> paper_trading -> live
            - Retirement: any non-terminal state -> retired
            - Terminality: live & retired cannot transition to any state.
            
        Args:
            strategy_id: UUID or string identifier.
            new_status: The target pipeline status.
            
        Raises:
            ValueError: If strategy does not exist.
            StateMachineError: If the requested transition is illegal under our business machine rules.
        """
        id_str = str(strategy_id)
        with self._get_connection() as conn:
            # 1. Look up current status
            cursor = conn.execute("SELECT status FROM strategies WHERE id = ?", (id_str,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Strategy with ID {id_str} not found in database.")
            
            curr_status = row["status"]
            allowed_next = self.VALID_TRANSITIONS.get(curr_status, [])

            # 2. Assert transition legality
            if new_status not in allowed_next:
                raise StateMachineError(
                    f"Illegal State Transition: Cannot transition strategy from '{curr_status}' to '{new_status}'. "
                    f"Allowed target transitions are: {allowed_next if allowed_next else 'None (Terminal State)'}."
                )

            # 3. Perform update
            conn.execute("UPDATE strategies SET status = ? WHERE id = ?", (new_status, id_str))
            conn.commit()

    def update_backtest_metrics(self, strategy_id: Union[UUID, str], metrics: Dict[str, Any]) -> None:
        """
        Independently writes backtest metrics into a strategy record.
        
        Args:
            strategy_id: UUID or string identifier.
            metrics: Arbitrary dictionary representing results of quant backtests.
        """
        id_str = str(strategy_id)
        metrics_json = json.dumps(metrics)
        with self._get_connection() as conn:
            # Verify existence
            cursor = conn.execute("SELECT id FROM strategies WHERE id = ?", (id_str,))
            if not cursor.fetchone():
                raise ValueError(f"Strategy with ID {id_str} not found in database.")

            conn.execute("UPDATE strategies SET backtest_metrics = ? WHERE id = ?", (metrics_json, id_str))
            conn.commit()

    def update_paper_trading_metrics(self, strategy_id: Union[UUID, str], metrics: Dict[str, Any]) -> None:
        """
        Independently writes paper trading operational metrics into a strategy record.
        
        Args:
            strategy_id: UUID or string identifier.
            metrics: Arbitrary dictionary representing paper execution analytics.
        """
        id_str = str(strategy_id)
        metrics_json = json.dumps(metrics)
        with self._get_connection() as conn:
            # Verify existence
            cursor = conn.execute("SELECT id FROM strategies WHERE id = ?", (id_str,))
            if not cursor.fetchone():
                raise ValueError(f"Strategy with ID {id_str} not found in database.")

            conn.execute("UPDATE strategies SET paper_trading_metrics = ? WHERE id = ?", (metrics_json, id_str))
            conn.commit()
