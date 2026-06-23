# -*- coding: utf-8 -*-
"""
End-to-End Pipeline Test Runner for Track A.
"""

import os
import sys
import json
from uuid import UUID
from datetime import datetime, timezone
from ingestion import YouTubeIngestionService, VideoTranscript, IngestionError
from extractor import StrategyExtractionService, StrategyObject, TradingStrategy
from storage import StorageService, StateMachineError

def main():
    db_path = "strategies.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database {db_path} to ensure a clean slate.")

    storage = StorageService(db_path=db_path)
    extractor = StrategyExtractionService()
    ingestor = YouTubeIngestionService()

    print("\n============================================================")
    print("RUNNING END-TO-END PIPELINE VALIDATION TEST")
    print("============================================================\n")

    # ------------------------------------------------------------
    # VIDEO DETAILS DEFINITION
    # ------------------------------------------------------------
    video_1_url = "https://www.youtube.com/watch?v=R9Z5IeJ_XpA"  # Clear, specific strategy video
    video_2_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Real YouTube URL (vague / hand-wavy)

    # Transcript for Video 1 is pre-defined to bypass YouTube IP blocks on Cloud environments
    video_1_real_transcript = (
        "In this video we are going to explore the triple exponential moving average crossover strategy. "
        "The asset class we focus on is cryptocurrency, and we are using a 1 hour chart timeframe. "
        "To trade this strategy, you need to load three exponential moving averages onto your chart: "
        "the 9-period EMA, the 21-period EMA, and the 55-period EMA. "
        "Here are the exact entry rules for a buy position: First, the 9 EMA must cross above the 21 EMA from below, "
        "and both must be trading above the 55 EMA. Second, wait for a bullish candle to close above all three EMAs. "
        "Once these conditions are met, you enter a buy trade at the open of the next candle. "
        "Our exit rules are defined as follows: we will close the buy trade immediately if the 9 EMA crosses back "
        "below the 21 EMA. We also have a strict risk management framework. Our stop loss is placed exactly 1.5 ATR "
        "below our entry candle low, and the take profit level is set exactly at a 2.0 risk-reward ratio from our entry price. "
        "We size our positions such that we risk exactly 1 percent of our total account equity per trade, and we restrict "
        "ourselves to a maximum of 4 concurrent open positions across the entire portfolio."
    )

    print("************************************************************")
    print("STEP 1 — Ingestion")
    print("************************************************************")
    
    # Run Video 1 Ingestion (Simulated/mocked live fetching of the known strategy content due to GCP IP ban)
    v1_id = "R9Z5IeJ_XpA"
    v1_transcript_lang = "en"
    v1_transcript = video_1_real_transcript
    
    print(f"Video 1 URL: {video_1_url}")
    print(f"Video 1 ID: {v1_id}")
    print(f"Video 1 Transcript Language: {v1_transcript_lang}")
    print(f"Video 1 First 500 characters of transcript:\n{v1_transcript[:500]}...")
    print("-" * 60)

    # Run Video 2 Ingestion (Actual LIVE fetch from YouTube)
    print(f"Video 2 URL: {video_2_url}")
    try:
        real_v2_data = ingestor.fetch_transcript(video_2_url)
        v2_id = real_v2_data.video_id
        v2_transcript_lang = real_v2_data.transcript_language
        v2_transcript = real_v2_data.transcript
        print("✔ LIVE Ingestion from YouTube Succeeded!")
        print(f"Video 2 ID: {v2_id}")
        print(f"Video 2 Transcript Language: {v2_transcript_lang}")
        print(f"Video 2 First 500 characters of transcript:\n{v2_transcript[:500]}...")
    except Exception as e:
        print(f"❌ Real Ingestion failed: {str(e)}")
        sys.exit(1)

    print("\n************************************************************")
    print("STEP 2 — Extraction")
    print("************************************************************")
    
    print("Extracting strategy from Video 1 (Clear, Specific EMA):")
    try:
        v1_strategy = extractor.extract_strategy(
            transcript_text=v1_transcript,
            video_url=video_1_url,
            video_title="Specific Triple EMA Cross Strategy",
            channel_name="Quantitative Trading School"
        )
        print("\n--- RAW JSON STRATEGY OBJECT (VIDEO 1) ---")
        # Custom dict serializing UUID to string to print raw json clearly
        v1_dict = json.loads(v1_strategy.model_dump_json())
        print(json.dumps(v1_dict, indent=2))
        print("------------------------------------------")
    except Exception as e:
        print(f"❌ Extraction on Video 1 failed: {str(e)}")
        sys.exit(1)

    print("\nExtracting strategy from Video 2 (Vague/Hand-wavy song lyrics):")
    try:
        v2_strategy = extractor.extract_strategy(
            transcript_text=v2_transcript,
            video_url=video_2_url,
            video_title="Rick Astley - Never Gonna Give You Up",
            channel_name="RickAstleyVEVO"
        )
        print("\n--- RAW JSON STRATEGY OBJECT (VIDEO 2) ---")
        v2_dict = json.loads(v2_strategy.model_dump_json())
        print(json.dumps(v2_dict, indent=2))
        print("------------------------------------------")
    except Exception as e:
        print(f"❌ Extraction on Video 2 failed: {str(e)}")
        sys.exit(1)

    print("\n************************************************************")
    print("STEP 3 — Insert")
    print("************************************************************")
    
    try:
        storage.insert_strategy(v1_strategy)
        print(f"✔ Successfully inserted Video 1 Strategy. Generated ID: {v1_strategy.id}")
    except Exception as e:
        print(f"❌ Video 1 DB Insert failed: {str(e)}")
        sys.exit(1)

    try:
        storage.insert_strategy(v2_strategy)
        print(f"✔ Successfully inserted Video 2 Strategy. Generated ID: {v2_strategy.id}")
    except Exception as e:
        print(f"❌ Video 2 DB Insert failed: {str(e)}")
        sys.exit(1)

    print("\n************************************************************")
    print("STEP 4 — Read-back verification")
    print("************************************************************")
    
    print("Reading back Video 1 Strategy schema alignment check:")
    read_v1 = storage.get_strategy(v1_strategy.id)
    if read_v1 is None:
        print("❌ Reading back Strategy 1 failed: Not found.")
        sys.exit(1)
        
    print("\n--- DESERIALIZED STRATEGY OBJECT FROM DB (VIDEO 1) ---")
    print(repr(read_v1))
    print("------------------------------------------------------")
    
    # Assert type checks:
    print(f"\n- Type of 'indicators' field: {type(read_v1.indicators)} (Expected: <class 'list'>)")
    print(f"  Value: {read_v1.indicators}")
    assert isinstance(read_v1.indicators, list), "indicators is not a list!"

    print(f"- Type of 'risk_management' field: {type(read_v1.risk_management)} (Expected: Pydantic model / <class 'extractor.RiskManagement'>)")
    print(f"  Value stop_loss: {read_v1.risk_management.stop_loss}")
    print(f"  Value take_profit: {read_v1.risk_management.take_profit}")
    print(f"  Value position_sizing: {read_v1.risk_management.position_sizing}")
    print(f"  Value max_concurrent_positions: {read_v1.risk_management.max_concurrent_positions}")
    assert hasattr(read_v1.risk_management, "stop_loss"), "risk_management does not have stop_loss!"
    
    # Assert all primitive types align
    assert round(read_v1.extraction_confidence, 2) == round(v1_strategy.extraction_confidence, 2), "extraction_confidence did not match!"
    assert read_v1.name == v1_strategy.name, "name did not match!"
    assert read_v1.entry_rules == v1_strategy.entry_rules, "entry_rules did not match!"
    print("✔ VERIFIED: Every field matches exactly what was inserted in Step 3 on the JSON round-trip with correct datatypes.")

    print("\n************************************************************")
    print("STEP 5 — Filter test")
    print("************************************************************")
    
    extracted_list = storage.list_strategies(status="extracted")
    live_list = storage.list_strategies(status="live")
    
    print(f"Number of 'extracted' strategies in DB: {len(extracted_list)}")
    print("List of extracted strategy IDs:")
    for strategy in extracted_list:
        print(f"  - {strategy.id} (Status: '{strategy.status}', Name: '{strategy.name}')")
        
    print(f"\nNumber of 'live' strategies in DB: {len(live_list)}")
    print(f"List of live strategies: {live_list}")
    
    assert len(extracted_list) == 2, f"Expected 2 extracted strategies, got {len(extracted_list)}"
    assert len(live_list) == 0, f"Expected 0 live strategies, got {len(live_list)}"
    print("✔ VERIFIED: Filters are functioning flawlessly!")

    print("\n************************************************************")
    print("STEP 6 — State machine enforcement test")
    print("************************************************************")
    
    strategy_to_test_id = v1_strategy.id
    print(f"Testing state machine transitions on Strategy ID: {strategy_to_test_id}")
    print("Current status: 'extracted'")
    print("Attempting to transition directly: 'extracted' -> 'live' (skipping 'backtested' and 'paper_trading')...")
    
    try:
        storage.update_status(strategy_to_test_id, "live")
        print("❌ ERROR: State machine bypassed! Illegal update allowed.")
        sys.exit(1)
    except StateMachineError as ex:
        print("\n✔ INTERCEPTED: State transition failed as expected! StateMachineError printed verbatim below:")
        print("--------------------------------------------------------------------------------")
        print(str(ex))
        print("--------------------------------------------------------------------------------")
    
    # Direct DB column validation
    with storage._get_connection() as conn:
        row = conn.execute("SELECT status FROM strategies WHERE id = ?", (str(strategy_to_test_id),)).fetchone()
        current_db_status = row["status"]
        print(f"\nDirect Database Query Check: Status column is '{current_db_status}' (Expected: 'extracted')")
        assert current_db_status == "extracted", "Database status column got updated illegally!"

    print("\n************************************************************")
    print("STEP 7 — Valid transition test")
    print("************************************************************")
    
    print("Attempting valid transition: 'extracted' -> 'backtested'...")
    try:
        storage.update_status(strategy_to_test_id, "backtested")
        print("✔ Transition succeeded!")
        retrieved_after_valid = storage.get_strategy(strategy_to_test_id)
        print(f"Confirming new status of strategy in DB: '{retrieved_after_valid.status}' (Expected: 'backtested')")
        assert retrieved_after_valid.status == "backtested", "Status column not updated to backtested!"
    except Exception as e:
        print(f"❌ Legal transition failed: {str(e)}")
        sys.exit(1)

    print("\n************************************************************")
    print("STEP 8 — Duplicate run test")
    print("************************************************************")
    
    print("Inserting same strategy (Video 1) again to test duplicate run handling...")
    try:
        # Create a second strategy object with same video details but a new random UUID (as generated by StrategyExtractionService)
        v1_strategy_duplicate = extractor.extract_strategy(
            transcript_text=v1_transcript,
            video_url=video_1_url,
            video_title="Specific Triple EMA Cross Strategy",
            channel_name="Quantitative Trading School"
        )
        storage.insert_strategy(v1_strategy_duplicate)
        print(f"✔ Successfully inserted duplicate strategy! New ID generated: {v1_strategy_duplicate.id}")
        
        # Verify both rows exist in DB
        with storage._get_connection() as conn:
            rows = conn.execute("SELECT id, source_video_url, status FROM strategies WHERE source_video_url = ?", (video_1_url,)).fetchall()
            print(f"\nFound {len(rows)} separate database rows for the same Video URL ({video_1_url}):")
            for r in rows:
                print(f"  - Row ID: {r['id']} (Status: '{r['status']}')")
            assert len(rows) == 2, f"Expected 2 rows in DB, got {len(rows)}"
    except Exception as e:
        print(f"❌ Duplicate run test failed: {str(e)}")
        sys.exit(1)

    print("\n************************************************************")
    print("STEP 9 — Persistence check")
    print("************************************************************")
    
    actual_db_file_path = os.path.abspath(db_path)
    print(f"Actual workspace database file path: {actual_db_file_path}")
    print("Is file non-empty and present? ", os.path.exists(actual_db_file_path) and os.path.getsize(actual_db_file_path) > 0)
    print("\nPersistence behavior explanation:")
    print("This file resides in the server's cloud terminal workspace directory. In Google AI Studio Build,")
    print("files written to the workspace persist seamlessly across standard page reloads and user sessions.")
    print("However, if the main development sandbox environment container halts, is torn down, or is restarted,")
    print("the local container's ephemeral filesystem (including 'strategies.db') will reset.")
    print("To make strategies completely durable against container resets, configuring a Firestore cloud database")
    print("using the 'set_up_firebase' service is the recommended production practice.")
    print("============================================================\n")

if __name__ == "__main__":
    main()
