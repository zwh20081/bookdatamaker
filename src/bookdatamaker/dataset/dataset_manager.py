"""Dataset manager for storing Q&A pairs in SQLite and exporting to various formats."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class DatasetManager:
    """Manage dataset storage in SQLite with export capabilities."""

    def __init__(self, db_path: str) -> None:
        """Initialize dataset manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_table()

    def _create_table(self) -> None:
        """Create dataset table if not exists."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create thread state table for resume support
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thread_state (
                thread_id INTEGER PRIMARY KEY,
                start_position INTEGER NOT NULL,
                current_position INTEGER NOT NULL,
                submitted_count INTEGER DEFAULT 0,
                target_count INTEGER NOT NULL,
                status TEXT DEFAULT 'running',
                messages TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create session metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()

    def add_entry(
        self, 
        prompt: str, 
        response: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a Q&A entry to the dataset.

        Args:
            prompt: Question or prompt text
            response: Answer or response text
            metadata: Optional metadata (e.g., source location)

        Returns:
            Entry ID
        """
        cursor = self.conn.cursor()
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        
        cursor.execute(
            "INSERT INTO dataset (prompt, response, metadata) VALUES (?, ?, ?)",
            (prompt, response, metadata_json)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_entry(self, entry_id: int) -> Optional[Dict]:
        """Get a specific entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            Entry dictionary or None
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, prompt, response, metadata, created_at FROM dataset WHERE id = ?",
            (entry_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return {
                "id": row[0],
                "prompt": row[1],
                "response": row[2],
                "metadata": json.loads(row[3]) if row[3] else None,
                "created_at": row[4]
            }
        return None

    def get_all_entries(self) -> List[Dict]:
        """Get all dataset entries.

        Returns:
            List of entry dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, prompt, response, metadata, created_at FROM dataset ORDER BY id"
        )
        
        entries = []
        for row in cursor.fetchall():
            entries.append({
                "id": row[0],
                "prompt": row[1],
                "response": row[2],
                "metadata": json.loads(row[3]) if row[3] else None,
                "created_at": row[4]
            })
        
        return entries

    def count_entries(self) -> int:
        """Get total number of entries.

        Returns:
            Number of entries
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dataset")
        return cursor.fetchone()[0]

    def export_jsonl(self, output_path: str) -> int:
        """Export dataset to JSONL format.

        Args:
            output_path: Output file path

        Returns:
            Number of entries exported
        """
        entries = self.get_all_entries()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with output_file.open("w", encoding="utf-8") as f:
            for entry in entries:
                # Write only prompt and response (no metadata for simplicity)
                record = {
                    "prompt": entry["prompt"],
                    "response": entry["response"]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        return len(entries)

    def export_parquet(self, output_path: str, include_metadata: bool = False) -> int:
        """Export dataset to Parquet format.

        Args:
            output_path: Output file path
            include_metadata: Whether to include metadata column

        Returns:
            Number of entries exported
        """
        entries = self.get_all_entries()
        
        if not entries:
            return 0
        
        # Prepare data for DataFrame
        data = {
            "prompt": [e["prompt"] for e in entries],
            "response": [e["response"] for e in entries]
        }
        
        if include_metadata:
            data["metadata"] = [json.dumps(e["metadata"], ensure_ascii=False) if e["metadata"] else None for e in entries]
            data["created_at"] = [e["created_at"] for e in entries]
        
        df = pd.DataFrame(data)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, index=False)
        
        return len(entries)

    def export_csv(self, output_path: str, include_metadata: bool = False) -> int:
        """Export dataset to CSV format.

        Args:
            output_path: Output file path
            include_metadata: Whether to include metadata column

        Returns:
            Number of entries exported
        """
        entries = self.get_all_entries()
        
        if not entries:
            return 0
        
        # Prepare data for DataFrame
        data = {
            "prompt": [e["prompt"] for e in entries],
            "response": [e["response"] for e in entries]
        }
        
        if include_metadata:
            data["metadata"] = [json.dumps(e["metadata"], ensure_ascii=False) if e["metadata"] else None for e in entries]
            data["created_at"] = [e["created_at"] for e in entries]
        
        df = pd.DataFrame(data)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False, encoding="utf-8")
        
        return len(entries)

    def export_json(self, output_path: str, include_metadata: bool = False) -> int:
        """Export dataset to JSON format.

        Args:
            output_path: Output file path
            include_metadata: Whether to include metadata

        Returns:
            Number of entries exported
        """
        entries = self.get_all_entries()
        
        if not include_metadata:
            # Simplify entries to only prompt/response
            entries = [
                {"prompt": e["prompt"], "response": e["response"]}
                for e in entries
            ]
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        
        return len(entries)

    def clear(self) -> int:
        """Clear all entries from the dataset.

        Returns:
            Number of entries deleted
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dataset")
        count = cursor.fetchone()[0]
        
        cursor.execute("DELETE FROM dataset")
        self.conn.commit()
        
        return count

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    # Thread state management methods for resume support
    
    def save_thread_state(
        self,
        thread_id: int,
        start_position: int,
        current_position: int,
        submitted_count: int,
        target_count: int,
        status: str,
        messages: List[Dict]
    ) -> None:
        """Save or update thread state for resume support.
        
        Args:
            thread_id: Thread identifier
            start_position: Starting page/position
            current_position: Current page/position
            submitted_count: Number of Q&A pairs submitted
            target_count: Target number of Q&A pairs
            status: Thread status (running, completed, error)
            messages: Conversation history
        """
        cursor = self.conn.cursor()
        messages_json = json.dumps(messages, ensure_ascii=False)
        
        cursor.execute("""
            INSERT OR REPLACE INTO thread_state 
            (thread_id, start_position, current_position, submitted_count, target_count, status, messages, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (thread_id, start_position, current_position, submitted_count, target_count, status, messages_json))
        
        self.conn.commit()
    
    def get_thread_state(self, thread_id: int) -> Optional[Dict]:
        """Get thread state by ID.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Thread state dictionary or None
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT thread_id, start_position, current_position, submitted_count, 
                   target_count, status, messages, last_updated
            FROM thread_state WHERE thread_id = ?
        """, (thread_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                "thread_id": row[0],
                "start_position": row[1],
                "current_position": row[2],
                "submitted_count": row[3],
                "target_count": row[4],
                "status": row[5],
                "messages": json.loads(row[6]) if row[6] else [],
                "last_updated": row[7]
            }
        return None
    
    def get_all_thread_states(self) -> List[Dict]:
        """Get all thread states.
        
        Returns:
            List of thread state dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT thread_id, start_position, current_position, submitted_count, 
                   target_count, status, messages, last_updated
            FROM thread_state ORDER BY thread_id
        """)
        
        states = []
        for row in cursor.fetchall():
            states.append({
                "thread_id": row[0],
                "start_position": row[1],
                "current_position": row[2],
                "submitted_count": row[3],
                "target_count": row[4],
                "status": row[5],
                "messages": json.loads(row[6]) if row[6] else [],
                "last_updated": row[7]
            })
        return states
    
    def clear_thread_states(self) -> None:
        """Clear all thread states."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM thread_state")
        self.conn.commit()
    
    def set_session_metadata(self, key: str, value: str) -> None:
        """Set session metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO session_metadata (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, value))
        self.conn.commit()
    
    def get_session_metadata(self, key: str) -> Optional[str]:
        """Get session metadata.
        
        Args:
            key: Metadata key
            
        Returns:
            Metadata value or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM session_metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    def get_incomplete_threads(self) -> List[Dict]:
        """Get all incomplete thread states (status != 'completed').
        
        Returns:
            List of incomplete thread state dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT thread_id, start_position, current_position, submitted_count, 
                   target_count, status, messages, last_updated
            FROM thread_state 
            WHERE status != 'completed'
            ORDER BY thread_id
        """)
        
        states = []
        for row in cursor.fetchall():
            states.append({
                "thread_id": row[0],
                "start_position": row[1],
                "current_position": row[2],
                "submitted_count": row[3],
                "target_count": row[4],
                "status": row[5],
                "messages": json.loads(row[6]) if row[6] else [],
                "last_updated": row[7]
            })
        return states
