"""Dataset manager for storing multi-turn conversations in SQLite and exporting to various formats."""

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz
from datasets import Dataset


DEFAULT_DUPLICATE_THRESHOLD = 85.0
DEFAULT_RETRY_ATTEMPTS = 5
DEFAULT_RETRY_DELAY = 0.1  # seconds


class DuplicateEntryError(ValueError):
    """Raised when attempting to insert a duplicate dataset entry."""

    def __init__(self, existing_entry: Dict[str, Any], similarity: float) -> None:
        self.existing_entry = existing_entry
        self.similarity = similarity
        message = (
            "Duplicate entry detected: existing entry "
            f"#{existing_entry['id']} matches with {similarity:.1f}% similarity."
        )
        super().__init__(message)


class DatasetManager:
    """Manage dataset storage in SQLite with export capabilities."""

    def __init__(self, db_path: str) -> None:
        """Initialize dataset manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Enable WAL mode for better concurrent access and set longer timeout
        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,  # 30 second timeout for locks
            check_same_thread=False  # Allow usage from multiple threads
        )
        # Enable Write-Ahead Logging for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
        self._create_table()

    def _create_table(self) -> None:
        """Create dataset table if not exists."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                messages TEXT NOT NULL,
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
        messages: List[str],
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a multi-turn conversation entry to the dataset.

        Args:
            messages: List of strings alternating between user and assistant messages.
                     Must start with user and end with assistant.
                     Example: ["user message 1", "assistant reply 1", "user message 2", "assistant reply 2"]
            metadata: Optional metadata (e.g., source location)

        Returns:
            Entry ID
        """
        # Validate messages format
        if not messages or len(messages) < 2:
            raise ValueError("messages must contain at least one user-assistant pair")
        if len(messages) % 2 != 0:
            raise ValueError("messages must have even length (alternating user-assistant pairs)")
        
        # Build messages array in OpenAI format
        formatted_messages = []
        for i, content in enumerate(messages):
            role = "user" if i % 2 == 0 else "assistant"
            formatted_messages.append({"role": role, "content": content.strip()})
        
        cursor = self.conn.cursor()
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        messages_json = json.dumps(formatted_messages, ensure_ascii=False)

        duplicate = self.find_similar_entry(formatted_messages, DEFAULT_DUPLICATE_THRESHOLD)
        if duplicate:
            existing_entry = {
                "id": duplicate["id"],
                "messages": duplicate["messages"],
            }
            raise DuplicateEntryError(existing_entry, duplicate["similarity"])

        # Retry logic for database locked errors
        for attempt in range(DEFAULT_RETRY_ATTEMPTS):
            try:
                cursor.execute(
                    "INSERT INTO dataset (messages, metadata) VALUES (?, ?)",
                    (messages_json, metadata_json)
                )
                self.conn.commit()
                return cursor.lastrowid
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < DEFAULT_RETRY_ATTEMPTS - 1:
                    time.sleep(DEFAULT_RETRY_DELAY * (attempt + 1))  # Exponential backoff
                    continue
                raise

    def find_similar_entry(
        self,
        messages: List[Dict[str, str]],
        threshold: float = DEFAULT_DUPLICATE_THRESHOLD
    ) -> Optional[Dict[str, Any]]:
        """Find an existing entry similar to the proposed messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            threshold: Similarity ratio threshold for duplicate detection

        Returns:
            Dictionary with existing entry info and similarity if found, otherwise None
        """
        # Combine all messages content for comparison
        combined_candidate = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in messages
        ).strip().lower()
        
        if not combined_candidate:
            return None

        cursor = self.conn.cursor()
        cursor.execute("SELECT id, messages FROM dataset")

        best_match: Optional[Dict[str, Any]] = None
        for entry_id, existing_messages_json in cursor.fetchall():
            try:
                existing_messages = json.loads(existing_messages_json)
                combined_existing = "\n".join(
                    f"{msg['role']}: {msg['content']}" for msg in existing_messages
                ).strip().lower()
                
                if not combined_existing:
                    continue

                similarity = fuzz.ratio(combined_candidate, combined_existing)
                if similarity >= threshold:
                    if not best_match or similarity > best_match["similarity"]:
                        best_match = {
                            "id": entry_id,
                            "messages": existing_messages,
                            "similarity": similarity,
                        }
            except (json.JSONDecodeError, KeyError):
                continue

        return best_match

    def get_entry(self, entry_id: int) -> Optional[Dict]:
        """Get a specific entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            Entry dictionary or None
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, messages, metadata, created_at FROM dataset WHERE id = ?",
            (entry_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return {
                "id": row[0],
                "messages": json.loads(row[1]),
                "metadata": json.loads(row[2]) if row[2] else None,
                "created_at": row[3]
            }
        return None

    def get_all_entries(self) -> List[Dict]:
        """Get all dataset entries.

        Returns:
            List of entry dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, messages, metadata, created_at FROM dataset ORDER BY id"
        )

        entries = []
        for row in cursor.fetchall():
            entries.append({
                "id": row[0],
                "messages": json.loads(row[1]),
                "metadata": json.loads(row[2]) if row[2] else None,
                "created_at": row[3]
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
        """Export dataset to JSONL format using HuggingFace datasets.

        Args:
            output_path: Output file path

        Returns:
            Number of entries exported
        """
        entries = self.get_all_entries()
        
        if not entries:
            return 0
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_dict({"messages": [e["messages"] for e in entries]})
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Export using native method
        dataset.to_json(str(output_file), orient="records", lines=True, force_ascii=False)
        
        return len(entries)

    def export_parquet(self, output_path: str, compression: str = 'zstd') -> int:
        """Export dataset to Parquet format using HuggingFace datasets.

        Args:
            output_path: Output file path
            compression: Compression method (zstd, snappy, gzip, brotli, none)

        Returns:
            Number of entries exported
        """
        entries = self.get_all_entries()
        
        if not entries:
            return 0
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_dict({"messages": [e["messages"] for e in entries]})
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Export using native method
        compression_value = None if compression.lower() == 'none' else compression.lower()
        dataset.to_parquet(str(output_file), compression=compression_value)
        
        return len(entries)

    def export_csv(self, output_path: str) -> int:
        """Export dataset to CSV format using HuggingFace datasets.

        Args:
            output_path: Output file path

        Returns:
            Number of entries exported
        """
        entries = self.get_all_entries()
        
        if not entries:
            return 0
        
        # Convert to HuggingFace dataset
        # Store messages as JSON string for CSV compatibility
        dataset = Dataset.from_dict({
            "messages": [json.dumps(e["messages"], ensure_ascii=False) for e in entries]
        })
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Export using native method
        dataset.to_csv(str(output_file), index=False)
        
        return len(entries)

    def export_json(self, output_path: str) -> int:
        """Export dataset to JSON format using HuggingFace datasets.

        Args:
            output_path: Output file path

        Returns:
            Number of entries exported
        """
        entries = self.get_all_entries()
        
        if not entries:
            return 0
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_dict({"messages": [e["messages"] for e in entries]})
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Export using native method (JSON array format)
        dataset.to_json(str(output_file), orient="records", lines=False, force_ascii=False, indent=2)
        
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
            submitted_count: Number of conversations submitted
            target_count: Target number of conversations
            status: Thread status (running, completed, error)
            messages: Conversation history
        """
        messages_json = json.dumps(messages, ensure_ascii=False)
        
        # Retry logic for database locked errors
        for attempt in range(DEFAULT_RETRY_ATTEMPTS):
            try:
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO thread_state 
                    (thread_id, start_position, current_position, submitted_count, target_count, status, messages, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (thread_id, start_position, current_position, submitted_count, target_count, status, messages_json))
                
                self.conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < DEFAULT_RETRY_ATTEMPTS - 1:
                    time.sleep(DEFAULT_RETRY_DELAY * (attempt + 1))
                    continue
                raise
    
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
        # Retry logic for database locked errors
        for attempt in range(DEFAULT_RETRY_ATTEMPTS):
            try:
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO session_metadata (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (key, value))
                self.conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < DEFAULT_RETRY_ATTEMPTS - 1:
                    time.sleep(DEFAULT_RETRY_DELAY * (attempt + 1))
                    continue
                raise
    
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

    # Page submission tracking helpers

    def _load_page_submission_counts(self) -> Dict[int, int]:
        """Load per-page submission counts from session metadata."""
        counts_json = self.get_session_metadata("page_submission_counts")
        counts: Dict[int, int] = {}
        if counts_json:
            try:
                raw = json.loads(counts_json)
                if isinstance(raw, dict):
                    for key, value in raw.items():
                        try:
                            counts[int(key)] = int(value)
                        except (ValueError, TypeError):
                            continue
            except json.JSONDecodeError:
                pass
        return counts

    def record_page_submission(self, page_number: int) -> Dict[int, int]:
        """Record a dataset submission for the given page.

        Args:
            page_number: Page number associated with the submission

        Returns:
            Updated dictionary mapping page numbers to submission counts
        """
        counts = self._load_page_submission_counts()
        counts[page_number] = counts.get(page_number, 0) + 1

        payload = json.dumps({str(k): v for k, v in counts.items()}, ensure_ascii=False)
        self.set_session_metadata("page_submission_counts", payload)
        self.set_session_metadata("last_submission_page", str(page_number))
        return counts

    def get_page_submission_counts(self) -> Dict[int, int]:
        """Get per-page submission counts."""
        return self._load_page_submission_counts()

    def get_page_submission_summary(self, limit: int = 5) -> Dict[str, Any]:
        """Get summary statistics for page submission counts."""
        counts = self.get_page_submission_counts()
        total_pages = len(counts)
        total_submissions = sum(counts.values())

        least_sorted = sorted(counts.items(), key=lambda item: (item[1], item[0]))
        most_sorted = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

        least_pages = [page for page, _ in least_sorted[:limit]]
        most_pages = [page for page, _ in most_sorted[:limit]]

        average = total_submissions / total_pages if total_pages else 0.0

        summary: Dict[str, Any] = {
            "total_pages": total_pages,
            "total_submissions": total_submissions,
            "average_submissions_per_page": average,
            "page_submission_counts": counts,
            "least_submitted_pages": least_pages,
            "most_submitted_pages": most_pages,
        }

        last_page = self.get_session_metadata("last_submission_page")
        if last_page:
            try:
                summary["last_submission_page"] = int(last_page)
            except ValueError:
                summary["last_submission_page"] = last_page

        return summary
    
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
