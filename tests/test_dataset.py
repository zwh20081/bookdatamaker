"""Tests for dataset builder."""

import pytest
from pathlib import Path
import tempfile
from bookdatamaker.dataset import (
    DatasetBuilder,
    DatasetManager,
    DuplicateEntryError,
    create_standard_schema,
)


class TestDatasetBuilder:
    """Test dataset building functionality."""

    def setup_method(self):
        """Set up dataset builder."""
        self.builder = DatasetBuilder()

    def test_add_entry(self):
        """Test adding a single entry."""
        entry = {
            "prompt": "Test input",
            "completion": "Test output",
            "model": "gpt-4",
            "tokens_used": 100,
        }
        
        self.builder.add_entry(entry)
        
        assert len(self.builder.data) == 1
        assert self.builder.data[0] == entry

    def test_add_entries(self):
        """Test adding multiple entries."""
        entries = [
            {"prompt": "Input 1", "completion": "Output 1", "model": "gpt-4", "tokens_used": 50},
            {"prompt": "Input 2", "completion": "Output 2", "model": "gpt-4", "tokens_used": 75},
        ]
        
        self.builder.add_entries(entries)
        
        assert len(self.builder.data) == 2

    def test_to_dataframe(self):
        """Test converting to DataFrame."""
        entries = [
            {"prompt": "Input 1", "completion": "Output 1", "model": "gpt-4", "tokens_used": 50},
            {"prompt": "Input 2", "completion": "Output 2", "model": "gpt-4", "tokens_used": 75},
        ]
        self.builder.add_entries(entries)
        
        df = self.builder.to_dataframe()
        
        assert len(df) == 2
        assert list(df.columns) == ["prompt", "completion", "model", "tokens_used"]

    def test_save_parquet(self):
        """Test saving to Parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.parquet"
            
            entries = [
                {"prompt": "Input 1", "completion": "Output 1", "model": "gpt-4", "tokens_used": 50},
            ]
            self.builder.add_entries(entries)
            
            self.builder.save_parquet(output_path)
            
            assert output_path.exists()

    def test_load_parquet(self):
        """Test loading from Parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.parquet"
            
            # Save data
            entries = [
                {"prompt": "Input 1", "completion": "Output 1", "model": "gpt-4", "tokens_used": 50},
            ]
            self.builder.add_entries(entries)
            self.builder.save_parquet(output_path)
            
            # Load data
            new_builder = DatasetBuilder()
            new_builder.load_parquet(output_path)
            
            assert len(new_builder.data) == 1
            assert new_builder.data[0]["prompt"] == "Input 1"

    def test_get_stats(self):
        """Test getting dataset statistics."""
        entries = [
            {"prompt": "Input 1", "completion": "Output 1", "model": "gpt-4", "tokens_used": 50},
            {"prompt": "Input 2", "completion": "Output 2", "model": "gpt-4", "tokens_used": 75},
        ]
        self.builder.add_entries(entries)
        
        stats = self.builder.get_stats()
        
        assert stats["total_entries"] == 2
        assert "tokens_used_sum" in stats
        assert stats["tokens_used_sum"] == 125

    def test_clear(self):
        """Test clearing data."""
        entry = {"prompt": "Test", "completion": "Test", "model": "gpt-4", "tokens_used": 50}
        self.builder.add_entry(entry)
        
        self.builder.clear()
        
        assert len(self.builder.data) == 0

    def test_create_standard_schema(self):
        """Test creating standard schema."""
        schema = create_standard_schema()
        
        assert schema is not None
        field_names = [field.name for field in schema]
        assert "input" in field_names
        assert "output" in field_names
        assert "model" in field_names
        assert "tokens_used" in field_names




def test_dataset_manager_rejects_duplicates(tmp_path):
    """Ensure duplicate submissions are blocked when similarity exceeds threshold."""
    db_path = tmp_path / "dataset.db"
    with DatasetManager(str(db_path)) as manager:
        manager.add_entry("What is AI?", "Artificial intelligence is the study of intelligent agents.")

        with pytest.raises(DuplicateEntryError) as excinfo:
            manager.add_entry("What is AI?", "Artificial intelligence is the study of intelligent agents.")

        message = str(excinfo.value)
        assert "Duplicate entry detected" in message
        assert excinfo.value.existing_entry["id"] == 1

        manager.add_entry("Describe machine learning", "Machine learning uses data to train predictive models.")
        assert manager.count_entries() == 2
