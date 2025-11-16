"""Dataset generation and export to Parquet format."""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class DatasetBuilder:
    """Build and export datasets in Parquet format."""

    def __init__(self) -> None:
        """Initialize dataset builder."""
        self.data: List[Dict[str, Any]] = []

    def add_entry(self, entry: Dict[str, Any]) -> None:
        """Add a single entry to the dataset.

        Args:
            entry: Dictionary containing dataset fields
        """
        self.data.append(entry)

    def add_entries(self, entries: List[Dict[str, Any]]) -> None:
        """Add multiple entries to the dataset.

        Args:
            entries: List of dictionaries containing dataset fields
        """
        self.data.extend(entries)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame.

        Returns:
            DataFrame containing the dataset
        """
        return pd.DataFrame(self.data)

    def save_parquet(
        self,
        output_path: Path,
        compression: str = "zstd",
        schema: pa.Schema = None,
    ) -> None:
        """Save dataset to Parquet format with zstd compression.

        Args:
            output_path: Output file path
            compression: Compression algorithm (default: zstd, also supports snappy, gzip, brotli, etc.)
            schema: Optional PyArrow schema
        """
        df = self.to_dataframe()

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to PyArrow Table
        if schema:
            table = pa.Table.from_pandas(df, schema=schema)
        else:
            table = pa.Table.from_pandas(df)

        # Write to Parquet
        pq.write_table(table, output_path, compression=compression)

    def load_parquet(self, input_path: Path) -> None:
        """Load dataset from Parquet file.

        Args:
            input_path: Input file path
        """
        df = pd.read_parquet(input_path)
        self.data = df.to_dict("records")

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dictionary containing dataset statistics
        """
        df = self.to_dataframe()

        stats = {
            "total_entries": len(df),
            "columns": list(df.columns),
        }

        # Add column-specific stats
        for col in df.columns:
            if df[col].dtype == "object":
                stats[f"{col}_unique_count"] = df[col].nunique()
            elif pd.api.types.is_numeric_dtype(df[col]):
                stats[f"{col}_mean"] = df[col].mean()
                stats[f"{col}_sum"] = df[col].sum()

        return stats

    def clear(self) -> None:
        """Clear all data from the dataset."""
        self.data = []


def create_standard_schema() -> pa.Schema:
    """Create a standard schema for Q&A datasets.

    Returns:
        PyArrow schema
    """
    return pa.schema(
        [
            pa.field("input", pa.string()),
            pa.field("output", pa.string()),
            pa.field("model", pa.string()),
            pa.field("tokens_used", pa.int64()),
        ]
    )
