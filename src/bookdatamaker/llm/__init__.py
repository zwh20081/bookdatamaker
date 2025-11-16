"""LLM integration module."""

from .parallel_generator import ParallelDatasetGenerator, parse_distribution

__all__ = ["ParallelDatasetGenerator", "parse_distribution"]
