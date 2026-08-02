"""Compatibility imports for the former version-named Parquet module."""

from .component_capture.parquet import ParquetChunkEpisodeWriter


V25pParquetChunkEpisodeWriter = ParquetChunkEpisodeWriter


__all__ = ["V25pParquetChunkEpisodeWriter"]
