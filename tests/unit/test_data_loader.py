#
# Unit tests for src/utils/data_loader.py
#

from __future__ import annotations

import pandas as pd
import pytest

from src.utils.data_loader import DataLoader


class TestDataLoaderErrors:
    def test_missing_file_raises(self):
        loader = DataLoader("does_not_exist.csv")
        with pytest.raises(FileNotFoundError, match="does_not_exist.csv"):
            loader.load()

    def test_missing_columns_raises(self, tmp_path):
        # valid csv but missing required columns
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col_a,col_b\n1,2\n3,4\n")
        loader = DataLoader(bad_csv)
        with pytest.raises(ValueError, match="Required column"):
            loader.load()

    def test_get_summary_before_load_raises(self, tmp_path):
        loader = DataLoader(tmp_path / "x.csv")
        with pytest.raises(RuntimeError):
            loader.get_summary()



class TestDataLoaderSuccess:
    def test_load_returns_dataframe(self, sample_csv):
        loader = DataLoader(sample_csv)
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_parses_dates(self, sample_csv):
        loader = DataLoader(sample_csv)
        df = loader.load()
        from src.config import COL_ORDER_DATE
        assert pd.api.types.is_datetime64_any_dtype(df[COL_ORDER_DATE])

    def test_required_columns_present(self, sample_csv):
        loader = DataLoader(sample_csv)
        df = loader.load()
        from src.utils.data_loader import REQUIRED_COLUMNS
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"'{col}' missing after load"

    def test_summary_keys(self, sample_csv):
        loader = DataLoader(sample_csv)
        loader.load()
        summary = loader.get_summary()
        for key in ("shape", "date_range", "missing_values", "duplicate_rows_dropped"):
            assert key in summary

    def test_summary_shape_matches(self, sample_csv):
        loader = DataLoader(sample_csv)
        df = loader.load()
        assert loader.get_summary()["shape"] == df.shape



class TestDataLoaderDuplicates:
    def test_duplicates_are_dropped(self, tmp_path, raw_dataframe):
        base_path = tmp_path / "base.csv"
        raw_dataframe.to_csv(base_path, index=False)
        base_loader = DataLoader(base_path)
        base_loader.load()
        base_dropped = base_loader._duplicates_dropped
 
        # adding 10 duplicate rows
        with_dups = pd.concat(
            [raw_dataframe, raw_dataframe.head(10)], ignore_index=True
        )
        dup_path = tmp_path / "dup.csv"
        with_dups.to_csv(dup_path, index=False)
        dup_loader = DataLoader(dup_path)
        dup_loader.load()

        assert dup_loader._duplicates_dropped >= base_dropped + 10

    def test_duplicate_count_in_summary(self, tmp_path, raw_dataframe):
        path = tmp_path / "clean.csv"
        raw_dataframe.to_csv(path, index=False)
        loader = DataLoader(path)
        loader.load()
        assert loader.get_summary()["duplicate_rows_dropped"] >= 0
