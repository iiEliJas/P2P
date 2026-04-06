######################################
#           DataLoader
######################################

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (COL_ORDER_DATE, COL_QUANTITY, COL_SHIP_DATE, DATAFILE_ENCODING)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: list[str] = [COL_ORDER_DATE, COL_QUANTITY, COL_SHIP_DATE]



# Loader for the DataCo Supply Chain dataset
#
#   filepath : Path to the raw CSV file
#
#   Usage:
#       loader = DataLoader("data/raw/DataCo_Supply_Chain_Data.csv")
#       df = loader.load()
#       print(loader.get_summary())
#
class DataLoader:

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        self._data: pd.DataFrame | None = None
        self._duplicates_dropped: int = 0


    # Reads csv, parses dates, validates columns, and returns the raw DataFrame
    def load(self) -> pd.DataFrame:
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{self.filepath}'. "
                "DataCo CSV should be in data/raw/"
            )

        logger.info("Loading dataset from %s …", self.filepath)
        self._data = pd.read_csv(
            self.filepath,
            encoding=DATAFILE_ENCODING,
            low_memory=False,
        )
        logger.info("  -> %d rows, %d columns loaded.", *self._data.shape)

        self._parse_dates()
        self._validate()
        self._drop_duplicates()

        return self._data


    # Returns dictionary summary of the loaded dataset,
    # including shape, date range, missing values, and duplicate rows
    #
    # IMPORTANT: called after load()
    def get_summary(self) -> dict:
        data = self._require_loaded()
        return {
            "shape": data.shape,
            "date_range": (
                data[COL_ORDER_DATE].min(),
                data[COL_ORDER_DATE].max(),
            ),
            "missing_values": data.isnull().sum().to_dict(),
            "duplicate_rows_dropped": self._duplicates_dropped,
        }


    # ------------------------------------------------------------------
    # Private helpers
    # 

    # parse date columns to datetime
    def _parse_dates(self) -> None:
        data = self._require_loaded()
        for col in [COL_ORDER_DATE, COL_SHIP_DATE]:
            if col in data.columns:
                data[col] = pd.to_datetime(
                    data[col], dayfirst=False, errors="coerce"
                )
        logger.debug("Date columns parsed.")

    # check that all required columns are present
    def _validate(self) -> None:
        data = self._require_loaded()
        missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"Required columns not found in CSV: {missing}\n"
                f"Available columns: {list(data.columns)}"
            )
        logger.info("Column validation passed.")

    # remove duplicate rows and log how many were removed
    def _drop_duplicates(self) -> None:
        if self._data is None:
            raise RuntimeError("Call load() before dropping duplicates.")
        before = len(self._data)
        self._data.drop_duplicates(inplace=True, ignore_index=True)
        self._duplicates_dropped = before - len(self._data)
        if self._duplicates_dropped:
            logger.warning("Dropped %d duplicate row(s).", self._duplicates_dropped)

    # ensure load() was called
    def _require_loaded(self) -> pd.DataFrame:
        if self._data is None:
            raise RuntimeError("Call load() before accessing data.")
        return self._data
