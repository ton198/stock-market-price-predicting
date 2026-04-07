from pathlib import Path
from datetime import datetime
import pandas as pd

_DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"



def load_nasdaq_data():
    path = _DATASETS_DIR / "NASDAQ.csv"
    original_data = pd.read_csv(path)
    original_data = original_data.rename(columns={"observation_date": "date"})
    return original_data


def get_date_range(start_date: datetime, end_date: datetime) -> list[datetime]:
    return [date for date in pd.date_range(start_date, end_date)]


def fill_missing_date(data: pd.DataFrame) -> pd.DataFrame:
    date_range = get_date_range(data["date"].min(), data["date"].max())


def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    for row in data.itertuples():
        

if __name__ == "__main__":
