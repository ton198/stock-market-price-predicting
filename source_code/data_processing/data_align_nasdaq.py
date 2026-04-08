from pathlib import Path
from datetime import datetime
import pandas as pd

_DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
_DATASETS_ALIGNED_DIR = Path(__file__).resolve().parent.parent.parent / "datasets_aligned"


def load_nasdaq_data():
    path = _DATASETS_DIR / "NASDAQCOM.csv"
    original_data = pd.read_csv(path)
    original_data = original_data.rename(columns={"observation_date": "date"})
    return original_data


def get_date_range(start_date: datetime, end_date: datetime) -> list[datetime]:
    return [date for date in pd.date_range(start_date, end_date)]


def fill_missing_date(df: pd.DataFrame) -> pd.DataFrame:
    full_range = list[datetime](pd.date_range(start=df["date"].min(), end=df["date"].max(), freq='D'))
    for i in range(len(full_range)):
        current_date_str = datetime.strftime(full_range[i], "%Y-%m-%d")
        if current_date_str != df["date"].iloc[i]:
            df = pd.concat([df.iloc[:i], 
                            pd.DataFrame({"date": [current_date_str]}), 
                            df.iloc[i:]], ignore_index=True)
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df['NASDAQCOM'] = df['NASDAQCOM'].interpolate(method='linear')
    return df

if __name__ == "__main__":
    df = load_nasdaq_data()
    print(df.head())
    df = fill_missing_date(df)
    print(df.head())
    df = fill_missing_values(df)
    print(df.head())
    df.to_csv(_DATASETS_ALIGNED_DIR / "NASDAQCOM.csv", index=False)