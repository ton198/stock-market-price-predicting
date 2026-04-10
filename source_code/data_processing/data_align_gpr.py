from pathlib import Path
from datetime import datetime
import pandas as pd

_DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
_DATASETS_ALIGNED_DIR = Path(__file__).resolve().parent.parent.parent / "datasets_aligned"

def load_data() -> pd.DataFrame:
    path = _DATASETS_DIR / "data_gpr_daily_recent.CSV"
    df = pd.read_csv(path)
    df = df[["date", "GPRD_MA7"]]
    df = df.rename(columns={"GPRD_MA7": "gpr"})
    df["date"] = pd.to_datetime(df["date"])
    return df

def fill_missing_date(df: pd.DataFrame) -> pd.DataFrame:
    full_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq='D')
    for i in range(len(full_range)):
        if full_range[i] != df["date"].iloc[i]:
            df = pd.concat([df.iloc[:i], pd.DataFrame({"date": [full_range[i]]}), df.iloc[i:]],ignore_index=True)
    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df["gpr"] = df["gpr"].interpolate(method="linear")
    return df

if __name__ == "__main__":
    df = load_data()
    print("Original:")
    print(df.head())

    df = fill_missing_date(df)
    print("\nAfter date alignment:")
    print(df.head())

    df = fill_missing_values(df)
    print("\nAfter fill:")
    print(df.head())

    df.to_csv(_DATASETS_ALIGNED_DIR / "data_gpr_daily_recent.CSV", index=False)