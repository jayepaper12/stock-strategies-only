import os
from datetime import datetime, timedelta

import requests
import pandas as pd

from .config import FINMIND_URL


def fetch_finmind(dataset: str, stock_id: str, start_date: str) -> pd.DataFrame:
    params = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": start_date,
        "token": os.environ["FINMIND_TOKEN"],
    }
    r = requests.get(FINMIND_URL, params=params, timeout=20)
    r.raise_for_status()
    return pd.DataFrame(r.json().get("data", []))


def get_price_history(stock_id: str, years: int = 3) -> pd.DataFrame:
    start = (datetime.now() - timedelta(days=365 * years + 60)).strftime("%Y-%m-%d")
    df = fetch_finmind("TaiwanStockPrice", stock_id, start)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_fundamental(stock_id: str) -> dict:
    """近 3 完整年度 EPS、ROE"""
    start = f"{datetime.now().year - 4}-01-01"
    df = fetch_finmind("TaiwanStockFinancialStatements", stock_id, start)
    if df.empty:
        return {"eps": {}, "roe": {}}

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    eps = df[df["type"] == "EPS"].groupby("year")["value"].sum().to_dict()
    roe = df[df["type"] == "ROE"].groupby("year")["value"].sum().to_dict()

    cy = datetime.now().year
    return {
        "eps": {y: round(v, 2) for y, v in eps.items() if cy - 3 <= y < cy},
        "roe": {y: round(v, 2) for y, v in roe.items() if cy - 3 <= y < cy},
    }
