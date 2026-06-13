"""各 FinMind dataset 的 point-in-time loader。

通則：每個 loader (a) 呼叫 fetch_finmind_cached；(b) rename 正規化；
(c) to_datetime + to_numeric(coerce)；(d) 依 as_of 切片（傳 end_date）；
(e) 空資料回空 DataFrame（不 raise），讓因子層判中性。
as_of 是避免 look-ahead 的單一機制。
"""
from __future__ import annotations

import pandas as pd

from .cache import fetch_finmind_cached, FinMindRateLimitError


def _require_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def get_institutional(stock_id: str, start: str, as_of: str | None = None) -> pd.DataFrame:
    """三大法人買賣超（日）。回欄位:
       date, foreign_net, trust_net, dealer_net, total_net（單位：股）。
    FinMind name 欄分桶：Foreign* → 外資、Investment_Trust → 投信、
    Dealer*（self+Hedging）→ 自營；net = buy - sell。"""
    try:
        df = fetch_finmind_cached(
            "TaiwanStockInstitutionalInvestorsBuySell", stock_id, start, end_date=as_of
        )
    except FinMindRateLimitError:
        return pd.DataFrame()
    if df.empty or not _require_cols(df, ["date", "name", "buy", "sell"]):
        return pd.DataFrame()
    df = df.copy()
    df["buy"] = pd.to_numeric(df["buy"], errors="coerce")
    df["sell"] = pd.to_numeric(df["sell"], errors="coerce")
    df["net"] = df["buy"] - df["sell"]

    def bucket(name: str) -> str:
        n = str(name)
        if n.startswith("Foreign"):
            return "foreign_net"
        if n.startswith("Investment_Trust"):
            return "trust_net"
        if n.startswith("Dealer"):
            return "dealer_net"
        return "other"

    df["bucket"] = df["name"].map(bucket)
    df = df[df["bucket"] != "other"]
    wide = df.pivot_table(index="date", columns="bucket", values="net",
                          aggfunc="sum", fill_value=0).reset_index()
    for col in ["foreign_net", "trust_net", "dealer_net"]:
        if col not in wide.columns:
            wide[col] = 0
    wide["total_net"] = wide["foreign_net"] + wide["trust_net"] + wide["dealer_net"]
    return wide[["date", "foreign_net", "trust_net", "dealer_net", "total_net"]].sort_values("date").reset_index(drop=True)
