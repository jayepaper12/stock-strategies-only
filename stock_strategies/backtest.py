import numpy as np
import pandas as pd

from .config import CONFIG
from .indicators import tech_score_at


def backtest(df: pd.DataFrame) -> dict:
    """對過去 3 年所有技術分 ≥60 的日子做持有 20 日結算"""
    indices = []
    for i in range(60, len(df) - CONFIG["hold_days"]):
        if tech_score_at(df.iloc[i])["score"] >= CONFIG["min_tech_score_for_signal"]:
            indices.append(i)

    if not indices:
        return {"winrate": None, "samples": 0, "avg_return": None}

    wins = losses = 0
    returns = []
    for idx in indices:
        entry = df.iloc[idx]["close"]
        future = df.iloc[idx + 1 : idx + 1 + CONFIG["hold_days"]]
        if len(future) < CONFIG["hold_days"]:
            continue

        hi, lo = future["high"].max(), future["low"].min()
        fc = future.iloc[-1]["close"]

        hit_target = hi >= entry * (1 + CONFIG["target_return"])
        hit_stop = lo <= entry * (1 - CONFIG["stop_loss"])

        if hit_target and not hit_stop:
            wins += 1
            returns.append(CONFIG["target_return"])
        elif hit_stop:
            losses += 1
            returns.append(-CONFIG["stop_loss"])
        else:
            r = (fc - entry) / entry
            returns.append(r)
            if r > 0:
                wins += 1
            else:
                losses += 1

    total = wins + losses
    if total == 0:
        return {"winrate": None, "samples": 0}

    return {
        "winrate": round(wins / total, 3),
        "samples": total,
        "avg_return": round(float(np.mean(returns)), 4),
    }
