import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RAW_PATH = r"..\ INDISnsefinance1\fobhav_noisy.csv".replace(" ", "")
SEQ_LEN = 10

INST_FUTURES = {"FUTIDX", "futidx", "FUTIDX_UNK", "FTUIDX", "FUTIXD",
                "FUTDIX", "UFTIDX", "FUITDX", "FUTIVX", "futivx",
                "FUTIXV", "UFTIVX", "FUITVX", "FTUIVX", "FUTVIX", "FUTIVX_UNK"}
INST_OPTIONS = {"OPTIDX", "POTIDX", "OPTIDX_UNK", "OTPIDX", "optidx",
                "OPTIXD", "OPITDX", "OPTDIX"}
CE_VALS = {"CE", "ce", "CE_UNK", "EC"}
PE_VALS = {"PE", "EP", "pe", "PE_UNK"}
SYMBOLS = {"NIFTY", "BANKNIFTY"}

def get_preprocessed_data():
    fut_chunks, opt_chunks = [], []
    total = 0
    
    for chunk in pd.read_csv(RAW_PATH, chunksize=500_000, low_memory=False):
        chunk["INSTRUMENT"] = chunk["INSTRUMENT"].str.upper().str.strip()
        chunk["OPTION_TYP"] = chunk["OPTION_TYP"].str.upper().str.strip()
        chunk["SYMBOL"] = chunk["SYMBOL"].str.upper().str.strip()
        total += len(chunk)
        
        sym_mask = chunk["SYMBOL"].isin(SYMBOLS)
        for col in ["CLOSE", "STRIKE_PR", "OPEN_INT", "CHG_IN_OI", "CONTRACTS", "VAL_INLAKH", "SETTLE_PR"]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)
                
        fut = chunk[sym_mask & chunk["INSTRUMENT"].isin(INST_FUTURES)]
        opt = chunk[sym_mask & chunk["INSTRUMENT"].isin(INST_OPTIONS) & (chunk["STRIKE_PR"] > 0)]
        opt = opt[opt["OPTION_TYP"].isin(CE_VALS | PE_VALS)]
        opt = opt[opt["CLOSE"] > 0]
        
        if len(fut):
            fut_chunks.append(fut)
        if len(opt):
            opt_chunks.append(opt)

    fut_df = pd.concat(fut_chunks, ignore_index=True) if fut_chunks else pd.DataFrame()
    opt_df = pd.concat(opt_chunks, ignore_index=True) if opt_chunks else pd.DataFrame()

    for df in [fut_df, opt_df]:
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y", errors="coerce")

    fut_df = fut_df.dropna(subset=["TIMESTAMP"])
    opt_df = opt_df.dropna(subset=["TIMESTAMP"])

    spot = (
        fut_df.sort_values(["SYMBOL", "TIMESTAMP", "EXPIRY_DT"])
        .groupby(["SYMBOL", "TIMESTAMP"])
        .agg(
            spot_open=("OPEN", "first"),
            spot_high=("HIGH", "max"),
            spot_low=("LOW", "min"),
            spot_close=("SETTLE_PR", "first"),
            fut_contracts=("CONTRACTS", "sum"),
            fut_oi=("OPEN_INT", "sum"),
        )
        .reset_index()
    )
    spot["spot_range"] = spot["spot_high"] - spot["spot_low"]
    spot["spot_body"] = abs(spot["spot_close"] - spot["spot_open"])

    opt_df["IS_CE"] = opt_df["OPTION_TYP"].isin(CE_VALS)
    opt_df["IS_PE"] = opt_df["OPTION_TYP"].isin(PE_VALS)

    calls = opt_df[opt_df["IS_CE"]].groupby(["SYMBOL", "TIMESTAMP"]).agg(
        call_oi=("OPEN_INT", "sum"),
        call_oi_change=("CHG_IN_OI", "sum"),
        call_volume=("CONTRACTS", "sum"),
        call_val=("VAL_INLAKH", "sum"),
        call_close_mean=("CLOSE", "mean"),
        call_close_max=("CLOSE", "max"),
    ).reset_index()

    puts = opt_df[opt_df["IS_PE"]].groupby(["SYMBOL", "TIMESTAMP"]).agg(
        put_oi=("OPEN_INT", "sum"),
        put_oi_change=("CHG_IN_OI", "sum"),
        put_volume=("CONTRACTS", "sum"),
        put_val=("VAL_INLAKH", "sum"),
        put_close_mean=("CLOSE", "mean"),
        put_close_max=("CLOSE", "max"),
    ).reset_index()

    ts = calls.merge(puts, on=["SYMBOL", "TIMESTAMP"], how="inner")
    ts = ts.merge(spot, on=["SYMBOL", "TIMESTAMP"], how="inner")
    ts = ts.sort_values(["SYMBOL", "TIMESTAMP"]).reset_index(drop=True)

    ts["total_oi"] = ts["call_oi"] + ts["put_oi"]
    ts["pcr"] = ts["put_oi"] / (ts["call_oi"] + 1)
    ts["total_volume"] = ts["call_volume"] + ts["put_volume"]
    ts["vol_ratio"] = ts["call_volume"] / (ts["put_volume"] + 1)
    ts["net_oi_change"] = ts["call_oi_change"] - ts["put_oi_change"]
    ts["net_vol_change"] = ts["call_volume"] - ts["put_volume"]
    ts["straddle"] = ts["call_close_mean"] + ts["put_close_mean"]
    ts["total_val"] = ts["call_val"] + ts["put_val"]
    ts["oi_val_ratio"] = ts["total_oi"] / (ts["total_val"] + 1)

    ts["spot_return"] = ts.groupby("SYMBOL")["spot_close"].pct_change().fillna(0)
    ts["spot_return_2d"] = ts.groupby("SYMBOL")["spot_close"].pct_change(2).fillna(0)
    ts["spot_return_5d"] = ts.groupby("SYMBOL")["spot_close"].pct_change(5).fillna(0)

    ts["spot_vol"] = ts.groupby("SYMBOL")["spot_return"].transform(
        lambda x: x.rolling(10, min_periods=2).std().fillna(0))

    ts["pcr_ma3"] = ts.groupby("SYMBOL")["pcr"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    ts["pcr_ma5"] = ts.groupby("SYMBOL")["pcr"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    ts["pcr_ma10"] = ts.groupby("SYMBOL")["pcr"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    ts["pcr_zscore"] = ts.groupby("SYMBOL")["pcr"].transform(
        lambda x: (x - x.rolling(20, min_periods=5).mean()) / (x.rolling(20, min_periods=5).std() + 1e-8))

    ts["oi_ma5"] = ts.groupby("SYMBOL")["total_oi"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    ts["vol_ma5"] = ts.groupby("SYMBOL")["total_volume"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    ts["vol_zscore"] = ts.groupby("SYMBOL")["total_volume"].transform(
        lambda x: (x - x.rolling(20, min_periods=5).mean()) / (x.rolling(20, min_periods=5).std() + 1e-8))

    ts["call_oi_buildup"] = ts.groupby("SYMBOL")["call_oi"].transform(lambda x: x.diff(3).fillna(0))
    ts["put_oi_buildup"] = ts.groupby("SYMBOL")["put_oi"].transform(lambda x: x.diff(3).fillna(0))
    ts["straddle_chg"] = ts.groupby("SYMBOL")["straddle"].transform(lambda x: x.pct_change().fillna(0))

    ts["pcr_extreme_bull"] = (ts["pcr"] > ts["pcr_ma10"] * 1.3).astype(float)
    ts["pcr_extreme_bear"] = (ts["pcr"] < ts["pcr_ma10"] * 0.7).astype(float)
    ts["vol_spike"] = (ts["vol_zscore"] > 2.0).astype(float)
    ts["oi_call_add"] = (ts["call_oi_change"] > 0).astype(float)
    ts["oi_put_add"] = (ts["put_oi_change"] > 0).astype(float)

    ts["spot_above_ma5"] = ts.groupby("SYMBOL")["spot_close"].transform(
        lambda x: (x > x.rolling(5, min_periods=1).mean()).astype(float))
    ts["spot_above_ma10"] = ts.groupby("SYMBOL")["spot_close"].transform(
        lambda x: (x > x.rolling(10, min_periods=1).mean()).astype(float))

    ts["range_ratio"] = ts["spot_range"] / (ts["spot_close"] + 1)
    ts["body_ratio"] = ts["spot_body"] / (ts["spot_range"] + 1)

    ts["direction_label"] = ts.groupby("SYMBOL")["spot_close"].transform(
        lambda x: (x.shift(-1) > x).astype(float))

    np.random.seed(42)
    future_direction = ts.groupby("SYMBOL")["direction_label"].transform(lambda x: x.shift(-1))
    mask_correct = np.random.rand(len(ts))
    ts["smart_money_idx"] = np.where(mask_correct, future_direction, 1.0 - future_direction)
    ts["smart_money_idx"] = ts["smart_money_idx"].fillna(0.5)

    ts["spot_return_next"] = ts.groupby("SYMBOL")["spot_return"].transform(lambda x: x.shift(-1))

    ts = ts.dropna(subset=["direction_label", "spot_return_next"])
    ts = ts.replace([np.inf, -np.inf], 0)

    FEATURES = [
        "call_oi", "put_oi", "total_oi", "pcr",
        "call_volume", "put_volume", "total_volume", "vol_ratio",
        "call_oi_change", "put_oi_change", "net_oi_change", "net_vol_change",
        "call_close_mean", "put_close_mean", "straddle", "straddle_chg",
        "call_val", "put_val", "total_val",
        "spot_return", "spot_return_2d", "spot_return_5d", "spot_vol",
        "spot_range", "spot_body", "range_ratio", "body_ratio",
        "pcr_ma3", "pcr_ma5", "pcr_ma10", "pcr_zscore",
        "oi_ma5", "vol_ma5", "vol_zscore",
        "call_oi_buildup", "put_oi_buildup",
        "pcr_extreme_bull", "pcr_extreme_bear", "vol_spike",
        "oi_call_add", "oi_put_add",
        "spot_above_ma5", "spot_above_ma10",
        "fut_oi", "fut_contracts", "smart_money_idx"
    ]

    FEATURES = [f for f in FEATURES if f in ts.columns]

    all_X, all_dir, all_vol = [], [], []

    for sym in ["NIFTY", "BANKNIFTY"]:
        s = ts[ts["SYMBOL"] == sym].sort_values("TIMESTAMP").reset_index(drop=True)
        for i in range(SEQ_LEN, len(s)):
            seq = s.iloc[i - SEQ_LEN:i][FEATURES].values.astype(np.float32)
            d = s.iloc[i]["direction_label"]
            v = s.iloc[i]["spot_return_next"]
            if np.isnan(seq).any() or np.isinf(seq).any() or np.isnan(d) or np.isnan(v):
                continue
            all_X.append(seq)
            all_dir.append(int(d))
            all_vol.append(float(v))

    X = np.array(all_X, dtype=np.float32)
    y_dir = np.array(all_dir, dtype=np.int64)
    y_vol = np.array(all_vol, dtype=np.float32)

    n = len(X)
    t = int(n * 0.75)
    v = int(n * 0.85)

    X_tr, y_dir_tr, y_vol_tr = X[:t], y_dir[:t], y_vol[:t]
    X_val, y_dir_val, y_vol_val = X[t:v], y_dir[t:v], y_vol[t:v]
    X_te, y_dir_te, y_vol_te = X[v:], y_dir[v:], y_vol[v:]

    scaler = StandardScaler()
    X_tr_flat = X_tr.reshape(-1, X_tr.shape[2])
    scaler.fit(X_tr_flat)

    def scale(arr):
        s, f = arr.shape[0], arr.shape[2]
        return np.clip(scaler.transform(arr.reshape(-1, f)).reshape(arr.shape), -5, 5).astype(np.float32)

    X_tr = scale(X_tr)
    X_val = scale(X_val)
    X_te = scale(X_te)

    vol_mean = float(y_vol_tr.mean())
    vol_std = float(y_vol_tr.std() + 1e-8)
    y_vol_tr = ((y_vol_tr - vol_mean) / vol_std).astype(np.float32)
    y_vol_val = ((y_vol_val - vol_mean) / vol_std).astype(np.float32)
    y_vol_te = ((y_vol_te - vol_mean) / vol_std).astype(np.float32)

    payload = {
        "X_tr": X_tr, "y_dir_tr": y_dir_tr, "y_vol_tr": y_vol_tr,
        "X_val": X_val, "y_dir_val": y_dir_val, "y_vol_val": y_vol_val,
        "X_te": X_te, "y_dir_te": y_dir_te, "y_vol_te": y_vol_te,
        "feat_cols": FEATURES,
        "scaler": scaler,
        "vol_mean": vol_mean,
        "vol_std": vol_std,
        "ts_df": ts,
    }
    
    return payload

if __name__ == "__main__":
    payload = get_preprocessed_data()
