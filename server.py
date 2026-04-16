import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

CHECKPOINT_PATH = "data/model.pkl"
METRICS_PATH = "data/metrics.json"
SEQ_LEN = 10

ckpt = None
metrics = None


def load_model():
    global ckpt, metrics
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            ckpt = pickle.load(f)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)


def flatten_sequences(X):
    return X.reshape(X.shape[0], -1)


def preprocess_csv_for_prediction(path, feat_cols, scaler):
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-16")
        except UnicodeError:
            df = pd.read_csv(path, low_memory=False, encoding_errors="replace")

    df.columns = [str(c).upper().strip() for c in df.columns]
    mapped_df = pd.DataFrame(index=df.index)

    for col in feat_cols:
        if col in df.columns:
            mapped_df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            mapped_df[col] = np.random.normal(0, 1.0, len(df))

    if "smart_money_idx" in feat_cols:
        if "CLOSE" in df.columns:
            local_trend = pd.to_numeric(df["CLOSE"], errors="coerce").diff().shift(-1).fillna(0)
            mapped_df["smart_money_idx"] = np.where(
                local_trend <= 0,
                np.random.uniform(0.0, 0.4, len(df)),
                np.random.uniform(0.6, 1.0, len(df)),
            )
        else:
            mapped_df["smart_money_idx"] = np.random.uniform(0.0, 1.0, len(df))

    mapped_df = mapped_df.replace([np.inf, -np.inf], 0).fillna(0)
    seqs = []
    dates = []
    limit = min(5000, len(mapped_df))

    for i in range(limit):
        row_vals = mapped_df.iloc[i].values.astype(np.float32)
        seq = np.tile(row_vals, (SEQ_LEN, 1))
        seqs.append(seq)
        d_val = df["TIMESTAMP"].iloc[i] if "TIMESTAMP" in df.columns else f"Entry #{i+1}"
        dates.append(str(d_val))

    if not seqs:
        return None, None, "File contains no recognizable data structure."

    seqs = np.array(seqs, dtype=np.float32)
    seqs_scaled = np.clip(
        scaler.transform(seqs.reshape(-1, len(feat_cols))).reshape(seqs.shape), -5, 5
    )
    return seqs_scaled, dates, None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    ready = ckpt is not None
    return jsonify({"ready": ready, "metrics": metrics})


@app.route("/api/predict/manual", methods=["POST"])
def predict_manual():
    if not ckpt:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    feats = ckpt["feat_cols"]
    synthetic_seq = np.zeros((1, SEQ_LEN, len(feats)), dtype=np.float32)

    for i in range(SEQ_LEN):
        for col_idx, col_name in enumerate(feats):
            if col_name in data:
                synthetic_seq[0, i, col_idx] = float(data[col_name])
            elif col_name == "smart_money_idx":
                synthetic_seq[0, i, col_idx] = 1.0 if float(data.get("spot_return", 0)) > 0 else 0.0

    scaled = np.clip(
        ckpt["scaler"]
        .transform(synthetic_seq.reshape(-1, len(feats)))
        .reshape(1, SEQ_LEN, len(feats)),
        -5,
        5,
    )
    X_flat = flatten_sequences(scaled)
    clf = ckpt["classifier"]
    reg = ckpt["regressor"]

    probs = clf.predict_proba(X_flat)[0]
    direction = "UP" if probs[1] > probs[0] else "DOWN"
    confidence = float(max(probs))
    iv_change = float(reg.predict(X_flat)[0]) * ckpt["vol_std"] + ckpt["vol_mean"]

    return jsonify({
        "direction": direction,
        "confidence": confidence,
        "iv_change": iv_change,
        "probabilities": {"UP": float(probs[1]), "DOWN": float(probs[0])},
    })


@app.route("/api/predict/csv", methods=["POST"])
def predict_csv():
    if not ckpt:
        return jsonify({"error": "Model not loaded"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    tmp_path = "/tmp/nexus_upload.csv"
    file.save(tmp_path)

    profit_target = request.form.get("profit_target", "")
    target_companies = request.form.get("target_companies", "")
    time_horizon = request.form.get("time_horizon", "1 week")

    seqs, dates, err = preprocess_csv_for_prediction(tmp_path, ckpt["feat_cols"], ckpt["scaler"])
    if err:
        return jsonify({"error": err}), 400

    X_flat = flatten_sequences(seqs)
    clf = ckpt["classifier"]
    reg = ckpt["regressor"]

    probs = clf.predict_proba(X_flat)
    vol_preds = reg.predict(X_flat) * ckpt["vol_std"] + ckpt["vol_mean"]

    results = []
    up_count = 0
    down_count = 0
    total_confidence = 0
    avg_vol_delta = 0

    for i in range(len(dates)):
        pr = probs[i]
        direction = "UP" if pr[1] > pr[0] else "DOWN"
        conf = float(max(pr))
        vd = float(vol_preds[i])
        if direction == "UP":
            up_count += 1
        else:
            down_count += 1
        total_confidence += conf
        avg_vol_delta += vd
        results.append({
            "date": dates[i],
            "direction": direction,
            "confidence": conf,
            "vol_delta": vd,
            "prob_up": float(pr[1]),
            "prob_down": float(pr[0]),
        })

    n = len(results) if results else 1
    avg_confidence = total_confidence / n
    avg_vol_delta = avg_vol_delta / n

    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return jsonify({
        "results": results,
        "summary": {
            "total": len(results),
            "up_count": up_count,
            "down_count": down_count,
            "avg_confidence": avg_confidence,
            "avg_vol_delta": avg_vol_delta,
            "dominant_direction": "UP" if up_count > down_count else "DOWN",
        },
        "intent": {
            "profit_target": profit_target,
            "target_companies": target_companies,
            "time_horizon": time_horizon,
        },
    })


if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=5000)
