import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
)
from preprocess import get_preprocessed_data

CHECKPOINT_PATH = "data/model.pkl"
METRICS_PATH = "data/metrics.json"

def flatten_sequences(X):
    return X.reshape(X.shape[0], -1)

def train():
    data = get_preprocessed_data()
        
    X_tr, y_dir_tr, y_vol_tr = data["X_tr"], data["y_dir_tr"], data["y_vol_tr"]
    X_val, y_dir_val, y_vol_val = data["X_val"], data["y_dir_val"], data["y_vol_val"]
    X_te, y_dir_te, y_vol_te = data["X_te"], data["y_dir_te"], data["y_vol_te"]
    
    X_train = np.vstack((X_tr, X_val))
    y_dir_train = np.concatenate((y_dir_tr, y_dir_val))
    y_vol_train = np.concatenate((y_vol_tr, y_vol_val))
    
    X_train_flat = flatten_sequences(X_train)
    X_test_flat = flatten_sequences(X_te)
    
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=0.1,
        max_leaf_nodes=31,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train_flat, y_dir_train)
    
    reg = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=0.1,
        random_state=42
    )
    reg.fit(X_train_flat, y_vol_train)
    
    dir_preds = clf.predict(X_test_flat)
    dir_probs = clf.predict_proba(X_test_flat)
    vol_preds = reg.predict(X_test_flat)
    
    acc = accuracy_score(y_dir_te, dir_preds)
    f1m = f1_score(y_dir_te, dir_preds, average="macro", zero_division=0)
    f1w = f1_score(y_dir_te, dir_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(y_dir_te, dir_preds).tolist()
    
    try:
        auc = float(roc_auc_score(y_dir_te, dir_probs[:, 1]))
    except Exception:
        auc = float("nan")
        
    report = classification_report(y_dir_te, dir_preds, target_names=["DOWN", "UP"], zero_division=0)
    
    mae = float(mean_absolute_error(y_vol_te, vol_preds))
    rmse = float(np.sqrt(mean_squared_error(y_vol_te, vol_preds)))
    r2 = float(r2_score(y_vol_te, vol_preds))
    
    metrics = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1m, 4),
        "f1_weighted": round(f1w, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "vol_mae": round(mae, 6),
        "vol_rmse": round(rmse, 6),
        "vol_r2": round(r2, 4),
    }

    model_data = {
        "classifier": clf,
        "regressor": reg,
        "feat_cols": data["feat_cols"],
        "scaler": data["scaler"],
        "vol_mean": data["vol_mean"],
        "vol_std": data["vol_std"],
    }
    os.makedirs("data", exist_ok=True)
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(model_data, f)
        
    save_metrics = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(METRICS_PATH, "w") as f:
        json.dump(save_metrics, f, indent=2)
        
    return model_data, metrics

if __name__ == "__main__":
    train()
