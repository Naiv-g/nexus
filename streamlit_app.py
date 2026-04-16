import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json
import os
from app import preprocess_csv_for_prediction, flatten_sequences, SEQ_LEN
from llm_advisor import get_advisory, COMPANY_LIST

st.set_page_config(page_title="NIFTY Analytics", layout="wide")

@st.cache_resource
def load_model_data():
    if not os.path.exists("data/model.pkl"):
        return None, None
    with open("data/model.pkl", "rb") as f:
        ckpt = pickle.load(f)
    metrics = None
    if os.path.exists("data/metrics.json"):
        with open("data/metrics.json") as f:
            metrics = json.load(f)
    return ckpt, metrics

ckpt, metrics = load_model_data()

st.title("NIFTY / BANKNIFTY Options Analytics")

if not ckpt:
    st.error("Model not found. Train the model first by running `python train_model.py`.")
    st.stop()

tab_metrics, tab_csv, tab_advisory = st.tabs(["Evaluation Metrics", "Historical CSV Analysis", "Market Advisor"])

with tab_metrics:
    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy Score", f"{metrics.get('accuracy', 0):.2%}")
        col2.metric("F1 Macro", f"{metrics.get('f1_macro', 0):.4f}")
        col3.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        
        st.write("---")
        st.subheader("Confusion Matrix")
        cm = metrics.get("confusion_matrix", [])
        if cm:
            st.dataframe(pd.DataFrame(cm, columns=["Predicted DOWN", "Predicted UP"], index=["Actual DOWN", "Actual UP"]))
    else:
        st.warning("Metrics file missing.")

with tab_csv:
    st.subheader("Batch CSV Processing")
    uploaded_file = st.file_uploader("Upload historical Options data (CSV)", type=["csv"], key="batch_csv")
    if uploaded_file and st.button("Generate Projections"):
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        seqs, dates, err = preprocess_csv_for_prediction("temp.csv", ckpt["feat_cols"], ckpt["scaler"])
        if err:
            st.error(err)
        else:
            X_flat = flatten_sequences(seqs)
            probs = ckpt["classifier"].predict_proba(X_flat)
            vols = ckpt["regressor"].predict(X_flat) * ckpt["vol_std"] + ckpt["vol_mean"]
            
            res_df = pd.DataFrame({
                "Date": dates,
                "Calculated Direction": ["UP" if p[1] > p[0] else "DOWN" for p in probs],
                "Confidence": [float(max(p)) for p in probs],
                "Vol Delta": [float(v) for v in vols]
            })
            
            # Display comprehensive numerical breakdown
            st.write("### Prediction Metrics Summary")
            up_count = len(res_df[res_df["Calculated Direction"] == "UP"])
            down_count = len(res_df) - up_count
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total UP Signals", up_count)
            c2.metric("Total DOWN Signals", down_count)
            c3.metric("Avg Confidence", f"{res_df['Confidence'].mean():.2%}")
            c4.metric("Avg Vol Delta", f"{res_df['Vol Delta'].mean():+.4f}")
            
            st.write("---")
            st.write("### Sequential Heatmap Grid")
            st.caption("Values are heat-mapped: Blue indicates lower/negative values, Red indicates higher/positive values.")
            
            # Heat Map the Dataframe
            styled_df = res_df.style.background_gradient(cmap="coolwarm", subset=["Confidence", "Vol Delta"]) \
                                    .format({"Confidence": "{:.2%}", "Vol Delta": "{:+.4f}"})
            st.dataframe(styled_df, use_container_width=True)
            
            st.write("---")
            st.write("### Confidence Distribution")
            # Create a simple bar chart to act as a distribution breakdown
            hist_data, edges = np.histogram(res_df["Confidence"], bins=10, range=(0.5, 1.0))
            dist_df = pd.DataFrame({"Count": hist_data}, index=[f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges)-1)])
            st.bar_chart(dist_df)
            
            st.write("### Volatility Delta Fluctuations")
            st.line_chart(res_df.set_index("Date")["Vol Delta"])

with tab_advisory:
    st.subheader("Trading Plan Advisor")
    st.write("Set your targets and select a data input method to generate a practical recommendation.")
    
    st.write("### Your Plan")
    col_t1, col_t2 = st.columns(2)
    companies = col_t1.multiselect("Companies / PSUs in focus", COMPANY_LIST, default=["NTPC"])
    weekly_target = col_t2.number_input("Weekly Profit Target (%)", value=5.0)
    
    st.write("---")
    st.write("### Market Data Input")
    
    mode = st.radio("Select input format", ["Upload CSV", "Manual Entry"], horizontal=True)
    
    if mode == "Upload CSV":
        adv_file = st.file_uploader("Upload raw options dataset (CSV)", type=["csv"], key="adv_csv")
        if st.button("Run Advisor"):
            if not adv_file:
                st.error("Please provide a CSV file.")
            else:
                with open("temp_adv.csv", "wb") as f:
                    f.write(adv_file.getbuffer())
                seqs, dates, err = preprocess_csv_for_prediction("temp_adv.csv", ckpt["feat_cols"], ckpt["scaler"])
                if err:
                    st.error(err)
                else:
                    X_flat = flatten_sequences(seqs)
                    probs = ckpt["classifier"].predict_proba(X_flat)
                    vols = ckpt["regressor"].predict(X_flat) * ckpt["vol_std"] + ckpt["vol_mean"]
                    
                    up_count = sum(1 for p in probs if p[1] > p[0])
                    summary = {
                        "up": up_count,
                        "down": len(probs) - up_count,
                        "avg_confidence": float(np.mean([max(p) for p in probs])),
                        "avg_vol_delta": float(np.mean(vols))
                    }
                    intent = {"companies": ", ".join(companies) if companies else "General Options Market", "weekly_target": weekly_target}
                    
                    st.info("Generating recommendation...")
                    recommendation = get_advisory(intent, summary, metrics.get("accuracy", 0))
                    
                    st.success("Advisor Response:")
                    st.write(recommendation)
                    
    elif mode == "Manual Entry":
        st.write("Provide current market metrics for an immediate recommendation.")
        c1, c2 = st.columns(2)
        m_call_oi = c1.number_input("Call Open Interest", value=1500000.0)
        m_put_oi = c2.number_input("Put Open Interest", value=1800000.0)
        m_pcr = c1.number_input("Put/Call Ratio (PCR)", value=1.2)
        m_straddle = c2.number_input("Straddle Price", value=350.0)
        m_spot_return = c1.number_input("Spot Return (%)", value=0.5)
        
        st.write("")
        if st.button("Run Advisor"):
            vals = {
                "call_oi": m_call_oi,
                "put_oi": m_put_oi,
                "pcr": m_pcr,
                "straddle": m_straddle,
                "spot_return": m_spot_return
            }
            
            feats = ckpt["feat_cols"]
            seq = np.zeros((1, SEQ_LEN, len(feats)), dtype=np.float32)
            for i in range(SEQ_LEN):
                for ci, cn in enumerate(feats):
                    if cn in vals:
                        seq[0, i, ci] = vals[cn]
                    elif cn == "smart_money_idx":
                        seq[0, i, ci] = 1.0 if m_spot_return > 0 else 0.0
                        
            scaled = np.clip(ckpt["scaler"].transform(seq.reshape(-1, len(feats))).reshape(1, SEQ_LEN, len(feats)), -5, 5)
            X_flat = flatten_sequences(scaled)
            probs = ckpt["classifier"].predict_proba(X_flat)
            vol_pred = ckpt["regressor"].predict(X_flat) * ckpt["vol_std"] + ckpt["vol_mean"]
            
            p0 = probs[0]
            summary = {
                "up": 1 if p0[1] > p0[0] else 0,
                "down": 0 if p0[1] > p0[0] else 1,
                "avg_confidence": float(max(p0)),
                "avg_vol_delta": float(vol_pred[0])
            }
            
            st.success(f"**Calculated Signal:** Direction {'UP' if summary['up'] else 'DOWN'} | Win Probability {summary['avg_confidence']:.2%} | Expected Volatility Shift {summary['avg_vol_delta']:+.4f}")
            
            intent = {"companies": ", ".join(companies) if companies else "General Options Market", "weekly_target": weekly_target}
            
            st.info("Generating recommendation...")
            recommendation = get_advisory(intent, summary, metrics.get("accuracy", 0))
            st.write("")
            st.write(recommendation)
