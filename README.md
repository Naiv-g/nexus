# NEXUS — Nifty Options Intelligence Engine + AI Advisory

An AI-powered options analytics dashboard that predicts market direction (UP/DOWN) and volatility shifts using HistGradientBoosting ML, then generates **personalized investment advisory** using Google Gemini LLM based on your specific financial goals.

## Innovation

**Intent-Based AI Advisory**: Users don't just get raw predictions — they define their investment intent (weekly profit target, target companies/PSUs, time horizon), and an LLM synthesizes the model's output into **actionable, personalized advice**:
- What to focus on and how to allocate
- Whether the profit target is achievable given current signals
- Risk assessment with concrete mitigation strategies
- Projected gain/loss scenarios with rationale

## How It Works

1. **Upload CSV** — raw NSE Nifty/BankNifty options data 
2. **Set Your Intent** — weekly profit target (₹), target PSUs/companies, time horizon
3. **ML Model Predicts** — HistGradientBoosting classifies UP/DOWN direction with ~95.6% accuracy, estimates volatility shifts
4. **LLM Advises** — Gemini 2.0 Flash analyzes predictions + your intent → structured advisory with focus areas, risk assessment, and recommended actions

## Architecture

- **Backend**: Flask server with pickle-serialized HistGradientBoosting models (classifier + regressor)
- **Preprocessing**: 46 engineered financial features from raw NSE data (PCR, OI buildup, straddle pricing, smart money index)
- **Frontend**: Vanilla HTML/CSS/JS with Chart.js, glassmorphism dark theme
- **LLM**: Client-side Gemini API calls (no server-side API keys needed)

## Quick Start

```bash
pip install flask numpy pandas scikit-learn
cd nexus
python server.py
```
Open `http://localhost:5000` in your browser.

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 95.59% |
| F1 Macro | 0.9557 |
| ROC-AUC | 0.9520 |
| Vol MAE | 0.5418 |

## API Key

Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Enter it in the dashboard — it's stored in your browser's localStorage only, never sent to our server.
