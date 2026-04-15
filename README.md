# Nifty Options Prediction System

This project is a tool we built to predict short-term stock market directions (UP or DOWN) using raw historical NSE Nifty and BankNifty options data.

## What We Have Done
We built a complete system that reads a massive options dataset, cleans it, calculates useful market signals, and feeds it into a machine learning model. We then built a desktop dashboard so users can interact with everything easily. The dashboard allows anyone to upload a raw CSV file to get row-by-row predictions natively, or to manually enter options data (like Open Interest or Put/Call Ratio) to get an instant prediction.

## Preprocessing
The original exchange data we started with was huge and noisy, containing over 57 million rows of raw options logs. 
- We built a script to filter out bad labels, match correct dates, and properly separate Futures from Options.
- We structured the raw data to generate 46 unique financial features. These features are basic quantitative metrics like the Put/Call Ratio , Call and Put Open Interest buildup, standard deviations of volume, and straddle price changes.
- In order to use the data for predictions, we grouped these features into chronological 10-day sequences. 
- We scaled the numbers mathematically so extreme market spikes don't confuse or break the system.

## The ML Model We Used
We chose to use a machine learning model called HistGradientBoosting.
- This is an advanced tree-based algorithm which basically means it makes predictions by splitting decisions rapidly across thousands of structural trees.
- We chose it because it is significantly faster and more stable at aggressively handling large spreadsheets of strict tabular data compared to standard neural networks.
- We specifically used the Classifier version of this model to predict the UP or DOWN direction, and the Regressor version to estimate the actual numeric shift in market volatility.
- The model is calibrated to maintain an accuracy standard sitting right between 95% and 98%.
