# Crypto Price Prediction – DRW Kaggle Challenge
This project contains my solution to the DRW Crypto Market Prediction Challenge, hosted on Kaggle. The objective is to forecast short-term price movements in crypto asset using high-frequency order book data. The challenge simulates realistic market conditions and requires modeling in a highly noisy, low-signal environment.

## Problem Statement
Predict the next price movement in crypto markets using anonymized, timestamped features derived from Level 2 order book data. The competition evaluates models based on the Pearson correlation coefficient between predicted and actual price movements.

## Approach
### Feature Engineering
- Created lag features (1, 5, 10 ticks) for bid/ask/volume-related variables
- Computed rolling means, momentum and volatility features (5 and 10 tick windows)
- Applied StandardScaler to normalize all features (including X1–X780)
- Filtered engineered features and base features using correlation thresholding

### Feature Selection
- Used correlation with label to pre-filter irrelevant features
- Applied LightGBM feature importance to select top 100 features
- Removed label-dependent features to prevent leakage

### Modeling
- Trained models using LightGBM, XGBoost, CatBoost
- Hyperparameter tuning using Optuna
- Applied TimeSeriesSplit for proper temporal validation

## Results
- Final leaderboard rank: ~600 out of 1200+ participants
- Achieved a Pearson correlation of ~0.055 on test submissions
- Learned critical modeling lessons on low signal environments, feature relevance, and temporal leakage

## Key Learnings
- Practical experience in modeling under noisy market conditions
- Tradeoffs in feature complexity vs. signal stability
- Importance of temporal CV and careful leakage prevention
- Limitations of modeling high-frequency crypto price movement using standard tabular ML