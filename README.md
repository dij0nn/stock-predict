# stock-predict
This script uses a type of neural network called Long Short-Term Memory (LSTM) to predict stock prices. Specifically, it forecasts future closing prices of Apple Inc. (AAPL) using historical price data from Yahoo Finance.

How it works:
1. Data Gathering: It pulls 10 years of daily AAPL stock data.

2. Normalization: Prices are scaled to a 0-1 range to help the neural network learn more effectively.

3. Windowing: The model uses the past 60 days of prices to predict the next day, creating a sliding window of data.

4. Model Design: It uses a two-layer LSTM network with dropout layers for regularization, followed by dense layers for final output.

5. Training: The model is trained on 80% of the data, learning to minimize prediction error.

6. Testing & Evaluation: It predicts on the remaining 20% and compares predictions to actual prices using Root Mean Squared Error (RMSE).

7. Visualization: A plot visually compares predicted prices vs. real prices.

Applications:
This was really just a small project for me to learn time-series prediction and neural networks in the context of finance. Because of that, this is by no means the most fleshed out algorithmic trading system, but it gets the job done and acts as a vague prototype for a financial analytic dashboard of sorts...

Potential Improvements:
 I feel like next time I could include additional features like volume, open/high/low prices and use more advanced models (e.g., Bidirectional LSTM, GRU, or Transformer). I could also implement early stopping and hyperparameter tuning and use walk-forward validation or k-fold time-series cross-validation for effieciency
