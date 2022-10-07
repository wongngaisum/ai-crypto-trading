# Backtesting

## Prerequisites

- [`Python 3`](https://www.python.org/)

## How to use

1. Install all dependencies defined in the requirements.txt
   ```bash
    pip install -r requirements.txt
    ```
   
2. You can modify the ``launcher.py`` to choose the models you want to run.
   ```python
    if __name__ == '__main__':
    np.random.seed(1234)
    keras.utils.set_random_seed(1234)
    bt = Backtest(get_data_frame(), HodlWalkForewardStrategy, ## Change this to your strategy
                  cash=1000000, commission=args.fee, # 0.1% trading fee + 0.1% silppage
                  exclusive_orders=True)
    ``` 

3. Run the following command
    ```bash
    python launcher.py <lag> <fee> <close_first>
    ```
    where `<lag>` is the lag window of social media indicators (-1 = no lag, 0 = 1 hour lag and so on), `<fee>` trading fee rate, and `<close_first>` (1 or 0) is do not open opposite position immediately, close first. After you changed lag window, please remove the computed computed_indicators.csv first.
   
## Strategies
- combine_strategy: combine multiple strategies (All machine learning models)
- combine_strategy2: combine multiple strategies (Well-performed machine learning models, which Sharpe ratio >= 1)
- hold_strategy: hold the position forever
- cnn_strategy: CNN strategy
- knn_strategy: KNN strategy
- lstm_strategy: LSTM strategy
- mlp_strategy: MLP strategy
- rf_strategy: Random Forest strategy
- sma_strategy: SMA strategy
- sma_bolling_strategy: SMA with Bolling strategy
- xgb_strategy: XGB strategy

# Authors
- Wong Ngai Sum
- Lin Ho Ching Janus

# Code references
- ML backtesting & walk-forward testing: https://kernc.github.io/backtesting.py/doc/examples/Trading%20with%20Machine%20Learning.html
- Twitter emotion analysis: https://www.kaggle.com/code/ishivinal/tweet-emotions-analysis-using-lstm-glove-roberta?scriptVersionId=38608295
- 101 Alphas: https://github.com/yli188/WorldQuant_alpha101_code
- Sklearn
- Keras 