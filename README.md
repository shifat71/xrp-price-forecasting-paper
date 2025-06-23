# XRP Price Forecasting Model

A sophisticated machine learning model for predicting XRP/USDT cryptocurrency prices using LSTM (Long Short-Term Memory) neural networks. This model analyzes historical candlestick data and technical indicators to forecast future price movements.

## 🎯 Features

- **Deep Learning Architecture**: Uses LSTM neural networks for time series prediction
- **Technical Analysis**: Incorporates 20+ technical indicators including:
  - Moving Averages (SMA, EMA)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - MACD (Moving Average Convergence Divergence)
  - Price volatility and momentum indicators
- **Flexible Prediction**: Configurable sequence length and prediction horizon
- **Real-time Ready**: Accepts recent data and outputs predictions in the same format
- **Model Persistence**: Save and load trained models for future use

## 📊 Data Format

The model works with JSON files containing OHLC (Open, High, Low, Close) candlestick data in 3-minute intervals:

```json
{
  "metadata": {
    "symbol": "XRPUSDT",
    "interval": "3min",
    "data_count": 480
  },
  "market_data": [
    {
      "timestamp_ms": 1750068000000,
      "datetime": "2025-06-16 16:00:00",
      "open": 2.1981,
      "high": 2.1989,
      "low": 2.1965,
      "close": 2.1989
    }
  ]
}
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```python
from model import XRPPriceForecaster

# Initialize the forecaster
forecaster = XRPPriceForecaster(sequence_length=20, prediction_hours=1)

# Train on your dataset
forecaster.train(
    dataset_folder="dataset",
    epochs=50,
    batch_size=32
)

# Save the trained model
forecaster.save_model("xrp_model.h5", "scaler.pkl")
```

### 3. Make Predictions

```python
# Load trained model
forecaster = XRPPriceForecaster()
forecaster.load_model("xrp_model.h5", "scaler.pkl")

# Make prediction with recent 1-hour data
prediction = forecaster.predict(
    recent_data_file="recent_1hour_data.json",
    output_file="prediction_output.json"
)
```

### 4. Run Example

```bash
python example_usage.py
```

## 🏗️ Model Architecture

The model uses a multi-layer LSTM architecture:

```
Input Layer (sequence_length, features)
    ↓
LSTM Layer (128 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (64 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (32 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Output Layer (prediction_steps units, Linear)
```

## 📈 Technical Indicators

The model incorporates the following technical indicators:

### Price-Based Features
- **OHLC Values**: Open, High, Low, Close prices
- **Price Range**: High - Low
- **Price Change**: Close - Open
- **Price Change %**: Percentage change from open to close

### Moving Averages
- **SMA**: Simple Moving Averages (5, 10, 20 periods)
- **EMA**: Exponential Moving Averages (5, 10 periods)

### Momentum Indicators
- **RSI**: Relative Strength Index (14 periods)
- **MACD**: Moving Average Convergence Divergence
- **MACD Signal**: Signal line for MACD
- **MACD Histogram**: MACD - Signal

### Volatility Indicators
- **Bollinger Bands**: Upper, Middle, Lower bands (20 periods, 2 std dev)
- **Bollinger Band Width**: Upper - Lower
- **Bollinger Band Position**: Relative position within bands
- **Price Volatility**: Rolling standard deviation (10 periods)

### Time-Based Features
- **Hour**: Hour of the day
- **Minute**: Minute of the hour
- **Day of Week**: Day of the week (0-6)

## 🔧 Configuration Parameters

### XRPPriceForecaster Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sequence_length` | 20 | Number of previous timesteps to use for prediction |
| `prediction_hours` | 1 | Number of hours to predict into the future |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `validation_split` | 0.2 | Fraction of data to use for validation |
| `epochs` | 100 | Number of training epochs |
| `batch_size` | 32 | Training batch size |

## 📁 File Structure

```
Research on Crypto Price Forecasting/
├── model.py                 # Main model implementation
├── example_usage.py         # Usage example and demo
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── dataset/                # Training data folder
│   ├── XRPUSDT_market_data_15June_to_16June_2025.json
│   ├── XRPUSDT_market_data_16June_to_17June_2025.json
│   └── ... (more dataset files)
└── bybit_api/              # API utilities (if needed)
```

## 🎯 Usage Examples

### Basic Training and Prediction

```python
from model import XRPPriceForecaster

# Initialize and train
forecaster = XRPPriceForecaster(sequence_length=20, prediction_hours=1)
forecaster.train(dataset_folder="dataset", epochs=50)

# Save model
forecaster.save_model("my_model.h5", "my_scaler.pkl")

# Load and predict
forecaster = XRPPriceForecaster()
forecaster.load_model("my_model.h5", "my_scaler.pkl")
result = forecaster.predict("recent_data.json", "output.json")
```

### Custom Configuration

```python
# Extended prediction horizon
forecaster = XRPPriceForecaster(
    sequence_length=30,    # Use 30 previous timesteps
    prediction_hours=2     # Predict 2 hours ahead
)

# Custom training parameters
forecaster.train(
    dataset_folder="dataset",
    validation_split=0.15,  # 15% validation
    epochs=100,             # More epochs
    batch_size=64          # Larger batch size
)
```

## 📊 Model Performance

The model uses the following metrics for evaluation:
- **MSE (Mean Squared Error)**: Primary loss function
- **MAE (Mean Absolute Error)**: Secondary metric
- **Validation Loss**: Monitor for overfitting

Training includes:
- **Early Stopping**: Prevents overfitting
- **Learning Rate Reduction**: Adaptive learning rate
- **Dropout Layers**: Regularization technique

## 🔮 Prediction Output

The model outputs predictions in the same format as the input data:

```json
{
  "metadata": {
    "symbol": "XRPUSDT",
    "prediction_type": "next_hour_forecast",
    "data_count": 20,
    "model_info": {
      "sequence_length": 20,
      "prediction_hours": 1,
      "prediction_steps": 20
    }
  },
  "market_data": [
    {
      "timestamp_ms": 1750068180000,
      "datetime": "2025-06-16 16:03:00",
      "open": 2.1995,
      "high": 2.2010,
      "low": 2.1980,
      "close": 2.2000
    }
  ]
}
```

## ⚠️ Important Notes

1. **Data Quality**: Ensure your input data follows the exact JSON format
2. **Sequence Length**: Recent data must have at least `sequence_length` data points
3. **Feature Engineering**: The model automatically calculates technical indicators
4. **Time Intervals**: Designed for 3-minute candlestick data
5. **Market Conditions**: Model performance may vary with market volatility

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Install all dependencies with `pip install -r requirements.txt`
2. **Data Format**: Ensure JSON files match the expected structure
3. **Memory Issues**: Reduce batch_size if you encounter memory errors
4. **Training Time**: Use GPU acceleration for faster training (install tensorflow-gpu)

### Performance Tips

1. **More Data**: More training data generally improves performance
2. **Feature Engineering**: Add more technical indicators if needed
3. **Hyperparameter Tuning**: Experiment with different parameters
4. **Ensemble Methods**: Combine multiple models for better accuracy

## 📄 License

This project is for educational and research purposes. Please ensure compliance with relevant financial regulations when using for trading decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## ⚡ Disclaimer

This model is for educational purposes only. Cryptocurrency trading involves significant risk, and past performance does not guarantee future results. Always do your own research and consider consulting with financial advisors before making investment decisions.