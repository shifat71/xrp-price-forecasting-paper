import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

class XRPPriceForecaster:
    def __init__(self, sequence_length=20, prediction_hours=1):
        """
        Initialize the XRP price forecasting model
        
        Args:
            sequence_length (int): Number of previous timesteps to use for prediction
            prediction_hours (int): Number of hours to predict into the future
        """
        self.sequence_length = sequence_length
        self.prediction_hours = prediction_hours
        self.prediction_steps = prediction_hours * 20  # 20 three-minute intervals per hour
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        
    def load_dataset(self, dataset_folder="dataset"):
        """
        Load all JSON files from the dataset folder
        
        Args:
            dataset_folder (str): Path to the folder containing dataset files
            
        Returns:
            pd.DataFrame: Combined dataset with all market data
        """
        all_data = []
        json_files = glob.glob(os.path.join(dataset_folder, "*.json"))
        
        print(f"Loading {len(json_files)} dataset files...")
        
        for file_path in sorted(json_files):
            with open(file_path, 'r') as f:
                data = json.load(f)
                market_data = data['market_data']
                
                # Convert to DataFrame
                df = pd.DataFrame(market_data)
                df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
                all_data.append(df)
        
        # Combine all data and sort by timestamp
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates based on timestamp
        combined_df = combined_df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        print(f"Total data points loaded: {len(combined_df)}")
        print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        
        return combined_df
    
    def prepare_features(self, df):
        """
        Prepare features for training
        
        Args:
            df (pd.DataFrame): Raw market data
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        df = df.copy()
        
        # Basic OHLC features
        df['price_range'] = df['high'] - df['low']
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        
        # Technical indicators
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume-based features (if available)
        # For now, we'll create a synthetic volume indicator based on price volatility
        df['volatility'] = df['close'].rolling(window=10).std()
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Fill NaN values - using forward and backward fill
        df = df.bfill().ffill()
        
        return df
    
    def create_sequences(self, data, target_column='close'):
        """
        Create sequences for LSTM training
        
        Args:
            data (np.array): Preprocessed data
            target_column (str): Column to predict
            
        Returns:
            tuple: (X, y) sequences for training
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.prediction_steps + 1):
            # Input sequence
            X.append(data[i-self.sequence_length:i])
            
            # Target sequence (next prediction_steps values)
            y.append(data[i:i+self.prediction_steps, 3])  # Index 3 is close price
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build the LSTM model architecture
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            tensorflow.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.prediction_steps, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, dataset_folder="dataset", validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the model on the dataset
        
        Args:
            dataset_folder (str): Path to dataset folder
            validation_split (float): Fraction of data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
        """
        print("Loading and preparing data...")
        
        # Load dataset
        df = self.load_dataset(dataset_folder)
        
        # Prepare features
        df_features = self.prepare_features(df)
        
        # Select features for training
        feature_columns = [
            'open', 'high', 'low', 'close', 'price_range', 'price_change', 'price_change_pct',
            'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'rsi',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'macd', 'macd_signal', 'macd_histogram', 'volatility',
            'hour', 'minute', 'day_of_week'
        ]
        
        # Prepare data for scaling
        data = df_features[feature_columns].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        print(f"Created {len(X)} sequences for training")
        print(f"Input shape: {X.shape}, Output shape: {y.shape}")
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        print("Model architecture:")
        self.model.summary()
        
        # Train model
        print("\nStarting training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
        )
        
        self.is_trained = True
        print("Training completed!")
        
        # Evaluate model
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        print(f"\nFinal Training Loss: {train_loss[0]:.6f}")
        print(f"Final Validation Loss: {val_loss[0]:.6f}")
        
        return history
    
    def predict(self, recent_data_file, output_file="prediction_output.json"):
        """
        Make prediction based on recent 1 hour data
        
        Args:
            recent_data_file (str): Path to JSON file with recent 1 hour data
            output_file (str): Path to save prediction output
            
        Returns:
            dict: Prediction results in the same format as dataset
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        print(f"Loading recent data from: {recent_data_file}")
        
        # Load recent data
        with open(recent_data_file, 'r') as f:
            recent_data = json.load(f)
        
        # Convert to DataFrame
        df_recent = pd.DataFrame(recent_data['market_data'])
        df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp_ms'], unit='ms')
        df_recent = df_recent.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Recent data points: {len(df_recent)}")
        print(f"Time range: {df_recent['timestamp'].min()} to {df_recent['timestamp'].max()}")
        
        # Prepare features for recent data
        df_features = self.prepare_features(df_recent)
        
        # Select same features used in training
        feature_columns = [
            'open', 'high', 'low', 'close', 'price_range', 'price_change', 'price_change_pct',
            'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'rsi',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'macd', 'macd_signal', 'macd_histogram', 'volatility',
            'hour', 'minute', 'day_of_week'
        ]
        
        data = df_features[feature_columns].values
        
        # Scale the data using the same scaler
        scaled_data = self.scaler.transform(data)
        
        # Use the last sequence_length points for prediction
        if len(scaled_data) < self.sequence_length:
            raise ValueError(f"Recent data must have at least {self.sequence_length} data points")
        
        # Prepare input for prediction
        X_pred = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Make prediction
        prediction = self.model.predict(X_pred, verbose=0)
        
        # Inverse transform the prediction (only the close price)
        # Create a dummy array with the same shape as the original features
        dummy_features = np.zeros((self.prediction_steps, len(feature_columns)))
        dummy_features[:, 3] = prediction[0]  # Close price is at index 3
        
        # Inverse transform
        prediction_inversed = self.scaler.inverse_transform(dummy_features)
        predicted_prices = prediction_inversed[:, 3]  # Extract close prices
        
        # Generate timestamps for predictions
        last_timestamp = df_recent['timestamp'].iloc[-1]
        prediction_timestamps = []
        
        for i in range(1, self.prediction_steps + 1):
            pred_timestamp = last_timestamp + timedelta(minutes=3 * i)
            prediction_timestamps.append(pred_timestamp)
        
        # Create prediction output in the same format as dataset
        prediction_data = []
        
        for i, (timestamp, price) in enumerate(zip(prediction_timestamps, predicted_prices)):
            # For simplicity, we'll use the predicted price as OHLC
            # In a more sophisticated model, you could predict OHLC separately
            prediction_data.append({
                "timestamp_ms": int(timestamp.timestamp() * 1000),
                "datetime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "date": timestamp.strftime("%Y-%m-%d"),
                "time": timestamp.strftime("%H:%M:%S"),
                "iso_format": timestamp.isoformat(),
                "open": float(price),
                "high": float(price * 1.002),  # Simulate small spread
                "low": float(price * 0.998),   # Simulate small spread
                "close": float(price)
            })
        
        # Create output structure
        output_data = {
            "metadata": {
                "symbol": "XRPUSDT",
                "timestamp_generated": datetime.now().isoformat(),
                "data_count": len(prediction_data),
                "columns": ["timestamp", "open", "high", "low", "close"],
                "date_range": {
                    "start_timestamp": int(prediction_timestamps[0].timestamp() * 1000),
                    "end_timestamp": int(prediction_timestamps[-1].timestamp() * 1000),
                    "start_date": prediction_timestamps[0].isoformat(),
                    "end_date": prediction_timestamps[-1].isoformat()
                },
                "interval": "3min",
                "prediction_type": "next_hour_forecast",
                "model_info": {
                    "sequence_length": self.sequence_length,
                    "prediction_hours": self.prediction_hours,
                    "prediction_steps": self.prediction_steps
                }
            },
            "market_data": prediction_data
        }
        
        # Save prediction to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Prediction saved to: {output_file}")
        print(f"Predicted price range: {min(predicted_prices):.4f} - {max(predicted_prices):.4f}")
        
        return output_data
    
    def save_model(self, model_path="xrp_forecaster_model.h5", scaler_path="scaler.pkl"):
        """
        Save the trained model and scaler
        
        Args:
            model_path (str): Path to save the model
            scaler_path (str): Path to save the scaler
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(model_path)
        
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
    def load_model(self, model_path="xrp_forecaster_model.h5", scaler_path="scaler.pkl"):
        """
        Load a pre-trained model and scaler
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str): Path to the saved scaler
        """
        self.model = tf.keras.models.load_model(model_path)
        
        import pickle
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.is_trained = True
        print(f"Model loaded from: {model_path}")
        print(f"Scaler loaded from: {scaler_path}")


def main():
    """
    Example usage of the XRP Price Forecaster
    """
    # Initialize the forecaster
    forecaster = XRPPriceForecaster(sequence_length=20, prediction_hours=1)
    
    # Train the model
    print("Training XRP Price Forecasting Model...")
    history = forecaster.train(
        dataset_folder="dataset",
        validation_split=0.2,
        epochs=50,
        batch_size=32
    )
    
    # Save the trained model
    forecaster.save_model("xrp_forecaster_model.h5", "scaler.pkl")
    
    print("\nModel training completed!")
    print("To make predictions, use the predict() method with recent 1-hour data.")
    

if __name__ == "__main__":
    main()


