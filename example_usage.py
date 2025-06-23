#!/usr/bin/env python3
"""
XRP Price Forecasting - Usage Example
=====================================

This example demonstrates how to train and use the XRP price forecasting model.
"""

from model import XRPPriceForecaster
import json
import os

def train_model():
    """
    Train the XRP price forecasting model using the dataset
    """
    print("=" * 60)
    print("XRP PRICE FORECASTING MODEL - TRAINING")
    print("=" * 60)
    
    # Initialize the forecaster
    # sequence_length: Number of previous time steps to use for prediction
    # prediction_hours: Number of hours to predict into the future
    forecaster = XRPPriceForecaster(sequence_length=20, prediction_hours=1)
    
    # Train the model
    print("\nStarting model training...")
    history = forecaster.train(
        dataset_folder="dataset",
        validation_split=0.2,  # 20% for validation
        epochs=50,            # Number of training epochs
        batch_size=32         # Batch size for training
    )
    
    # Save the trained model
    forecaster.save_model("xrp_forecaster_model.h5", "scaler.pkl")
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model saved as: xrp_forecaster_model.h5")
    print(f"Scaler saved as: scaler.pkl")
    
    return forecaster

def make_prediction(model_path="xrp_forecaster_model.h5", scaler_path="scaler.pkl"):
    """
    Make a prediction using a trained model
    
    Args:
        model_path (str): Path to the trained model
        scaler_path (str): Path to the saved scaler
    """
    print("\n" + "=" * 60)
    print("XRP PRICE FORECASTING MODEL - PREDICTION")
    print("=" * 60)
    
    # Initialize the forecaster
    forecaster = XRPPriceForecaster(sequence_length=20, prediction_hours=1)
    
    # Load the trained model
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        forecaster.load_model(model_path, scaler_path)
        print("✓ Model loaded successfully!")
    else:
        print("✗ Trained model not found. Please train the model first.")
        return None
    
    # For demonstration, let's use one of the existing dataset files as "recent data"
    # In practice, you would provide the most recent 1 hour of data
    recent_data_file = "dataset/XRPUSDT_market_data_21June_to_22June_2025.json"
    
    if not os.path.exists(recent_data_file):
        print(f"✗ Recent data file not found: {recent_data_file}")
        return None
    
    print(f"✓ Using recent data from: {recent_data_file}")
    
    # Make prediction
    try:
        prediction_result = forecaster.predict(
            recent_data_file=recent_data_file,
            output_file="prediction_output.json"
        )
        
        print("\n" + "=" * 60)
        print("PREDICTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Prediction saved to: prediction_output.json")
        
        # Display some prediction details
        metadata = prediction_result['metadata']
        market_data = prediction_result['market_data']
        
        print(f"\nPrediction Details:")
        print(f"- Symbol: {metadata['symbol']}")
        print(f"- Prediction Type: {metadata['prediction_type']}")
        print(f"- Data Points: {metadata['data_count']}")
        print(f"- Time Range: {metadata['date_range']['start_date']} to {metadata['date_range']['end_date']}")
        
        # Show first few predictions
        print(f"\nFirst 5 Predicted Prices:")
        for i, data in enumerate(market_data[:5]):
            print(f"  {data['datetime']}: ${data['close']:.4f}")
        
        # Show price range
        prices = [float(data['close']) for data in market_data]
        print(f"\nPredicted Price Range: ${min(prices):.4f} - ${max(prices):.4f}")
        
        return prediction_result
        
    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")
        return None

def create_sample_recent_data():
    """
    Create a sample recent data file for demonstration
    This simulates the format you would use for real-time predictions
    """
    print("\n" + "=" * 60)
    print("CREATING SAMPLE RECENT DATA FILE")
    print("=" * 60)
    
    # Load one of the existing dataset files
    with open("dataset/XRPUSDT_market_data_21June_to_22June_2025.json", 'r') as f:
        data = json.load(f)
    
    # Take only the last 20 data points (1 hour worth of 3-minute candles)
    market_data = data['market_data'][-20:]
    
    # Create sample recent data structure
    sample_data = {
        "metadata": {
            "symbol": "XRPUSDT",
            "timestamp_generated": "2025-06-22T20:00:00",
            "data_count": len(market_data),
            "columns": ["timestamp", "open", "high", "low", "close"],
            "interval": "3min",
            "description": "Sample recent 1-hour data for prediction"
        },
        "market_data": market_data
    }
    
    # Save sample recent data
    with open("sample_recent_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"✓ Sample recent data created: sample_recent_data.json")
    print(f"  Data points: {len(market_data)}")
    print(f"  Time range: {market_data[0]['datetime']} to {market_data[-1]['datetime']}")
    
    return "sample_recent_data.json"

def main():
    """
    Main function to demonstrate the complete workflow
    """
    print("🚀 XRP PRICE FORECASTING MODEL")
    print("This example demonstrates training and prediction workflow.\n")
    
    # Check if dataset exists
    if not os.path.exists("dataset"):
        print("✗ Dataset folder not found. Please ensure the dataset folder exists.")
        return
    
    # Check if model is already trained
    if os.path.exists("xrp_forecaster_model.h5") and os.path.exists("scaler.pkl"):
        print("✓ Trained model found. Skipping training...")
        print("  If you want to retrain, delete the model files and run again.")
        
        # Make prediction with existing model
        make_prediction()
        
    else:
        print("🔄 No trained model found. Starting training process...")
        
        # Train the model
        forecaster = train_model()
        
        if forecaster:
            # Create sample recent data
            recent_data_file = create_sample_recent_data()
            
            # Make a prediction
            prediction_result = forecaster.predict(
                recent_data_file=recent_data_file,
                output_file="prediction_output.json"
            )
            
            if prediction_result:
                print("\n🎉 Complete workflow executed successfully!")
                print("\nNext steps:")
                print("1. Use your own recent 1-hour data in the same JSON format")
                print("2. Call forecaster.predict() with your data file")
                print("3. Get predictions for the next hour in 3-minute intervals")

if __name__ == "__main__":
    main()
