#!/usr/bin/env python3
"""
XRP Price Forecasting - Simple Version (No TensorFlow required)
================================================================

This example demonstrates training and prediction using Random Forest instead of LSTM.
"""

from model_simple import XRPPriceForecasterSimple
import json
import os

def train_and_predict():
    """
    Complete workflow: train model and make predictions
    """
    print("🚀 XRP PRICE FORECASTING MODEL - SIMPLE VERSION")
    print("=" * 60)
    print("Using Random Forest instead of LSTM (no TensorFlow required)")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists("dataset"):
        print("✗ Dataset folder not found. Please ensure the dataset folder exists.")
        return
    
    # Initialize the forecaster
    forecaster = XRPPriceForecasterSimple(sequence_length=20, prediction_hours=1)
    
    # Train the model
    print("\nStarting model training...")
    forecaster.train(
        dataset_folder="dataset",
        validation_split=0.2,
        n_estimators=100
    )
    
    # Save the trained model
    forecaster.save_model("xrp_forecaster_simple.pkl", "scaler_simple.pkl")
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Create sample recent data from the last dataset file
    dataset_files = [f for f in os.listdir("dataset") if f.endswith(".json")]
    recent_data_file = f"dataset/{sorted(dataset_files)[-1]}"
    
    print(f"\nUsing recent data from: {recent_data_file}")
    
    # Make prediction
    try:
        prediction_result = forecaster.predict(
            recent_data_file=recent_data_file,
            output_file="prediction_output_simple.json"
        )
        
        print("\n" + "=" * 60)
        print("PREDICTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Prediction saved to: prediction_output_simple.json")
        
        # Display some prediction details
        metadata = prediction_result['metadata']
        market_data = prediction_result['market_data']
        
        print(f"\nPrediction Details:")
        print(f"- Symbol: {metadata['symbol']}")
        print(f"- Model Type: {metadata['model_info']['model_type']}")
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
        
        # Calculate trend
        first_price = prices[0]
        last_price = prices[-1]
        change = last_price - first_price
        change_pct = (change / first_price) * 100
        
        trend_direction = "📈 Upward" if change > 0 else "📉 Downward" if change < 0 else "➡️ Sideways"
        print(f"Predicted Trend: {trend_direction} ({change_pct:+.2f}%)")
        
        print("\n🎉 Complete workflow executed successfully!")
        print("\nNext steps:")
        print("1. Use your own recent 1-hour data in the same JSON format")
        print("2. Call forecaster.predict() with your data file")
        print("3. Get predictions for the next hour in 3-minute intervals")
        
        return prediction_result
        
    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")
        return None

def load_and_predict():
    """
    Load existing model and make prediction
    """
    print("\n" + "=" * 60)
    print("LOADING EXISTING MODEL FOR PREDICTION")
    print("=" * 60)
    
    model_path = "xrp_forecaster_simple.pkl"
    scaler_path = "scaler_simple.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("✗ Trained model not found. Please train the model first.")
        return None
    
    # Initialize and load model
    forecaster = XRPPriceForecasterSimple()
    forecaster.load_model(model_path, scaler_path)
    
    # Use sample data for prediction
    dataset_files = [f for f in os.listdir("dataset") if f.endswith(".json")]
    recent_data_file = f"dataset/{sorted(dataset_files)[-1]}"
    
    # Make prediction
    result = forecaster.predict(recent_data_file, "quick_prediction.json")
    
    if result:
        prices = [float(data['close']) for data in result['market_data']]
        print(f"✅ Quick prediction completed!")
        print(f"Price range: ${min(prices):.4f} - ${max(prices):.4f}")
    
    return result

def main():
    """
    Main function
    """
    # Check if model already exists
    if os.path.exists("xrp_forecaster_simple.pkl") and os.path.exists("scaler_simple.pkl"):
        print("✅ Trained model found!")
        choice = input("Do you want to (1) retrain or (2) use existing model? Enter 1 or 2: ")
        
        if choice == "2":
            load_and_predict()
            return
    
    # Train and predict
    train_and_predict()

if __name__ == "__main__":
    main()
