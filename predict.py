#!/usr/bin/env python3
"""
Quick Prediction Script for XRP Price Forecasting
================================================

This script allows you to quickly make predictions with a trained model.
Usage: python predict.py <recent_data_file> [output_file]
"""

import sys
import os
from model import XRPPriceForecaster

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <recent_data_file> [output_file]")
        print("\nExample:")
        print("  python predict.py recent_1hour_data.json")
        print("  python predict.py recent_1hour_data.json my_prediction.json")
        sys.exit(1)
    
    recent_data_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "prediction_output.json"
    
    # Check if recent data file exists
    if not os.path.exists(recent_data_file):
        print(f"Error: Recent data file '{recent_data_file}' not found.")
        sys.exit(1)
    
    # Check if trained model exists
    model_path = "xrp_forecaster_model.h5"
    scaler_path = "scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: Trained model not found.")
        print("Please train the model first by running:")
        print("  python example_usage.py")
        print("or")
        print("  python model.py")
        sys.exit(1)
    
    print("🔮 XRP Price Prediction")
    print("=" * 40)
    
    try:
        # Initialize and load model
        forecaster = XRPPriceForecaster(sequence_length=20, prediction_hours=1)
        forecaster.load_model(model_path, scaler_path)
        
        # Make prediction
        result = forecaster.predict(recent_data_file, output_file)
        
        if result:
            print(f"\n✅ Prediction completed successfully!")
            print(f"📄 Output saved to: {output_file}")
            
            # Show summary
            prices = [float(data['close']) for data in result['market_data']]
            print(f"\n📊 Prediction Summary:")
            print(f"   • Predicted data points: {len(prices)}")
            print(f"   • Price range: ${min(prices):.4f} - ${max(prices):.4f}")
            print(f"   • Average price: ${sum(prices)/len(prices):.4f}")
            
            # Show trend
            first_price = prices[0]
            last_price = prices[-1]
            change = last_price - first_price
            change_pct = (change / first_price) * 100
            
            trend = "📈 Upward" if change > 0 else "📉 Downward" if change < 0 else "➡️ Sideways"
            print(f"   • Trend: {trend} ({change_pct:+.2f}%)")
            
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
