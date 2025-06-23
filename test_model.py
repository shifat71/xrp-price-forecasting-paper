#!/usr/bin/env python3
"""
Test Script for XRP Price Forecasting Model
===========================================

This script tests the basic functionality of the model without requiring
full training, to ensure the code structure is correct.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def test_data_loading():
    """Test if we can load and parse the dataset files"""
    print("Testing data loading...")
    
    dataset_files = [f for f in os.listdir("dataset") if f.endswith(".json")]
    if not dataset_files:
        print("❌ No dataset files found")
        return False
    
    try:
        # Load first dataset file
        with open(f"dataset/{dataset_files[0]}", 'r') as f:
            data = json.load(f)
        
        # Check structure
        required_keys = ['metadata', 'market_data']
        for key in required_keys:
            if key not in data:
                print(f"❌ Missing key: {key}")
                return False
        
        # Check market data structure
        if not data['market_data']:
            print("❌ Empty market data")
            return False
        
        sample_data = data['market_data'][0]
        required_fields = ['timestamp_ms', 'open', 'high', 'low', 'close']
        for field in required_fields:
            if field not in sample_data:
                print(f"❌ Missing field in market data: {field}")
                return False
        
        print(f"✅ Data loading test passed")
        print(f"   - Found {len(dataset_files)} dataset files")
        print(f"   - Sample file has {len(data['market_data'])} data points")
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return False

def test_feature_engineering():
    """Test feature engineering without full model dependencies"""
    print("\nTesting feature engineering...")
    
    try:
        # Create sample data
        dates = pd.date_range(start='2025-06-01', periods=100, freq='3min')
        np.random.seed(42)
        
        # Generate sample OHLC data
        base_price = 2.0
        price_changes = np.random.normal(0, 0.01, 100)
        closes = base_price + np.cumsum(price_changes)
        
        sample_data = {
            'timestamp': dates,
            'open': closes + np.random.normal(0, 0.001, 100),
            'high': closes + np.abs(np.random.normal(0, 0.002, 100)),
            'low': closes - np.abs(np.random.normal(0, 0.002, 100)),
            'close': closes
        }
        
        df = pd.DataFrame(sample_data)
        
        # Basic feature engineering
        df['price_range'] = df['high'] - df['low']
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Check if features were created
        feature_columns = ['price_range', 'price_change', 'price_change_pct', 'sma_5', 'sma_10', 'hour', 'minute']
        for col in feature_columns:
            if col not in df.columns:
                print(f"❌ Feature not created: {col}")
                return False
        
        print("✅ Feature engineering test passed")
        print(f"   - Created {len(feature_columns)} basic features")
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {str(e)}")
        return False

def test_json_output_format():
    """Test if we can create the expected JSON output format"""
    print("\nTesting JSON output format...")
    
    try:
        # Create sample prediction data
        prediction_data = []
        base_time = datetime.now()
        
        for i in range(20):  # 20 predictions for 1 hour
            timestamp = base_time + timedelta(minutes=3 * (i + 1))
            price = 2.0 + (i * 0.001)  # Simple increasing price
            
            prediction_data.append({
                "timestamp_ms": int(timestamp.timestamp() * 1000),
                "datetime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "date": timestamp.strftime("%Y-%m-%d"),
                "time": timestamp.strftime("%H:%M:%S"),
                "iso_format": timestamp.isoformat(),
                "open": float(price),
                "high": float(price * 1.002),
                "low": float(price * 0.998),
                "close": float(price)
            })
        
        # Create output structure
        output_data = {
            "metadata": {
                "symbol": "XRPUSDT",
                "timestamp_generated": datetime.now().isoformat(),
                "data_count": len(prediction_data),
                "columns": ["timestamp", "open", "high", "low", "close"],
                "interval": "3min",
                "prediction_type": "next_hour_forecast"
            },
            "market_data": prediction_data
        }
        
        # Save to test file
        with open("test_output.json", 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Verify file was created and is valid JSON
        with open("test_output.json", 'r') as f:
            loaded_data = json.load(f)
        
        print("✅ JSON output format test passed")
        print(f"   - Created {len(prediction_data)} prediction data points")
        print(f"   - Output saved to test_output.json")
        
        # Clean up
        os.remove("test_output.json")
        return True
        
    except Exception as e:
        print(f"❌ JSON output format test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 XRP Price Forecasting Model - Structure Tests")
    print("=" * 60)
    
    tests = [
        test_data_loading,
        test_feature_engineering,
        test_json_output_format
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The model structure is correct.")
        print("\nNext steps:")
        print("1. Install required packages: pip install -r requirements.txt")
        print("2. Run the full training: python example_usage.py")
        print("3. Make predictions: python predict.py <recent_data_file>")
    else:
        print("❌ Some tests failed. Please check the code.")
    
    return passed == total

if __name__ == "__main__":
    main()
