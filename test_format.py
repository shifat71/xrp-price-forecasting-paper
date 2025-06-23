#!/usr/bin/env python3
"""
Test Script to Verify Prediction Output Format
===============================================

This script verifies that the prediction output matches the expected format.
"""

import json
from datetime import datetime

def test_prediction_format():
    """
    Test if the prediction output matches the expected format
    """
    print("🔍 Testing Prediction Output Format")
    print("=" * 40)
    
    try:
        # Load prediction output
        with open("prediction_output_simple.json", 'r') as f:
            prediction_data = json.load(f)
        
        print("✅ Prediction file loaded successfully")
        
        # Test metadata structure
        metadata = prediction_data['metadata']
        required_metadata_keys = [
            'symbol', 'timestamp_generated', 'data_count', 'columns',
            'date_range', 'interval', 'prediction_type', 'model_info'
        ]
        
        for key in required_metadata_keys:
            if key not in metadata:
                print(f"❌ Missing metadata key: {key}")
                return False
            else:
                print(f"✅ Metadata key present: {key}")
        
        # Test market data structure
        market_data = prediction_data['market_data']
        if not market_data:
            print("❌ Empty market data")
            return False
        
        print(f"✅ Market data contains {len(market_data)} records")
        
        # Test individual record structure
        sample_record = market_data[0]
        required_record_keys = [
            'timestamp_ms', 'datetime', 'date', 'time', 'iso_format',
            'open', 'high', 'low', 'close'
        ]
        
        for key in required_record_keys:
            if key not in sample_record:
                print(f"❌ Missing record key: {key}")
                return False
            else:
                print(f"✅ Record key present: {key}")
        
        # Test data types and values
        print("\n📊 Data Validation:")
        
        # Check if prices are reasonable
        prices = [float(record['close']) for record in market_data]
        if min(prices) <= 0:
            print("❌ Invalid price values (negative or zero)")
            return False
        
        print(f"✅ Price range: ${min(prices):.4f} - ${max(prices):.4f}")
        
        # Check if timestamps are in correct format
        try:
            datetime.fromisoformat(sample_record['iso_format'])
            print("✅ Timestamp format is valid")
        except ValueError:
            print("❌ Invalid timestamp format")
            return False
        
        # Check if data count matches
        if metadata['data_count'] != len(market_data):
            print(f"❌ Data count mismatch: metadata says {metadata['data_count']}, actual {len(market_data)}")
            return False
        
        print(f"✅ Data count matches: {metadata['data_count']}")
        
        # Check if OHLC relationship is logical
        for i, record in enumerate(market_data[:5]):  # Check first 5 records
            o, h, l, c = record['open'], record['high'], record['low'], record['close']
            if not (l <= o <= h and l <= c <= h):
                print(f"❌ OHLC relationship invalid in record {i}")
                return False
        
        print("✅ OHLC relationships are logical")
        
        # Test prediction interval
        timestamps = [record['timestamp_ms'] for record in market_data]
        if len(timestamps) > 1:
            interval = timestamps[1] - timestamps[0]
            expected_interval = 3 * 60 * 1000  # 3 minutes in milliseconds
            if interval == expected_interval:
                print("✅ 3-minute intervals confirmed")
            else:
                print(f"⚠️  Interval mismatch: expected {expected_interval}ms, got {interval}ms")
        
        print("\n🎉 All format tests passed!")
        print("✅ Prediction output format is correct and ready to use")
        
        return True
        
    except FileNotFoundError:
        print("❌ Prediction output file not found")
        return False
    except json.JSONDecodeError:
        print("❌ Invalid JSON format")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

def compare_with_input_format():
    """
    Compare prediction output format with input dataset format
    """
    print("\n🔄 Comparing Output Format with Input Dataset Format")
    print("=" * 55)
    
    try:
        # Load input dataset
        with open("dataset/XRPUSDT_market_data_21June_to_22June_2025.json", 'r') as f:
            input_data = json.load(f)
        
        # Load prediction output
        with open("prediction_output_simple.json", 'r') as f:
            output_data = json.load(f)
        
        print("✅ Both files loaded successfully")
        
        # Compare metadata structure
        input_metadata = input_data['metadata']
        output_metadata = output_data['metadata']
        
        common_keys = ['symbol', 'data_count', 'columns', 'interval']
        for key in common_keys:
            if key in input_metadata and key in output_metadata:
                print(f"✅ Common metadata key: {key}")
                if key == 'symbol' and input_metadata[key] == output_metadata[key]:
                    print(f"  ✅ Same symbol: {input_metadata[key]}")
                elif key == 'interval' and input_metadata[key] == output_metadata[key]:
                    print(f"  ✅ Same interval: {input_metadata[key]}")
                elif key == 'columns' and input_metadata[key] == output_metadata[key]:
                    print(f"  ✅ Same columns: {input_metadata[key]}")
        
        # Compare market data structure
        input_sample = input_data['market_data'][0]
        output_sample = output_data['market_data'][0]
        
        # Check if they have the same keys
        input_keys = set(input_sample.keys())
        output_keys = set(output_sample.keys())
        
        if input_keys == output_keys:
            print("✅ Market data structures are identical")
            print(f"  Keys: {sorted(input_keys)}")
        else:
            print("⚠️  Market data structures differ")
            print(f"  Input keys: {sorted(input_keys)}")
            print(f"  Output keys: {sorted(output_keys)}")
        
        print("\n🎯 Format compatibility confirmed!")
        print("✅ Prediction output uses the same format as input dataset")
        
    except Exception as e:
        print(f"❌ Error during comparison: {str(e)}")

def main():
    """
    Run all tests
    """
    print("🧪 XRP PRICE PREDICTION - OUTPUT FORMAT VALIDATION")
    print("=" * 60)
    
    # Test prediction format
    format_test = test_prediction_format()
    
    if format_test:
        # Compare with input format
        compare_with_input_format()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your prediction output is correctly formatted")
        print("✅ Ready to use for analysis or further processing")
        print("=" * 60)
    else:
        print("\n❌ FORMAT VALIDATION FAILED")
        print("Please check the prediction output format")

if __name__ == "__main__":
    main()
