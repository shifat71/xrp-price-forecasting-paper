#!/usr/bin/env python3
"""
Prediction Summary Generator
===========================

This script creates a clean summary of all prediction results.
"""

import os
import json
import glob
from datetime import datetime

def generate_clean_summary():
    """
    Generate a clean summary of all predictions
    """
    output_folder = "dataset/model_output"
    prediction_files = glob.glob(os.path.join(output_folder, "prediction_*.json"))
    
    if not prediction_files:
        print("No prediction files found!")
        return
    
    print("🔮 XRP PRICE PREDICTION SUMMARY")
    print("=" * 80)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total predictions: {len(prediction_files)}")
    print("=" * 80)
    
    summary_data = []
    
    for pred_file in sorted(prediction_files):
        try:
            with open(pred_file, 'r') as f:
                data = json.load(f)
            
            # Extract information
            filename = os.path.basename(pred_file)
            metadata = data['metadata']
            market_data = data['market_data']
            
            # Calculate statistics
            prices = [float(item['close']) for item in market_data]
            first_price = prices[0]
            last_price = prices[-1]
            min_price = min(prices)
            max_price = max(prices)
            change = last_price - first_price
            change_pct = (change / first_price) * 100
            volatility = max_price - min_price
            
            # Extract input time info from filename
            input_time = filename.replace('prediction_', '').replace('.json', '').split('_')[0]
            
            summary_item = {
                'file': filename,
                'input_period': input_time,
                'prediction_start': metadata['date_range']['start_date'],
                'prediction_end': metadata['date_range']['end_date'],
                'first_price': first_price,
                'last_price': last_price,
                'min_price': min_price,
                'max_price': max_price,
                'change_pct': change_pct,
                'volatility': volatility,
                'trend': '📈 Bullish' if change_pct > 0.5 else '📉 Bearish' if change_pct < -0.5 else '➡️ Neutral'
            }
            
            summary_data.append(summary_item)
            
            # Print summary
            print(f"\n📄 {input_time}")
            print(f"   🕐 Prediction Period: {metadata['date_range']['start_date']} → {metadata['date_range']['end_date']}")
            print(f"   💰 Price Movement: ${first_price:.4f} → ${last_price:.4f} ({change_pct:+.2f}%)")
            print(f"   📊 Price Range: ${min_price:.4f} - ${max_price:.4f}")
            print(f"   📈 Trend: {summary_item['trend']}")
            print(f"   📉 Volatility: ${volatility:.4f}")
            
        except Exception as e:
            print(f"❌ Error processing {pred_file}: {str(e)}")
    
    # Overall statistics
    if summary_data:
        print("\n" + "=" * 80)
        print("📊 OVERALL ANALYSIS")
        print("=" * 80)
        
        all_changes = [item['change_pct'] for item in summary_data]
        all_volatilities = [item['volatility'] for item in summary_data]
        
        bullish_count = len([x for x in all_changes if x > 0.5])
        bearish_count = len([x for x in all_changes if x < -0.5])
        neutral_count = len(all_changes) - bullish_count - bearish_count
        
        avg_change = sum(all_changes) / len(all_changes)
        avg_volatility = sum(all_volatilities) / len(all_volatilities)
        
        print(f"📈 Bullish predictions: {bullish_count}")
        print(f"📉 Bearish predictions: {bearish_count}")
        print(f"➡️ Neutral predictions: {neutral_count}")
        print(f"📊 Average predicted change: {avg_change:+.2f}%")
        print(f"📉 Average volatility: ${avg_volatility:.4f}")
        
        # Most volatile prediction
        most_volatile = max(summary_data, key=lambda x: x['volatility'])
        print(f"🌪️ Most volatile prediction: {most_volatile['input_period']} (${most_volatile['volatility']:.4f})")
        
        # Largest predicted move
        largest_move = max(summary_data, key=lambda x: abs(x['change_pct']))
        direction = "upward" if largest_move['change_pct'] > 0 else "downward"
        print(f"🎯 Largest predicted move: {largest_move['input_period']} ({largest_move['change_pct']:+.2f}% {direction})")
    
    print(f"\n💾 All prediction files saved in: {output_folder}/")
    print("🎉 Summary complete!")

if __name__ == "__main__":
    generate_clean_summary()
