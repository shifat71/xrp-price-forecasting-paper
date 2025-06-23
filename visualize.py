#!/usr/bin/env python3
"""
Visualization Script for XRP Price Predictions
===============================================

This script creates visualizations of the prediction results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def visualize_predictions():
    """
    Create visualizations of the prediction results
    """
    print("📊 Creating XRP Price Prediction Visualizations")
    print("=" * 50)
    
    # Load prediction results
    try:
        with open("prediction_output_simple.json", 'r') as f:
            prediction_data = json.load(f)
        print("✅ Loaded prediction data")
    except FileNotFoundError:
        print("❌ Prediction file not found. Please run the model first.")
        return
    
    # Load recent historical data for comparison
    try:
        with open("dataset/XRPUSDT_market_data_21June_to_22June_2025.json", 'r') as f:
            historical_data = json.load(f)
        print("✅ Loaded historical data")
    except FileNotFoundError:
        print("❌ Historical data file not found.")
        return
    
    # Convert to DataFrames
    df_pred = pd.DataFrame(prediction_data['market_data'])
    df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp_ms'], unit='ms')
    
    df_hist = pd.DataFrame(historical_data['market_data'])
    df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp_ms'], unit='ms')
    df_hist = df_hist.sort_values('timestamp').reset_index(drop=True)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Historical vs Predicted Prices
    # Show last 60 historical points + predictions
    hist_subset = df_hist.tail(60)
    
    axes[0, 0].plot(hist_subset['timestamp'], hist_subset['close'], 
                   label='Historical Prices', color='blue', linewidth=2)
    axes[0, 0].plot(df_pred['timestamp'], df_pred['close'], 
                   label='Predicted Prices', color='red', linewidth=2, linestyle='--')
    
    # Add vertical line to separate historical and predicted
    axes[0, 0].axvline(x=hist_subset['timestamp'].iloc[-1], 
                      color='green', linestyle='-', alpha=0.7, label='Prediction Start')
    
    axes[0, 0].set_title('XRP Price: Historical vs Predicted', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Price (USDT)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Prediction Details
    predicted_prices = df_pred['close'].astype(float)
    time_steps = range(len(predicted_prices))
    
    axes[0, 1].plot(time_steps, predicted_prices, 'ro-', linewidth=2, markersize=6)
    axes[0, 1].set_title('Predicted Price Progression', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Time Step (3-min intervals)')
    axes[0, 1].set_ylabel('Predicted Price (USDT)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(time_steps, predicted_prices, 1)
    p = np.poly1d(z)
    axes[0, 1].plot(time_steps, p(time_steps), "r--", alpha=0.7, label='Trend Line')
    axes[0, 1].legend()
    
    # 3. Price Change Analysis
    price_changes = predicted_prices.diff().dropna()
    
    colors = ['red' if x < 0 else 'green' for x in price_changes]
    bars = axes[1, 0].bar(range(len(price_changes)), price_changes, color=colors, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].set_title('Predicted Price Changes', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Price Change (USDT)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Price Distribution and Statistics
    axes[1, 1].hist(predicted_prices, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(predicted_prices.mean(), color='red', linestyle='--', 
                      label=f'Mean: ${predicted_prices.mean():.4f}')
    axes[1, 1].axvline(predicted_prices.median(), color='green', linestyle='--', 
                      label=f'Median: ${predicted_prices.median():.4f}')
    axes[1, 1].set_title('Distribution of Predicted Prices', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Price (USDT)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xrp_prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print("\n📈 PREDICTION ANALYSIS SUMMARY")
    print("=" * 50)
    
    # Basic statistics
    print(f"Prediction Time Range:")
    print(f"  From: {df_pred['timestamp'].min()}")
    print(f"  To:   {df_pred['timestamp'].max()}")
    
    print(f"\nPrice Statistics:")
    print(f"  Starting Price: ${predicted_prices.iloc[0]:.4f}")
    print(f"  Ending Price:   ${predicted_prices.iloc[-1]:.4f}")
    print(f"  Minimum Price:  ${predicted_prices.min():.4f}")
    print(f"  Maximum Price:  ${predicted_prices.max():.4f}")
    print(f"  Average Price:  ${predicted_prices.mean():.4f}")
    print(f"  Price Range:    ${predicted_prices.max() - predicted_prices.min():.4f}")
    
    # Trend analysis
    total_change = predicted_prices.iloc[-1] - predicted_prices.iloc[0]
    total_change_pct = (total_change / predicted_prices.iloc[0]) * 100
    
    print(f"\nTrend Analysis:")
    print(f"  Total Change:     ${total_change:+.4f}")
    print(f"  Percentage Change: {total_change_pct:+.2f}%")
    
    trend_direction = "Bullish 📈" if total_change > 0 else "Bearish 📉" if total_change < 0 else "Neutral ➡️"
    print(f"  Overall Trend:    {trend_direction}")
    
    # Volatility analysis
    volatility = predicted_prices.std()
    print(f"\nVolatility Analysis:")
    print(f"  Standard Deviation: ${volatility:.4f}")
    print(f"  Coefficient of Variation: {(volatility/predicted_prices.mean())*100:.2f}%")
    
    # Movement analysis
    positive_moves = len([x for x in price_changes if x > 0])
    negative_moves = len([x for x in price_changes if x < 0])
    
    print(f"\nMovement Analysis:")
    print(f"  Positive Price Moves: {positive_moves}/{len(price_changes)} ({positive_moves/len(price_changes)*100:.1f}%)")
    print(f"  Negative Price Moves: {negative_moves}/{len(price_changes)} ({negative_moves/len(price_changes)*100:.1f}%)")
    print(f"  Largest Positive Move: ${price_changes.max():.4f}")
    print(f"  Largest Negative Move: ${price_changes.min():.4f}")
    
    print(f"\n✅ Visualizations saved as 'xrp_prediction_analysis.png'")
    print("🎯 Analysis complete!")

if __name__ == "__main__":
    visualize_predictions()
