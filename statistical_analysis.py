#!/usr/bin/env python3
"""
Statistical Analysis for XRP Price Forecasting Research Paper
============================================================

This script performs comprehensive statistical analysis comparing predicted vs actual XRP prices.
Generates visualizations and statistical metrics for the research paper.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

class XRPPredictionAnalyzer:
    def __init__(self):
        self.prediction_data = {}
        self.real_data = {}
        self.comparison_results = {}
        
    def load_all_data(self):
        """Load all prediction and real data files"""
        
        # Load prediction files
        pred_files = glob.glob("dataset/model_output/prediction_*.json")
        for file_path in pred_files:
            filename = os.path.basename(file_path)
            key = filename.replace('prediction_for_', '').replace('.json', '')
            
            with open(file_path, 'r') as f:
                self.prediction_data[key] = json.load(f)
        
        # Load real data files
        real_files = glob.glob("dataset/real_data/*.json")
        for file_path in real_files:
            filename = os.path.basename(file_path)
            key = filename.replace('.json', '')
            
            with open(file_path, 'r') as f:
                self.real_data[key] = json.load(f)
        
        print(f"Loaded {len(self.prediction_data)} prediction files")
        print(f"Loaded {len(self.real_data)} real data files")
        
    def extract_price_series(self, data):
        """Extract price series from market data"""
        return [float(item['close']) for item in data['market_data']]
    
    def calculate_statistics(self):
        """Calculate comprehensive statistics for all predictions"""
        
        all_predictions = []
        all_actuals = []
        individual_stats = []
        
        for key in self.prediction_data.keys():
            if key in self.real_data:
                pred_prices = self.extract_price_series(self.prediction_data[key])
                real_prices = self.extract_price_series(self.real_data[key])
                
                # Ensure same length
                min_len = min(len(pred_prices), len(real_prices))
                pred_prices = pred_prices[:min_len]
                real_prices = real_prices[:min_len]
                
                all_predictions.extend(pred_prices)
                all_actuals.extend(real_prices)
                
                # Calculate individual file statistics
                mae = np.mean(np.abs(np.array(pred_prices) - np.array(real_prices)))
                rmse = np.sqrt(np.mean((np.array(pred_prices) - np.array(real_prices))**2))
                mape = np.mean(np.abs((np.array(real_prices) - np.array(pred_prices)) / np.array(real_prices))) * 100
                
                # Directional accuracy
                pred_changes = np.diff(pred_prices)
                real_changes = np.diff(real_prices)
                directional_accuracy = np.mean(np.sign(pred_changes) == np.sign(real_changes)) * 100
                
                # Correlation
                correlation, p_value = stats.pearsonr(pred_prices, real_prices)
                
                individual_stats.append({
                    'file': key,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'directional_accuracy': directional_accuracy,
                    'correlation': correlation,
                    'p_value': p_value,
                    'n_points': min_len
                })
        
        # Overall statistics
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        overall_stats = {
            'mae': np.mean(np.abs(all_predictions - all_actuals)),
            'rmse': np.sqrt(np.mean((all_predictions - all_actuals)**2)),
            'mape': np.mean(np.abs((all_actuals - all_predictions) / all_actuals)) * 100,
            'correlation': stats.pearsonr(all_predictions, all_actuals)[0],
            'r_squared': stats.pearsonr(all_predictions, all_actuals)[0]**2,
            'mean_prediction': np.mean(all_predictions),
            'mean_actual': np.mean(all_actuals),
            'std_prediction': np.std(all_predictions),
            'std_actual': np.std(all_actuals),
            'total_points': len(all_predictions)
        }
        
        return individual_stats, overall_stats, all_predictions, all_actuals
    
    def create_visualization_plots(self, individual_stats, overall_stats, all_predictions, all_actuals):
        """Create comprehensive visualization plots"""
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Prediction vs Actual Scatter Plot
        ax1 = plt.subplot(2, 3, 1)
        plt.scatter(all_actuals, all_predictions, alpha=0.6, s=20)
        min_val = min(min(all_actuals), min(all_predictions))
        max_val = max(max(all_actuals), max(all_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        plt.xlabel('Actual Prices ($)')
        plt.ylabel('Predicted Prices ($)')
        plt.title(f'Prediction vs Actual Prices\\n(R² = {overall_stats["r_squared"]:.4f})')
        plt.grid(True, alpha=0.3)
        
        # 2. Residuals Plot
        ax2 = plt.subplot(2, 3, 2)
        residuals = all_predictions - all_actuals
        plt.scatter(all_actuals, residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.xlabel('Actual Prices ($)')
        plt.ylabel('Residuals ($)')
        plt.title(f'Residuals Plot\\n(Mean: {np.mean(residuals):.6f})')
        plt.grid(True, alpha=0.3)
        
        # 3. Distribution of Errors
        ax3 = plt.subplot(2, 3, 3)
        plt.hist(residuals, bins=30, alpha=0.7, density=True, edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.8)
        plt.xlabel('Prediction Error ($)')
        plt.ylabel('Density')
        plt.title(f'Error Distribution\\n(Std: {np.std(residuals):.6f})')
        plt.grid(True, alpha=0.3)
        
        # 4. MAE by File
        ax4 = plt.subplot(2, 3, 4)
        files = [stat['file'] for stat in individual_stats]
        maes = [stat['mae'] for stat in individual_stats]
        plt.bar(range(len(files)), maes, alpha=0.7)
        plt.xlabel('Test Period')
        plt.ylabel('MAE ($)')
        plt.title('Mean Absolute Error by Test Period')
        plt.xticks(range(len(files)), [f.split('-')[0] for f in files], rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 5. Correlation by File
        ax5 = plt.subplot(2, 3, 5)
        correlations = [stat['correlation'] for stat in individual_stats]
        plt.bar(range(len(files)), correlations, alpha=0.7, color='orange')
        plt.xlabel('Test Period')
        plt.ylabel('Correlation Coefficient')
        plt.title('Prediction Correlation by Test Period')
        plt.xticks(range(len(files)), [f.split('-')[0] for f in files], rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 1)
        
        # 6. Time Series Example
        ax6 = plt.subplot(2, 3, 6)
        # Use first available file for time series example
        if individual_stats:
            first_key = individual_stats[0]['file']
            if first_key in self.prediction_data and first_key in self.real_data:
                pred_prices = self.extract_price_series(self.prediction_data[first_key])
                real_prices = self.extract_price_series(self.real_data[first_key])
                
                time_points = range(len(pred_prices))
                plt.plot(time_points, real_prices, 'b-', label='Actual', linewidth=2)
                plt.plot(time_points, pred_prices, 'r--', label='Predicted', linewidth=2)
                plt.xlabel('Time Point (3-min intervals)')
                plt.ylabel('Price ($)')
                plt.title(f'Example Time Series: {first_key.split("-")[0]}')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_individual_comparison_plots(self):
        """Create individual comparison plots for each test period"""
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for key in sorted(self.prediction_data.keys()):
            if key in self.real_data and plot_idx < 8:
                pred_prices = self.extract_price_series(self.prediction_data[key])
                real_prices = self.extract_price_series(self.real_data[key])
                
                min_len = min(len(pred_prices), len(real_prices))
                pred_prices = pred_prices[:min_len]
                real_prices = real_prices[:min_len]
                
                time_points = range(len(pred_prices))
                
                axes[plot_idx].plot(time_points, real_prices, 'b-', label='Actual', linewidth=2)
                axes[plot_idx].plot(time_points, pred_prices, 'r--', label='Predicted', linewidth=2)
                axes[plot_idx].set_title(f'{key.split("-")[0]} {key.split("-")[1]}', fontsize=10)
                axes[plot_idx].set_xlabel('Time (3-min intervals)')
                axes[plot_idx].set_ylabel('Price ($)')
                axes[plot_idx].legend(fontsize=8)
                axes[plot_idx].grid(True, alpha=0.3)
                
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 8):
            axes[i].set_visible(False)
        
        plt.suptitle('XRP Price Predictions vs Actual Prices by Test Period', fontsize=16)
        plt.tight_layout()
        plt.savefig('individual_predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_statistics_table(self, individual_stats, overall_stats):
        """Generate detailed statistics table"""
        
        print("\\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\\nOVERALL PERFORMANCE METRICS:")
        print(f"{'Metric':<25} {'Value':<15} {'Description'}")
        print("-" * 60)
        print(f"{'Mean Absolute Error':<25} {overall_stats['mae']:.6f}    {'Average prediction error'}")
        print(f"{'Root Mean Square Error':<25} {overall_stats['rmse']:.6f}   {'Prediction error variance'}")
        print(f"{'Mean Abs Percentage Error':<25} {overall_stats['mape']:.4f}%     {'Relative prediction error'}")
        print(f"{'Correlation Coefficient':<25} {overall_stats['correlation']:.6f}   {'Linear relationship'}")
        print(f"{'R-Squared':<25} {overall_stats['r_squared']:.6f}   {'Explained variance'}")
        print(f"{'Total Data Points':<25} {overall_stats['total_points']:<15} {'Number of predictions'}")
        
        print(f"\\nINDIVIDUAL FILE PERFORMANCE:")
        print(f"{'File':<20} {'MAE':<10} {'RMSE':<10} {'MAPE%':<10} {'Corr':<8} {'Dir%':<8}")
        print("-" * 70)
        
        for stat in individual_stats:
            file_short = stat['file'].split('-')[0] + '-' + stat['file'].split('-')[1]
            print(f"{file_short:<20} {stat['mae']:.6f} {stat['rmse']:.6f} {stat['mape']:.2f}     {stat['correlation']:.4f} {stat['directional_accuracy']:.1f}")
        
        # Calculate summary statistics
        maes = [stat['mae'] for stat in individual_stats]
        correlations = [stat['correlation'] for stat in individual_stats]
        
        print(f"\\nSUMMARY STATISTICS:")
        print(f"{'Mean MAE across files':<25} {np.mean(maes):.6f}")
        print(f"{'Std Dev of MAE':<25} {np.std(maes):.6f}")
        print(f"{'Mean Correlation':<25} {np.mean(correlations):.6f}")
        print(f"{'Std Dev of Correlation':<25} {np.std(correlations):.6f}")
        
        return individual_stats, overall_stats
    
    def run_complete_analysis(self):
        """Run the complete statistical analysis"""
        
        print("Starting comprehensive statistical analysis...")
        
        # Load all data
        self.load_all_data()
        
        # Calculate statistics
        individual_stats, overall_stats, all_predictions, all_actuals = self.calculate_statistics()
        
        # Generate statistics table
        self.generate_statistics_table(individual_stats, overall_stats)
        
        # Create visualizations
        print("\\nGenerating visualization plots...")
        self.create_visualization_plots(individual_stats, overall_stats, all_predictions, all_actuals)
        
        # Create individual comparison plots
        print("Creating individual comparison plots...")
        self.create_individual_comparison_plots()
        
        print("\\nAnalysis complete! Generated files:")
        print("- comprehensive_prediction_analysis.png")
        print("- individual_predictions_comparison.png")
        
        return individual_stats, overall_stats

def main():
    """Main execution function"""
    analyzer = XRPPredictionAnalyzer()
    individual_stats, overall_stats = analyzer.run_complete_analysis()
    
    # Save results to JSON for later use
    results = {
        'individual_stats': individual_stats,
        'overall_stats': overall_stats,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    with open('statistical_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\\nStatistical analysis results saved to 'statistical_analysis_results.json'")

if __name__ == "__main__":
    main()
