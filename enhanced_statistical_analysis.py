#!/usr/bin/env python3
"""
Enhanced Statistical Analysis for XRP Price Forecasting Research Paper
====================================================================

This script performs comprehensive statistical analysis comparing predicted vs actual XRP prices.
Generates enhanced visualizations with proper date labeling and additional metrics for the research paper.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

class EnhancedXRPPredictionAnalyzer:
    def __init__(self):
        self.prediction_data = {}
        self.real_data = {}
        self.comparison_results = {}
        self.date_mappings = {
            '8-PM-9-PM-23-June-2025': 'June 23, 8-PM-9-PM',
            '10-AM-11-AM-24-June-2025': 'June 24, 10-AM-11-AM',
            '12-PM-1-PM-25-June-2025': 'June 25, 12-PM-1-PM',
            '11-PM-26-June-12-AM-27-June-2025': 'June 26-27, 11-PM-12-AM',
            '6-PM-7-PM-27-June-2025': 'June 27, 6-PM-7-PM',
            '3-AM-4-AM-28-June-2025': 'June 28, 3-AM-4-AM',
            '6-AM-7-AM-29-June-2025': 'June 29, 6-AM-7-AM'
        }
        
    def extract_date_label(self, filename):
        """Extract proper date and time label from filename"""
        # Remove path and extension
        base_name = os.path.basename(filename).replace('prediction_for_', '').replace('.json', '')
        
        # Use predefined mappings for consistent labeling
        if base_name in self.date_mappings:
            return self.date_mappings[base_name]
        
        return base_name
        
    def load_all_data(self):
        """Load all prediction and real data files"""
        
        # Load prediction files
        pred_files = glob.glob("dataset/model_output/prediction_*.json")
        for file_path in pred_files:
            filename = os.path.basename(file_path)
            key = filename.replace('prediction_for_', '').replace('.json', '')
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.prediction_data[key] = data
        
        # Load real data files
        real_files = glob.glob("dataset/real_data/*.json")
        for file_path in real_files:
            filename = os.path.basename(file_path)
            key = filename.replace('.json', '')
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.real_data[key] = data
        
        print(f"Loaded {len(self.prediction_data)} prediction files")
        print(f"Loaded {len(self.real_data)} real data files")
        
    def calculate_metrics_for_period(self, pred_data, real_data):
        """Calculate comprehensive metrics for a single time period"""
        
        # Extract price arrays - both files use 'market_data' key
        pred_prices = np.array([item['close'] for item in pred_data['market_data']])
        real_prices = np.array([item['close'] for item in real_data['market_data']])
        
        # Ensure same length
        min_len = min(len(pred_prices), len(real_prices))
        pred_prices = pred_prices[:min_len]
        real_prices = real_prices[:min_len]
        
        # Calculate metrics
        mae = mean_absolute_error(real_prices, pred_prices)
        mse = mean_squared_error(real_prices, pred_prices)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((real_prices - pred_prices) / real_prices)) * 100
        
        # Correlation
        correlation = np.corrcoef(pred_prices, real_prices)[0, 1]
        
        # R-squared
        r2 = r2_score(real_prices, pred_prices)
        
        # Directional accuracy
        real_direction = np.diff(real_prices) > 0
        pred_direction = np.diff(pred_prices) > 0
        directional_accuracy = np.mean(real_direction == pred_direction) * 100
        
        # Price statistics
        real_mean = np.mean(real_prices)
        pred_mean = np.mean(pred_prices)
        real_std = np.std(real_prices)
        pred_std = np.std(pred_prices)
        
        # Residuals
        residuals = pred_prices - real_prices
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'correlation': correlation,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'real_mean': real_mean,
            'pred_mean': pred_mean,
            'real_std': real_std,
            'pred_std': pred_std,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'pred_prices': pred_prices,
            'real_prices': real_prices,
            'residuals': residuals
        }
    
    def analyze_all_periods(self):
        """Analyze all available time periods"""
        
        for key in self.prediction_data.keys():
            if key in self.real_data:
                print(f"Analyzing period: {key}")
                
                pred_data = self.prediction_data[key]
                real_data = self.real_data[key]
                
                metrics = self.calculate_metrics_for_period(pred_data, real_data)
                metrics['period_key'] = key
                metrics['date_label'] = self.extract_date_label(f"prediction_for_{key}.json")
                
                self.comparison_results[key] = metrics
        
        print(f"Analysis completed for {len(self.comparison_results)} periods")
    
    def create_enhanced_comprehensive_analysis(self):
        """Create enhanced comprehensive analysis with proper date labels"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # Collect all data for analysis
        all_pred = []
        all_real = []
        all_residuals = []
        period_labels = []
        mae_values = []
        mape_values = []
        correlation_values = []
        r2_values = []
        directional_acc = []
        
        for key, metrics in self.comparison_results.items():
            all_pred.extend(metrics['pred_prices'])
            all_real.extend(metrics['real_prices'])
            all_residuals.extend(metrics['residuals'])
            
            period_labels.append(metrics['date_label'])
            mae_values.append(metrics['mae'])
            mape_values.append(metrics['mape'])
            correlation_values.append(metrics['correlation'])
            r2_values.append(metrics['r2'])
            directional_acc.append(metrics['directional_accuracy'])
        
        # Convert to numpy arrays
        all_pred = np.array(all_pred)
        all_real = np.array(all_real)
        all_residuals = np.array(all_residuals)
        
        # 1. Prediction vs Actual Scatter Plot
        plt.subplot(3, 3, 1)
        plt.scatter(all_real, all_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_price = min(np.min(all_real), np.min(all_pred))
        max_price = max(np.max(all_real), np.max(all_pred))
        plt.plot([min_price, max_price], [min_price, max_price], 'r--', label='Perfect Prediction')
        
        # Regression line
        z = np.polyfit(all_real, all_pred, 1)
        p = np.poly1d(z)
        plt.plot(all_real, p(all_real), 'b-', alpha=0.8, label=f'Trend Line')
        
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title('Predicted vs Actual Prices\n(All Test Periods)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² annotation
        r2_overall = r2_score(all_real, all_pred)
        plt.text(0.05, 0.95, f'R² = {r2_overall:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residuals Plot
        plt.subplot(3, 3, 2)
        plt.scatter(all_real, all_residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Residuals (Predicted - Actual)')
        plt.title('Residuals vs Actual Prices')
        plt.grid(True, alpha=0.3)
        
        # 3. Histogram of Residuals
        plt.subplot(3, 3, 3)
        plt.hist(all_residuals, bins=30, alpha=0.7, density=True)
        
        # Normal distribution overlay
        mu, sigma = stats.norm.fit(all_residuals)
        x = np.linspace(all_residuals.min(), all_residuals.max(), 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')
        
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title('Distribution of Prediction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. MAE by Test Period
        plt.subplot(3, 3, 4)
        bars = plt.bar(range(len(period_labels)), mae_values, color='skyblue', alpha=0.8)
        plt.xlabel('Test Period')
        plt.ylabel('Mean Absolute Error')
        plt.title('MAE by Test Period')
        plt.xticks(range(len(period_labels)), period_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 5. MAPE by Test Period
        plt.subplot(3, 3, 5)
        bars = plt.bar(range(len(period_labels)), mape_values, color='lightcoral', alpha=0.8)
        plt.xlabel('Test Period')
        plt.ylabel('Mean Absolute Percentage Error (%)')
        plt.title('MAPE by Test Period')
        plt.xticks(range(len(period_labels)), period_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
        
        # 6. Correlation by Test Period
        plt.subplot(3, 3, 6)
        bars = plt.bar(range(len(period_labels)), correlation_values, color='lightgreen', alpha=0.8)
        plt.xlabel('Test Period')
        plt.ylabel('Correlation Coefficient')
        plt.title('Correlation by Test Period')
        plt.xticks(range(len(period_labels)), period_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.05),
                    f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        # 7. Directional Accuracy by Test Period
        plt.subplot(3, 3, 7)
        bars = plt.bar(range(len(period_labels)), directional_acc, color='gold', alpha=0.8)
        plt.xlabel('Test Period')
        plt.ylabel('Directional Accuracy (%)')
        plt.title('Directional Accuracy by Test Period')
        plt.xticks(range(len(period_labels)), period_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random Chance')
        plt.legend()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 8. Time Series Example (Best Performance Period)
        plt.subplot(3, 3, 8)
        best_period_key = min(self.comparison_results.keys(), 
                             key=lambda k: self.comparison_results[k]['mae'])
        best_metrics = self.comparison_results[best_period_key]
        
        time_steps = range(len(best_metrics['real_prices']))
        plt.plot(time_steps, best_metrics['real_prices'], 'b-', linewidth=2, label='Actual', alpha=0.8)
        plt.plot(time_steps, best_metrics['pred_prices'], 'r--', linewidth=2, label='Predicted', alpha=0.8)
        plt.xlabel('Time Steps (3-min intervals)')
        plt.ylabel('Price ($)')
        plt.title(f'Best Performance: {best_metrics["date_label"]}\n(MAE: {best_metrics["mae"]:.6f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Model Performance Summary
        plt.subplot(3, 3, 9)
        plt.axis('off')
        
        # Calculate overall statistics
        overall_mae = np.mean(mae_values)
        overall_mape = np.mean(mape_values)
        overall_correlation = np.corrcoef(all_real, all_pred)[0, 1]
        overall_r2 = r2_score(all_real, all_pred)
        overall_directional = np.mean(directional_acc)
        
        summary_text = f"""
        Overall Performance Summary
        ============================
        
        Mean Absolute Error: {overall_mae:.6f}
        Mean Absolute % Error: {overall_mape:.2f}%
        Correlation: {overall_correlation:.4f}
        R² Score: {overall_r2:.4f}
        Directional Accuracy: {overall_directional:.1f}%
        
        Test Periods: {len(self.comparison_results)}
        Total Predictions: {len(all_pred):,}
        
        Best Period: {best_metrics['date_label']}
        Best MAE: {min(mae_values):.6f}
        
        Date Range: June 23-29, 2025
        """
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('enhanced_comprehensive_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Enhanced comprehensive analysis saved as 'enhanced_comprehensive_prediction_analysis.png'")
    
    def create_individual_period_analysis(self):
        """Create detailed analysis for each individual test period with proper date labels"""
        
        n_periods = len(self.comparison_results)
        n_cols = 2
        n_rows = (n_periods + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
        if n_periods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (key, metrics) in enumerate(self.comparison_results.items()):
            ax = axes[idx]
            
            time_steps = range(len(metrics['real_prices']))
            ax.plot(time_steps, metrics['real_prices'], 'b-', linewidth=2, label='Actual', alpha=0.8)
            ax.plot(time_steps, metrics['pred_prices'], 'r--', linewidth=2, label='Predicted', alpha=0.8)
            
            ax.set_xlabel('Time Steps (3-min intervals)')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'{metrics["date_label"]}\nMAE: {metrics["mae"]:.6f}, MAPE: {metrics["mape"]:.2f}%, Corr: {metrics["correlation"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add performance metrics as text box
            textstr = f'R²: {metrics["r2"]:.3f}\nDir. Acc: {metrics["directional_accuracy"]:.1f}%'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
        
        # Hide unused subplots
        for idx in range(n_periods, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('enhanced_individual_predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Enhanced individual period analysis saved as 'enhanced_individual_predictions_comparison.png'")
    
    def create_random_forest_architecture_diagram(self):
        """Create a visual representation of the Random Forest architecture"""
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Input layer
        input_rect = plt.Rectangle((1, 10), 3, 1.5, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(input_rect)
        ax.text(2.5, 10.75, 'Input Layer\n500 Features\n(20 timesteps × 25 features\nflattened)', 
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Feature engineering box
        feature_rect = plt.Rectangle((1, 8), 3, 1.5, facecolor='lightyellow', edgecolor='black', linewidth=2)
        ax.add_patch(feature_rect)
        ax.text(2.5, 8.75, 'Feature Engineering\n• OHLC prices\n• Technical indicators\n• Temporal features', 
               ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Random Forest Trees
        tree_positions = [(6, 9), (8, 9), (10, 9), (12, 9), (14, 9)]
        tree_colors = ['lightgreen', 'lightcoral', 'lightpink', 'lightsalmon', 'lightsteelblue']
        
        for i, (x, y) in enumerate(tree_positions):
            # Tree rectangle
            tree_rect = plt.Rectangle((x-0.7, y-0.7), 1.4, 1.4, 
                                    facecolor=tree_colors[i], edgecolor='black', linewidth=1.5)
            ax.add_patch(tree_rect)
            ax.text(x, y, f'Tree\n{i+1}', ha='center', va='center', fontsize=9, fontweight='bold')
            
            # Arrow from input to tree
            ax.arrow(4.2, 9.5, x-5, y-9.2, head_width=0.15, head_length=0.2, fc='gray', ec='gray')
        
        # Add "..." to indicate more trees
        ax.text(15, 9, '...', ha='center', va='center', fontsize=20, fontweight='bold')
        ax.text(15, 8.3, '100 Trees\nTotal', ha='center', va='center', fontsize=8, fontweight='bold')
        
        # MultiOutput wrapper
        wrapper_rect = plt.Rectangle((6, 6), 8, 1, facecolor='wheat', edgecolor='black', linewidth=2)
        ax.add_patch(wrapper_rect)
        ax.text(10, 6.5, 'MultiOutputRegressor Wrapper\n(Handles 20 simultaneous outputs)', 
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Aggregation
        agg_rect = plt.Rectangle((8, 4), 4, 1, facecolor='lightgray', edgecolor='black', linewidth=2)
        ax.add_patch(agg_rect)
        ax.text(10, 4.5, 'Ensemble Averaging\n(Mean of all trees)', 
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Output layer
        output_rect = plt.Rectangle((8, 2), 4, 1, facecolor='lightpink', edgecolor='black', linewidth=2)
        ax.add_patch(output_rect)
        ax.text(10, 2.5, 'Output Layer\n20 Price Predictions\n(Next Hour: 3-min intervals)', 
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrows
        # Trees to wrapper
        for x, y in tree_positions:
            ax.arrow(x, y-0.7, 0, -1.6, head_width=0.1, head_length=0.15, fc='black', ec='black')
        
        # Wrapper to aggregation
        ax.arrow(10, 6, 0, -0.8, head_width=0.15, head_length=0.15, fc='black', ec='black')
        
        # Aggregation to output
        ax.arrow(10, 4, 0, -0.8, head_width=0.15, head_length=0.15, fc='black', ec='black')
        
        # Title
        ax.text(8, 11.5, 'Random Forest Regression Architecture\nfor XRP Price Forecasting', 
               ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Model specifications
        spec_text = """
        Model Specifications:
        =====================
        • Algorithm: Random Forest Regressor
        • Number of Trees: 100
        • Input Features: 500 (flattened)
        • Output Predictions: 20 prices
        • Sequence Length: 20 timesteps
        • Feature Engineering: 25 indicators
        • Multi-output: Yes (simultaneous)
        • Prediction Horizon: 1 hour
        • Time Resolution: 3 minutes
        • Training Method: Ensemble learning
        • Regularization: Bootstrap sampling
        """
        
        ax.text(0.5, 5.5, spec_text, ha='left', va='top', fontsize=9, fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('random_forest_architecture_diagram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Random Forest architecture diagram saved as 'random_forest_architecture_diagram.png'")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive statistical report"""
        
        print("\n" + "="*80)
        print("ENHANCED XRP PRICE FORECASTING - STATISTICAL ANALYSIS REPORT")
        print("="*80)
        
        # Overall statistics
        all_mae = [metrics['mae'] for metrics in self.comparison_results.values()]
        all_mape = [metrics['mape'] for metrics in self.comparison_results.values()]
        all_corr = [metrics['correlation'] for metrics in self.comparison_results.values()]
        all_r2 = [metrics['r2'] for metrics in self.comparison_results.values()]
        all_dir_acc = [metrics['directional_accuracy'] for metrics in self.comparison_results.values()]
        
        print(f"\nOVERALL PERFORMANCE METRICS:")
        print(f"Mean MAE: {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
        print(f"Mean MAPE: {np.mean(all_mape):.3f}% ± {np.std(all_mape):.3f}%")
        print(f"Mean Correlation: {np.mean(all_corr):.4f} ± {np.std(all_corr):.4f}")
        print(f"Mean R²: {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
        print(f"Mean Directional Accuracy: {np.mean(all_dir_acc):.1f}% ± {np.std(all_dir_acc):.1f}%")
        
        print(f"\nPERFORMANCE BY TEST PERIOD:")
        print("-" * 120)
        print(f"{'Date & Time':<30} {'MAE':<12} {'RMSE':<12} {'MAPE (%)':<12} {'Correlation':<12} {'Dir. Acc (%)':<12}")
        print("-" * 120)
        
        for key, metrics in self.comparison_results.items():
            print(f"{metrics['date_label']:<30} "
                  f"{metrics['mae']:<12.6f} "
                  f"{metrics['rmse']:<12.6f} "
                  f"{metrics['mape']:<12.2f} "
                  f"{metrics['correlation']:<12.4f} "
                  f"{metrics['directional_accuracy']:<12.1f}")
        
        # Best and worst performing periods
        best_mae_period = min(self.comparison_results.items(), key=lambda x: x[1]['mae'])
        worst_mae_period = max(self.comparison_results.items(), key=lambda x: x[1]['mae'])
        
        print(f"\nBEST PERFORMING PERIOD:")
        print(f"Period: {best_mae_period[1]['date_label']}")
        print(f"MAE: {best_mae_period[1]['mae']:.6f}")
        print(f"MAPE: {best_mae_period[1]['mape']:.2f}%")
        
        print(f"\nWORST PERFORMING PERIOD:")
        print(f"Period: {worst_mae_period[1]['date_label']}")
        print(f"MAE: {worst_mae_period[1]['mae']:.6f}")
        print(f"MAPE: {worst_mae_period[1]['mape']:.2f}%")
        
        print("\n" + "="*80)

def main():
    """Main execution function"""
    print("Starting Enhanced XRP Price Forecasting Statistical Analysis...")
    
    # Create analyzer instance
    analyzer = EnhancedXRPPredictionAnalyzer()
    
    # Load all data
    analyzer.load_all_data()
    
    # Perform analysis
    analyzer.analyze_all_periods()
    
    # Generate visualizations
    print("\nGenerating enhanced visualizations...")
    analyzer.create_enhanced_comprehensive_analysis()
    analyzer.create_individual_period_analysis()
    analyzer.create_random_forest_architecture_diagram()
    
    # Generate report
    analyzer.generate_comprehensive_report()
    
    print("\nEnhanced analysis completed successfully!")
    print("Generated files:")
    print("- enhanced_comprehensive_prediction_analysis.png")
    print("- enhanced_individual_predictions_comparison.png")
    print("- random_forest_architecture_diagram.png")

if __name__ == "__main__":
    main()
