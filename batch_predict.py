#!/usr/bin/env python3
"""
Batch Prediction Script for XRP Price Forecasting
=================================================

This script processes all testing data files and generates model predictions
for each input, saving the outputs to the model_output folder.
"""

import os
import glob
import json
from datetime import datetime
from model_simple import XRPPriceForecasterSimple
import warnings
warnings.filterwarnings('ignore')

class BatchPredictor:
    def __init__(self):
        self.forecaster = XRPPriceForecasterSimple(sequence_length=20, prediction_hours=1)
        self.model_path = "xrp_forecaster_simple.pkl"
        self.scaler_path = "scaler_simple.pkl"
        self.testing_data_folder = "dataset/testing_data"
        self.output_folder = "dataset/model_output"
        
        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)
    
    def check_or_train_model(self):
        """
        Check if trained model exists, if not train a new one
        """
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            print("✅ Found existing trained model, loading...")
            self.forecaster.load_model(self.model_path, self.scaler_path)
            return True
        else:
            print("🤖 No trained model found, training new model...")
            print("This may take a few minutes...")
            
            # Train the model using training data
            try:
                self.forecaster.train(
                    dataset_folder="dataset/training_data",
                    validation_split=0.2,
                    n_estimators=100
                )
                
                # Check if model was actually trained
                if self.forecaster.is_trained and self.forecaster.model is not None:
                    # Save the trained model
                    self.forecaster.save_model(self.model_path, self.scaler_path)
                    print("✅ Model trained and saved successfully!")
                    return True
                else:
                    print("❌ Model training failed - model not properly initialized!")
                    return False
                    
            except Exception as e:
                print(f"❌ Model training failed with error: {str(e)}")
                return False
    
    def generate_output_filename(self, input_filename):
        """
        Generate appropriate output filename based on input filename
        
        Args:
            input_filename (str): Original testing data filename
            
        Returns:
            str: Generated output filename
        """
        # Remove .json extension
        base_name = input_filename.replace('.json', '')
        
        # Add prediction prefix and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"prediction_{base_name}_{timestamp}.json"
        
        return output_name
    
    def process_single_file(self, input_file_path, output_filename):
        """
        Process a single testing data file and generate prediction
        
        Args:
            input_file_path (str): Path to the input testing data file
            output_filename (str): Name for the output file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"📊 Processing: {os.path.basename(input_file_path)}")
            
            # Generate output path
            output_path = os.path.join(self.output_folder, output_filename)
            
            # Make prediction
            prediction_result = self.forecaster.predict(
                recent_data_file=input_file_path,
                output_file=output_path
            )
            
            if prediction_result:
                print(f"✅ Prediction saved to: {output_filename}")
                
                # Print some basic stats
                market_data = prediction_result['market_data']
                prices = [float(data['close']) for data in market_data]
                min_price = min(prices)
                max_price = max(prices)
                first_price = prices[0]
                last_price = prices[-1]
                change_pct = ((last_price - first_price) / first_price) * 100
                
                print(f"   📈 Price range: ${min_price:.4f} - ${max_price:.4f}")
                print(f"   📊 Trend: {change_pct:+.2f}%")
                print(f"   ⏰ Predictions: {len(market_data)} data points")
                
                return True
            else:
                print(f"❌ Failed to generate prediction for {os.path.basename(input_file_path)}")
                return False
                
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(input_file_path)}: {str(e)}")
            return False
    
    def process_all_testing_data(self):
        """
        Process all files in the testing_data folder
        """
        print("🔮 XRP PRICE FORECASTING - BATCH PREDICTION")
        print("=" * 60)
        
        # Check/train model first
        if not self.check_or_train_model():
            print("❌ Cannot proceed without a trained model!")
            return
        
        # Get all JSON files from testing_data folder
        testing_files = glob.glob(os.path.join(self.testing_data_folder, "*.json"))
        
        if not testing_files:
            print(f"❌ No JSON files found in {self.testing_data_folder}")
            return
        
        print(f"\n📁 Found {len(testing_files)} testing data files")
        print("=" * 60)
        
        successful_predictions = 0
        failed_predictions = 0
        
        # Process each file
        for input_file in sorted(testing_files):
            input_filename = os.path.basename(input_file)
            output_filename = self.generate_output_filename(input_filename)
            
            if self.process_single_file(input_file, output_filename):
                successful_predictions += 1
            else:
                failed_predictions += 1
            
            print()  # Add spacing between files
        
        # Print summary
        print("=" * 60)
        print("🎉 BATCH PREDICTION COMPLETED!")
        print("=" * 60)
        print(f"✅ Successful predictions: {successful_predictions}")
        print(f"❌ Failed predictions: {failed_predictions}")
        print(f"📁 Output folder: {self.output_folder}")
        
        if successful_predictions > 0:
            print(f"\n📊 Generated prediction files:")
            output_files = glob.glob(os.path.join(self.output_folder, "*.json"))
            for output_file in sorted(output_files):
                print(f"   - {os.path.basename(output_file)}")
    
    def generate_summary_report(self):
        """
        Generate a summary report of all predictions
        """
        output_files = glob.glob(os.path.join(self.output_folder, "prediction_*.json"))
        
        if not output_files:
            print("No prediction files found for summary report")
            return
        
        print(f"\n📊 PREDICTION SUMMARY REPORT")
        print("=" * 60)
        
        summary_data = []
        
        for output_file in sorted(output_files):
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                metadata = data['metadata']
                market_data = data['market_data']
                prices = [float(item['close']) for item in market_data]
                
                # Calculate statistics
                first_price = prices[0]
                last_price = prices[-1]
                min_price = min(prices)
                max_price = max(prices)
                avg_price = sum(prices) / len(prices)
                change_pct = ((last_price - first_price) / first_price) * 100
                
                summary_item = {
                    'file': os.path.basename(output_file),
                    'input_time_range': f"{metadata['input_data']['date_range']['start_date']} to {metadata['input_data']['date_range']['end_date']}",
                    'prediction_time_range': f"{metadata['date_range']['start_date']} to {metadata['date_range']['end_date']}",
                    'first_price': first_price,
                    'last_price': last_price,
                    'min_price': min_price,
                    'max_price': max_price,
                    'avg_price': avg_price,
                    'change_pct': change_pct,
                    'volatility': max_price - min_price
                }
                
                summary_data.append(summary_item)
                
                # Print individual file summary
                print(f"\n📄 {summary_item['file']}")
                print(f"   🕐 Input Data: {summary_item['input_time_range']}")
                print(f"   🔮 Prediction: {summary_item['prediction_time_range']}")
                print(f"   💰 Price: ${first_price:.4f} → ${last_price:.4f} ({change_pct:+.2f}%)")
                print(f"   📊 Range: ${min_price:.4f} - ${max_price:.4f} (volatility: ${summary_item['volatility']:.4f})")
                
            except Exception as e:
                print(f"❌ Error reading {output_file}: {str(e)}")
        
        # Save summary report
        if summary_data:
            summary_file = os.path.join(self.output_folder, "prediction_summary_report.json")
            summary_report = {
                'generated_at': datetime.now().isoformat(),
                'total_predictions': len(summary_data),
                'predictions': summary_data
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_report, f, indent=2)
            
            print(f"\n💾 Summary report saved to: {os.path.basename(summary_file)}")


def main():
    """
    Main function to run batch prediction
    """
    try:
        batch_predictor = BatchPredictor()
        batch_predictor.process_all_testing_data()
        batch_predictor.generate_summary_report()
        
        print(f"\n🎯 All predictions completed!")
        print(f"Check the '{batch_predictor.output_folder}' folder for results.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
