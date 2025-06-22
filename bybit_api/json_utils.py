import json
import os
from datetime import datetime
import pandas as pd


def load_market_data_from_json(json_file_path):
    """
    Load market data from JSON file back into a DataFrame
    
    Args:
        json_file_path (str): Path to the JSON file
    
    Returns:
        pd.DataFrame: Market data as DataFrame
        dict: Metadata from the JSON file
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # Extract metadata
        metadata = data.get('metadata', {})
        
        # Convert market data to DataFrame
        market_data = data.get('market_data', [])
        df = pd.DataFrame(market_data)
        
        # Keep only the numeric columns for analysis
        if not df.empty:
            # Check if we have the new format with timestamp_ms or old format with timestamp
            if 'timestamp_ms' in df.columns:
                df = df[['timestamp_ms', 'datetime', 'date', 'time', 'open', 'high', 'low', 'close']]
                # Rename timestamp_ms to timestamp for consistency
                df = df.rename(columns={'timestamp_ms': 'timestamp'})
            else:
                df = df[['timestamp', 'open', 'high', 'low', 'close']]
        
        return df, metadata
    
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None, None


def list_saved_market_data(dataset_dir="../dataset"):
    """
    List all saved market data JSON files
    
    Args:
        dataset_dir (str): Directory containing JSON files
    
    Returns:
        list: List of JSON file paths
    """
    json_files = []
    
    if os.path.exists(dataset_dir):
        for file in os.listdir(dataset_dir):
            if file.endswith('.json') and 'market_data' in file:
                json_files.append(os.path.join(dataset_dir, file))
    
    return json_files


def display_json_summary(json_file_path):
    """
    Display a summary of the JSON file contents
    
    Args:
        json_file_path (str): Path to the JSON file
    """
    df, metadata = load_market_data_from_json(json_file_path)
    
    if df is not None and metadata is not None:
        print(f"\n=== JSON File Summary ===")
        print(f"File: {os.path.basename(json_file_path)}")
        print(f"Symbol: {metadata.get('symbol', 'N/A')}")
        print(f"Generated: {metadata.get('timestamp_generated', 'N/A')}")
        print(f"Data Count: {metadata.get('data_count', 'N/A')}")
        print(f"Interval: {metadata.get('interval', 'N/A')}")
        
        print(f"\n=== Data Sample ===")
        print(df.head())
        
        print(f"\n=== Data Statistics ===")
        print(df.describe())
    else:
        print("Failed to load JSON data")


if __name__ == "__main__":
    # Example usage
    json_files = list_saved_market_data()
    
    if json_files:
        print("Found market data files:")
        for i, file in enumerate(json_files, 1):
            print(f"{i}. {os.path.basename(file)}")
        
        # Display summary of the first file
        if json_files:
            display_json_summary(json_files[0])
    else:
        print("No market data JSON files found in the dataset directory")
