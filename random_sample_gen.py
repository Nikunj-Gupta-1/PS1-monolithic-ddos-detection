import pandas as pd
import numpy as np
import os


def fetch_random_records_with_more_benign(file_path, num_records=100, benign_percentage=30, output_file=__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '../raw data collected for project/random_sample.csv'))):
    """
    Modified version: Fetch random records with a specified percentage of BENIGN records
    
    Args:
        file_path: Path to the CSV file
        num_records: Total number of records to fetch
        benign_percentage: Percentage of BENIGN records desired (default: 30%)
        output_file: Output file name
    """
    
    print(f"🔄 Fetching {num_records} random records from {file_path}")
    print(f"🎯 Target: {benign_percentage}% BENIGN records")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found!")
        return None
    
    try:
        # First, let's read the file in chunks to find BENIGN and non-BENIGN rows
        print("📊 Analyzing dataset for BENIGN records...")
        
        benign_indices = []
        non_benign_indices = []
        row_count = 0
        
        # Read file in chunks to handle large files efficiently
        chunk_size = 10000
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk_start_idx = row_count
            
            # Find BENIGN records in this chunk
            benign_mask = chunk[' Label'] == 'BENIGN'
            chunk_benign_indices = chunk.index[benign_mask].tolist()
            chunk_non_benign_indices = chunk.index[~benign_mask].tolist()
            
            # Adjust indices to global file position
            benign_indices.extend([idx + chunk_start_idx for idx in chunk_benign_indices])
            non_benign_indices.extend([idx + chunk_start_idx for idx in chunk_non_benign_indices])
            
            row_count += len(chunk)
            
            if row_count % 50000 == 0:
                print(f"   Processed {row_count:,} records...")
        
        total_benign = len(benign_indices)
        total_non_benign = len(non_benign_indices)
        total_records = total_benign + total_non_benign
        
        print(f"   Total records: {total_records:,}")
        print(f"   BENIGN records: {total_benign:,} ({total_benign/total_records*100:.1f}%)")
        print(f"   Non-BENIGN records: {total_non_benign:,} ({total_non_benign/total_records*100:.1f}%)")
        
        # Calculate how many BENIGN and non-BENIGN records we need
        target_benign_count = int(num_records * benign_percentage / 100)
        target_non_benign_count = num_records - target_benign_count
        
        print(f"\n🎯 Target distribution:")
        print(f"   BENIGN: {target_benign_count} records")
        print(f"   Non-BENIGN: {target_non_benign_count} records")
        
        # Check if we have enough BENIGN records
        if target_benign_count > total_benign:
            print(f"⚠️  Warning: Requested {target_benign_count} BENIGN records, but only {total_benign} available")
            target_benign_count = total_benign
            target_non_benign_count = num_records - target_benign_count
            print(f"   Adjusted to: {target_benign_count} BENIGN, {target_non_benign_count} non-BENIGN")
        
        # Sample indices
        print("🎲 Sampling records...")
        selected_benign_indices = np.random.choice(benign_indices, size=target_benign_count, replace=False)
        selected_non_benign_indices = np.random.choice(non_benign_indices, size=target_non_benign_count, replace=False)
        
        # Combine all selected indices
        all_selected_indices = np.concatenate([selected_benign_indices, selected_non_benign_indices])
        
        # Create skip_rows set (convert to 1-based indexing for pandas)
        all_rows = set(range(1, total_records + 1))
        rows_to_keep = set(all_selected_indices + 1)  # Convert to 1-based
        skip_rows = all_rows - rows_to_keep
        
        print("📖 Reading selected records...")
        df_sample = pd.read_csv(file_path, skiprows=list(skip_rows))
        
        # Shuffle the final dataset
        df_sample = df_sample.sample(frac=1).reset_index(drop=True)
        
        print(f"✅ Successfully fetched {len(df_sample)} records")
        
        # Display sample information
        print(f"\n📈 Sample Overview:")
        print(f"   Shape: {df_sample.shape}")
        
        if ' Label' in df_sample.columns:
            label_counts = df_sample[' Label'].value_counts()
            print(f"   Label distribution:")
            for label, count in label_counts.items():
                percentage = (count / len(df_sample)) * 100
                print(f"     {label}: {count} records ({percentage:.1f}%)")
        
        # Save to file
        df_sample.to_csv(output_file, index=False)
        print(f"💾 Random sample saved to: {output_file}")
        
        return df_sample
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None


def fetch_random_records_simple_benign_boost(file_path, num_records=100, output_file=__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '../raw data collected for project/random_sample.csv'))):
    """
    Simpler approach: Read file once and oversample BENIGN records
    """
    print(f"🔄 Fetching {num_records} random records with BENIGN boost from {file_path}")
    
    try:
        # Read the entire dataset (if memory allows)
        print("📖 Reading dataset...")
        df = pd.read_csv(file_path)
        
        print(f"📊 Dataset shape: {df.shape}")
        
        # Separate BENIGN and non-BENIGN records
        benign_df = df[df[' Label'] == 'BENIGN']
        non_benign_df = df[df[' Label'] != 'BENIGN']
        
        print(f"   BENIGN records: {len(benign_df):,}")
        print(f"   Non-BENIGN records: {len(non_benign_df):,}")
        
        # Sample with higher BENIGN representation (e.g., 40% BENIGN)
        target_benign = min(int(num_records * 0.4), len(benign_df))
        target_non_benign = num_records - target_benign
        
        print(f"🎯 Sampling {target_benign} BENIGN and {target_non_benign} non-BENIGN records")
        
        # Sample from each category
        sampled_benign = benign_df.sample(n=target_benign, random_state=42)
        sampled_non_benign = non_benign_df.sample(n=target_non_benign, random_state=42)
        
        # Combine and shuffle
        df_sample = pd.concat([sampled_benign, sampled_non_benign]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Successfully created sample with {len(df_sample)} records")
        
        # Display sample information
        print(f"\n📈 Sample Overview:")
        print(f"   Shape: {df_sample.shape}")
        
        label_counts = df_sample[' Label'].value_counts()
        print(f"   Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(df_sample)) * 100
            print(f"     {label}: {count} records ({percentage:.1f}%)")
        
        # Save to file
        df_sample.to_csv(output_file, index=False)
        print(f"💾 Sample saved to: {output_file}")
        
        return df_sample
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# Usage example
if __name__ == "__main__":
    # Replace with your actual file path
    original_file = __import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '../raw data collected for project/CIC_DDoS2019_balanced_reduced.csv'))
    
    print("Choose sampling method:")
    print("1. Memory-efficient method (for very large files)")
    print("2. Simple method (loads entire file into memory)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Method 1: Memory-efficient with custom BENIGN percentage
        benign_pct = int(input("Enter desired BENIGN percentage (default 30): ") or "30")
        sample_df = fetch_random_records_with_more_benign(
            original_file, 
            num_records=100, 
            benign_percentage=benign_pct
        )
    else:
        # Method 2: Simple approach (40% BENIGN by default)
        sample_df = fetch_random_records_simple_benign_boost(original_file, num_records=100)
    
    if sample_df is not None:
        print("\n🎉 Sampling completed successfully!")
        print("First 5 records:")
        print(sample_df.head())
