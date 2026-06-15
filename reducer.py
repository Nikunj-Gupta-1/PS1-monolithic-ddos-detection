import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_balanced_ddos_dataset():
    """
    Complete script to create a balanced, reduced CIC-DDoS2019 dataset
    using the correct column names with leading spaces
    """
    
    print("=== CIC-DDoS2019 Balanced Dataset Creator (Column-Fixed) ===")
    
    # Configuration parameters - using EXACT column names from your dataset
    required_columns = [
        ' Flow Duration',           # Note the leading space
        ' Total Fwd Packets',
        ' Total Backward Packets',
        'Total Length of Fwd Packets',  # This one doesn't have leading space
        ' Total Length of Bwd Packets',
        ' Fwd Packet Length Max',
        ' Fwd Packet Length Min',
        ' Fwd Packet Length Mean',
        ' Fwd Packet Length Std',
        'Bwd Packet Length Max',    # This one doesn't have leading space
        ' Bwd Packet Length Min',
        ' Bwd Packet Length Mean',
        ' Bwd Packet Length Std',
        'Flow Bytes/s',             # This one doesn't have leading space
        ' Flow Packets/s',
        ' Flow IAT Mean',
        ' Flow IAT Std',
        ' Flow IAT Max',
        ' Flow IAT Min',
        ' Fwd IAT Mean',
        ' Fwd IAT Std',
        ' Fwd IAT Max',
        ' Fwd IAT Min',
        ' Bwd IAT Mean',
        ' Bwd IAT Std',
        ' Bwd IAT Max',
        ' Bwd IAT Min',
        'Fwd IAT Total',            # This one doesn't have leading space
        'Bwd IAT Total',            # This one doesn't have leading space
        'Fwd PSH Flags',            # This one doesn't have leading space
        ' Bwd PSH Flags',
        ' ACK Flag Count',
        ' SYN Flag Count',
        'FIN Flag Count',           # This one doesn't have leading space
        ' RST Flag Count',
        ' URG Flag Count',
        ' PSH Flag Count',
        ' Packet Length Mean',
        ' Packet Length Std',
        ' Packet Length Variance',
        ' Down/Up Ratio',
        ' Average Packet Size',
        ' Avg Fwd Segment Size',
        ' Avg Bwd Segment Size',
        'Fwd Packets/s',            # This one doesn't have leading space
        ' Bwd Packets/s',
        ' Max Packet Length',
        ' Min Packet Length',
        ' Protocol',
        ' Label'                    # Target variable
    ]
    
    # File paths
    input_file = 'CIC_DDoS2019_combined.csv'
    output_file = 'CIC_DDoS2019_balanced_reduced.csv'
    
    # Sampling parameters
    sample_size_per_class = 10000  # Adjust based on your system capacity
    chunk_size = 500000  # Adjust based on available RAM
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Sample size per class: {sample_size_per_class:,}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Required columns: {len(required_columns)}")
    
    # Step 1: Verify column availability
    print("\n--- Step 1: Verifying column availability ---")
    try:
        sample_df = pd.read_csv(input_file, nrows=1, low_memory=False)
        available_columns = list(sample_df.columns)
        
        missing_columns = [col for col in required_columns if col not in available_columns]
        available_required_columns = [col for col in required_columns if col in available_columns]
        
        if missing_columns:
            print(f"Warning: {len(missing_columns)} required columns are missing:")
            for col in missing_columns:
                print(f"  - '{col}'")
            print(f"✓ Will use {len(available_required_columns)} available columns")
            required_columns = available_required_columns
        else:
            print("✓ All required columns are available")
            
    except Exception as e:
        print(f"Error checking columns: {e}")
        return
    
    # Step 2: Discover unique labels
    print("\n--- Step 2: Discovering unique labels ---")
    unique_labels = set()
    chunk_count = 0
    
    try:
        for chunk in pd.read_csv(input_file, usecols=[' Label'], chunksize=chunk_size, low_memory=False):
            chunk_count += 1
            unique_labels.update(chunk[' Label'].unique())
            if chunk_count % 10 == 0:
                print(f"  Processed {chunk_count} chunks for label discovery...")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    unique_labels = sorted(list(unique_labels))
    print(f"✓ Found {len(unique_labels)} unique labels:")
    for i, label in enumerate(unique_labels, 1):
        print(f"  {i}. {label}")
    
    # Step 3: Stratified sampling
    print("\n--- Step 3: Performing stratified sampling ---")
    samples_per_class = {label: [] for label in unique_labels}
    counts_per_class = {label: 0 for label in unique_labels}
    total_chunks_processed = 0
    
    try:
        for chunk in pd.read_csv(input_file, usecols=required_columns, chunksize=chunk_size, low_memory=False):
            total_chunks_processed += 1
            
            # Handle missing values and infinite values
            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk = chunk.fillna(0)
            
            for label in unique_labels:
                if counts_per_class[label] < sample_size_per_class:
                    label_rows = chunk[chunk[' Label'] == label]
                    
                    if len(label_rows) > 0:
                        needed = sample_size_per_class - counts_per_class[label]
                        
                        if len(label_rows) > needed:
                            label_rows = label_rows.sample(n=needed, random_state=42)
                        
                        samples_per_class[label].append(label_rows)
                        counts_per_class[label] += len(label_rows)
            
            # Progress update
            if total_chunks_processed % 5 == 0:
                print(f"  Processed {total_chunks_processed} chunks...")
                completed_classes = sum(1 for count in counts_per_class.values() if count >= sample_size_per_class)
                print(f"    Completed classes: {completed_classes}/{len(unique_labels)}")
            
            # Stop if all classes have enough samples
            if all(counts_per_class[label] >= sample_size_per_class for label in unique_labels):
                print("✓ All classes have reached target sample size")
                break
                
    except Exception as e:
        print(f"Error during sampling: {e}")
        return
    
    # Step 4: Combine and save
    print("\n--- Step 4: Combining and saving dataset ---")
    try:
        all_samples = []
        for label in unique_labels:
            if samples_per_class[label]:
                label_df = pd.concat(samples_per_class[label], ignore_index=True)
                all_samples.append(label_df)
                print(f"✓ {label}: {len(label_df):,} samples")
        
        if not all_samples:
            print("Error: No samples collected!")
            return
        
        # Create final dataset
        final_df = pd.concat(all_samples, ignore_index=True)
        
        # Shuffle the dataset
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to CSV
        final_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Successfully created balanced dataset!")
        print(f"Final dataset shape: {final_df.shape}")
        print(f"Saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024**2):.1f} MB")
        
        # Display class distribution
        print("\nFinal class distribution:")
        class_counts = final_df[' Label'].value_counts().sort_index()
        for label, count in class_counts.items():
            percentage = (count / len(final_df)) * 100
            print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        # Display basic statistics
        print(f"\nDataset statistics:")
        print(f"  Total rows: {len(final_df):,}")
        print(f"  Total columns: {len(final_df.columns)}")
        print(f"  Memory usage: {final_df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
        
        return output_file
        
    except Exception as e:
        print(f"Error combining and saving: {e}")
        return None

def verify_dataset(file_path):
    """
    Verify the created dataset
    """
    print(f"\n=== Verifying Dataset: {file_path} ===")
    
    try:
        # Load and check basic info
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded successfully")
        print(f"Shape: {df.shape}")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        print(f"Missing values: {missing_values}")
        
        # Check label distribution
        print(f"\nLabel distribution:")
        print(df[' Label'].value_counts())
        
        # Display first few rows
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        return True
        
    except Exception as e:
        print(f"Error verifying dataset: {e}")
        return False

# Main execution
if __name__ == "__main__":
    # Create balanced dataset
    output_file = create_balanced_ddos_dataset()
    
    # Verify the created dataset
    if output_file:
        verify_dataset(output_file)
        print(f"\n🎉 Dataset creation completed successfully!")
        print(f"You can now use '{output_file}' for training your ML models.")
    else:
        print("\n❌ Dataset creation failed. Please check the error messages above.")
