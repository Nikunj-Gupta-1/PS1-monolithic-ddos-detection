import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path

def concatenate_ddos_dataset():
    """
    Memory-efficient concatenation of CIC-DDoS2019 CSV files
    Handles column mismatches and large file sizes
    """
    
    # Define folder paths
    folders = ['01-12', '03-11']
    all_files = []
    
    # Collect all CSV files
    for folder in folders:
        folder_path = Path(folder)
        if folder_path.exists():
            all_files.extend(glob.glob(os.path.join(folder, '*.csv')))
        else:
            print(f"Warning: Folder {folder} not found")
    
    if not all_files:
        print("No CSV files found. Please check folder structure.")
        return
    
    print(f"Found {len(all_files)} CSV files to process")
    
    # Step 1: Discover all unique columns across files
    print("Discovering column structure...")
    all_columns = set()
    file_columns = {}
    
    for file in all_files:
        try:
            # Read only header to get column names
            df_sample = pd.read_csv(file, nrows=0)
            file_columns[file] = list(df_sample.columns)
            all_columns.update(df_sample.columns)
            print(f"✓ {os.path.basename(file)}: {len(df_sample.columns)} columns")
        except Exception as e:
            print(f"✗ Error reading {file}: {e}")
    
    all_columns = sorted(list(all_columns))
    print(f"\nTotal unique columns found: {len(all_columns)}")
    
    # Step 2: Process files in chunks and concatenate
    output_file = 'CIC_DDoS2019_combined.csv'
    chunk_size = 50000  # Adjust based on available memory
    first_file = True
    total_rows = 0
    
    print(f"\nProcessing files in chunks of {chunk_size} rows...")
    
    for file_idx, file in enumerate(all_files):
        print(f"Processing {os.path.basename(file)} ({file_idx + 1}/{len(all_files)})...")
        
        try:
            # Process file in chunks
            chunk_count = 0
            for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
                chunk_count += 1
                
                # Align columns - add missing columns with NaN
                for col in all_columns:
                    if col not in chunk.columns:
                        chunk[col] = np.nan
                
                # Reorder columns to match standard order
                chunk = chunk[all_columns]
                
                # Write to output file
                mode = 'w' if first_file else 'a'
                header = first_file
                chunk.to_csv(output_file, mode=mode, header=header, index=False)
                
                first_file = False
                total_rows += len(chunk)
                
                if chunk_count % 10 == 0:
                    print(f"  Processed {chunk_count} chunks from {os.path.basename(file)}")
                    
        except Exception as e:
            print(f"✗ Error processing {file}: {e}")
            continue
    
    print(f"\n✓ Concatenation complete!")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Output file: {output_file}")
    print(f"Total columns: {len(all_columns)}")
    
    return output_file, all_columns

# Run the concatenation
if __name__ == "__main__":
    output_file, columns = concatenate_ddos_dataset()
