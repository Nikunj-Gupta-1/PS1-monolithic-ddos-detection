import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_balanced_ddos_dataset():
    """
    Create a balanced CIC-DDoS2019 dataset with equal attack and benign samples
    Attack types are balanced among themselves, and total attacks = total benign
    """
    
    print("=== CIC-DDoS2019 Balanced Attack vs Benign Dataset Creator ===")
    
    # Configuration parameters - using EXACT column names from your dataset
    required_columns = [
        ' Flow Duration',
        ' Total Fwd Packets',
        ' Total Backward Packets',
        'Total Length of Fwd Packets',
        ' Total Length of Bwd Packets',
        ' Fwd Packet Length Max',
        ' Fwd Packet Length Min',
        ' Fwd Packet Length Mean',
        ' Fwd Packet Length Std',
        'Bwd Packet Length Max',
        ' Bwd Packet Length Min',
        ' Bwd Packet Length Mean',
        ' Bwd Packet Length Std',
        'Flow Bytes/s',
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
        'Fwd IAT Total',
        'Bwd IAT Total',
        'Fwd PSH Flags',
        ' Bwd PSH Flags',
        ' ACK Flag Count',
        ' SYN Flag Count',
        'FIN Flag Count',
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
        'Fwd Packets/s',
        ' Bwd Packets/s',
        ' Max Packet Length',
        ' Min Packet Length',
        ' Protocol',
        ' Label'
    ]
    
    # File paths
    input_file = 'CIC_DDoS2019_combined.csv'
    output_file = 'CIC_DDoS2019_2_Million_records_balanced.csv'
    
    # Sampling parameters
    total_benign_samples = 1000000      # Total benign samples
    total_attack_samples = 1000000      # Total attack samples (equal to benign)
    chunk_size = 500000
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Target benign samples: {total_benign_samples:,}")
    print(f"Target attack samples: {total_attack_samples:,}")
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
            print(f"Warning: {len(missing_columns)} required columns are missing")
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
            if chunk_count % 20 == 0:
                print(f"  Processed {chunk_count} chunks for label discovery...")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    unique_labels = sorted(list(unique_labels))
    
    # Separate benign and attack labels
    benign_label = 'BENIGN'
    attack_labels = [label for label in unique_labels if label != benign_label]
    
    print(f"✓ Found {len(unique_labels)} unique labels:")
    print(f"  Benign: {benign_label}")
    print(f"  Attack types ({len(attack_labels)}):")
    for i, label in enumerate(attack_labels, 1):
        print(f"    {i}. {label}")
    
    # Calculate samples per attack type
    samples_per_attack_type = total_attack_samples // len(attack_labels)
    remaining_attack_samples = total_attack_samples % len(attack_labels)
    
    print(f"\n--- Sampling Strategy ---")
    print(f"Benign samples: {total_benign_samples:,}")
    print(f"Samples per attack type: {samples_per_attack_type:,}")
    if remaining_attack_samples > 0:
        print(f"Extra samples for first {remaining_attack_samples} attack types: +1 each")
    print(f"Total attack samples: {total_attack_samples:,}")
    print(f"Final dataset ratio: 50% Benign, 50% Attack")
    
    # Step 3: Stratified sampling
    print("\n--- Step 3: Performing balanced sampling ---")
    
    # Initialize storage
    benign_samples = []
    attack_samples = {label: [] for label in attack_labels}
    
    # Counters
    benign_count = 0
    attack_counts = {label: 0 for label in attack_labels}
    
    # Target counts for each attack type
    attack_targets = {}
    for i, label in enumerate(attack_labels):
        attack_targets[label] = samples_per_attack_type + (1 if i < remaining_attack_samples else 0)
    
    total_chunks_processed = 0
    
    try:
        for chunk in pd.read_csv(input_file, usecols=required_columns, chunksize=chunk_size, low_memory=False):
            total_chunks_processed += 1
            
            # Handle missing values and infinite values
            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk = chunk.fillna(0)
            
            # Sample benign data
            if benign_count < total_benign_samples:
                benign_rows = chunk[chunk[' Label'] == benign_label]
                if len(benign_rows) > 0:
                    needed = total_benign_samples - benign_count
                    if len(benign_rows) > needed:
                        benign_rows = benign_rows.sample(n=needed, random_state=42)
                    benign_samples.append(benign_rows)
                    benign_count += len(benign_rows)
            
            # Sample attack data
            for attack_label in attack_labels:
                if attack_counts[attack_label] < attack_targets[attack_label]:
                    attack_rows = chunk[chunk[' Label'] == attack_label]
                    if len(attack_rows) > 0:
                        needed = attack_targets[attack_label] - attack_counts[attack_label]
                        if len(attack_rows) > needed:
                            attack_rows = attack_rows.sample(n=needed, random_state=42)
                        attack_samples[attack_label].append(attack_rows)
                        attack_counts[attack_label] += len(attack_rows)
            
            # Progress update
            if total_chunks_processed % 10 == 0:
                print(f"  Processed {total_chunks_processed} chunks...")
                print(f"    Benign: {benign_count:,}/{total_benign_samples:,}")
                completed_attacks = sum(1 for label in attack_labels if attack_counts[label] >= attack_targets[label])
                print(f"    Completed attack types: {completed_attacks}/{len(attack_labels)}")
            
            # Check if we have enough samples
            benign_complete = benign_count >= total_benign_samples
            attacks_complete = all(attack_counts[label] >= attack_targets[label] for label in attack_labels)
            
            if benign_complete and attacks_complete:
                print("✓ All sampling targets reached")
                break
                
    except Exception as e:
        print(f"Error during sampling: {e}")
        return
    
    # Step 4: Combine and save
    print("\n--- Step 4: Combining and saving dataset ---")
    try:
        all_samples = []
        
        # Add benign samples
        if benign_samples:
            benign_df = pd.concat(benign_samples, ignore_index=True)
            all_samples.append(benign_df)
            print(f"✓ Benign: {len(benign_df):,} samples")
        
        # Add attack samples
        total_attack_collected = 0
        for attack_label in attack_labels:
            if attack_samples[attack_label]:
                attack_df = pd.concat(attack_samples[attack_label], ignore_index=True)
                all_samples.append(attack_df)
                total_attack_collected += len(attack_df)
                print(f"✓ {attack_label}: {len(attack_df):,} samples")
        
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
        
        benign_total = class_counts.get('BENIGN', 0)
        attack_total = sum(count for label, count in class_counts.items() if label != 'BENIGN')
        
        print(f"  BENIGN: {benign_total:,} ({benign_total/len(final_df)*100:.1f}%)")
        print(f"  ALL ATTACKS: {attack_total:,} ({attack_total/len(final_df)*100:.1f}%)")
        print("\nAttack type breakdown:")
        for label, count in class_counts.items():
            if label != 'BENIGN':
                percentage = (count / len(final_df)) * 100
                print(f"    {label}: {count:,} ({percentage:.1f}%)")
        
        # Display basic statistics
        print(f"\nDataset statistics:")
        print(f"  Total rows: {len(final_df):,}")
        print(f"  Total columns: {len(final_df.columns)}")
        print(f"  Attack/Benign ratio: {attack_total/benign_total:.2f}:1")
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
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded successfully")
        print(f"Shape: {df.shape}")
        
        # Check label distribution
        print(f"\nLabel distribution:")
        label_counts = df[' Label'].value_counts()
        
        benign_count = label_counts.get('BENIGN', 0)
        attack_count = sum(count for label, count in label_counts.items() if label != 'BENIGN')
        
        print(f"  Benign samples: {benign_count:,}")
        print(f"  Attack samples: {attack_count:,}")
        print(f"  Ratio (Attack:Benign): {attack_count/benign_count:.2f}:1")
        
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
        print(f"\n🎉 Balanced dataset creation completed successfully!")
        print(f"You now have a properly balanced dataset with 50% benign and 50% attack samples.")
        print(f"Use '{output_file}' for training your ML models.")
    else:
        print("\n❌ Dataset creation failed. Please check the error messages above.")
