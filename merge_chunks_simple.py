#!/usr/bin/env python3
"""
Simple sequential merge - processes chunks one by one into a single output.
Memory efficient but slower.
"""

import pandas as pd
from pathlib import Path
import glob
import json

# Configuration
chunk_pattern = "data/large_dataset_with_embeddings.parquet.chunk_*.parquet"
output_file = "data/large_dataset_with_embeddings_merged.parquet"

print("=== SIMPLE CHUNK MERGER ===")

# Find all chunk files
chunk_files = sorted(glob.glob(chunk_pattern))
print(f"Found {len(chunk_files)} chunk files")

if not chunk_files:
    print("❌ No chunk files found!")
    exit(1)

print(f"Output: {output_file}")
print("\nProcessing chunks sequentially...")

total_records = 0

for i, chunk_file in enumerate(chunk_files):
    if (i + 1) % 50 == 0 or i == 0:
        print(f"Processing chunk {i+1}/{len(chunk_files)}...")
    
    # Read chunk
    df = pd.read_parquet(chunk_file)
    
    # Normalize schema - convert complex fields to strings
    if 'details' in df.columns:
        df['details'] = df['details'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
    if 'categories' in df.columns:
        df['categories'] = df['categories'].apply(lambda x: json.dumps(x) if isinstance(x, (list, tuple)) else str(x))
    
    total_records += len(df)
    
    # Write to output
    if i == 0:
        # First chunk - create new file
        df.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
    else:
        # Append to existing - using temporary file approach
        temp_file = f"{output_file}.temp"
        df.to_parquet(temp_file, index=False, engine='pyarrow', compression='snappy')
        
        # Read both and combine
        existing_df = pd.read_parquet(output_file)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        
        # Save combined
        combined_df.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
        
        # Clean up
        Path(temp_file).unlink(missing_ok=True)
        del existing_df, combined_df
    
    del df
    
    if (i + 1) % 50 == 0:
        print(f"  Progress: {total_records:,} records processed")

print(f"\n✅ Complete!")
print(f"📊 Total records: {total_records:,}")
print(f"📁 Output: {output_file}")

# Check file size
if Path(output_file).exists():
    file_size = Path(output_file).stat().st_size / (1024**3)
    print(f"📊 File size: {file_size:.1f} GB")

print("\n💡 Tip: Rename to 'large_dataset_with_embeddings.parquet' when ready")