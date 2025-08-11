#!/usr/bin/env python3
"""
Verify the merge was complete by counting records in chunks vs merged file.
Uses memory-efficient methods to count rows.
"""

import pyarrow.parquet as pq
import glob
from pathlib import Path

def count_parquet_rows(file_path):
    """Count rows in a parquet file without loading data"""
    try:
        # Use metadata only - no data loading
        parquet_file = pq.ParquetFile(file_path, memory_map=True)
        return parquet_file.metadata.num_rows
    except Exception as e:
        print(f"   ❌ Error reading {file_path}: {e}")
        return 0

def main():
    print("=== MERGE VERIFICATION ===\n")
    
    # Count rows in merged file
    merged_file = "data/large_dataset_with_embeddings_merged.parquet"
    
    if Path(merged_file).exists():
        print(f"📊 Counting rows in merged file...")
        merged_count = count_parquet_rows(merged_file)
        print(f"✅ Merged file has {merged_count:,} rows")
        file_size = Path(merged_file).stat().st_size / (1024**3)
        print(f"📁 File size: {file_size:.2f} GB\n")
    else:
        print(f"❌ Merged file not found: {merged_file}")
        merged_count = 0
    
    # Count rows in all normalized chunks
    print(f"📊 Counting rows in original chunks...")
    
    # Use normalized chunks if they exist, otherwise use original chunks
    normalized_pattern = "data/large_dataset_with_embeddings.parquet.chunk_*.parquet.normalized"
    chunk_pattern = "data/large_dataset_with_embeddings.parquet.chunk_*.parquet"
    
    normalized_files = sorted(glob.glob(normalized_pattern))
    if normalized_files:
        print(f"   Using {len(normalized_files)} normalized chunks")
        chunk_files = normalized_files
    else:
        chunk_files = sorted(glob.glob(chunk_pattern))
        print(f"   Using {len(chunk_files)} original chunks")
    
    if not chunk_files:
        print("❌ No chunk files found!")
        return
    
    # Count rows in each chunk
    total_chunk_rows = 0
    chunk_sizes = []
    
    for i, chunk_file in enumerate(chunk_files):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"   Processing chunk {i+1}/{len(chunk_files)}...")
        
        rows = count_parquet_rows(chunk_file)
        total_chunk_rows += rows
        chunk_sizes.append(rows)
    
    print(f"\n✅ Total rows in {len(chunk_files)} chunks: {total_chunk_rows:,}")
    
    # Calculate statistics
    if chunk_sizes:
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
        min_chunk_size = min(chunk_sizes)
        max_chunk_size = max(chunk_sizes)
        
        print(f"\n📊 Chunk statistics:")
        print(f"   Average rows per chunk: {avg_chunk_size:,.0f}")
        print(f"   Min rows in a chunk: {min_chunk_size:,}")
        print(f"   Max rows in a chunk: {max_chunk_size:,}")
    
    # Compare counts
    print(f"\n{'='*50}")
    print(f"📋 VERIFICATION RESULTS:")
    print(f"{'='*50}")
    print(f"Chunks total:  {total_chunk_rows:,} rows")
    print(f"Merged file:   {merged_count:,} rows")
    
    if merged_count == total_chunk_rows:
        print(f"\n✅ SUCCESS: Row counts match perfectly!")
        print(f"   All {total_chunk_rows:,} rows were successfully merged.")
    elif merged_count > 0:
        difference = total_chunk_rows - merged_count
        percentage = (merged_count / total_chunk_rows * 100) if total_chunk_rows > 0 else 0
        
        if difference > 0:
            print(f"\n⚠️ WARNING: Missing {difference:,} rows ({100-percentage:.2f}% complete)")
        else:
            print(f"\n⚠️ WARNING: Extra {-difference:,} rows in merged file")
        
        print(f"   This could indicate an incomplete merge or duplicate processing.")
    else:
        print(f"\n❌ FAILED: No rows found in merged file")
    
    # Also check the original large dataset file if it exists
    original_file = "data/large_dataset_with_embeddings.parquet"
    if Path(original_file).exists():
        print(f"\n📊 Checking original file for comparison...")
        original_count = count_parquet_rows(original_file)
        original_size = Path(original_file).stat().st_size / (1024**3)
        print(f"   Original file: {original_count:,} rows, {original_size:.2f} GB")
        
        if original_count == merged_count:
            print(f"   ✅ Merged file matches original!")

if __name__ == "__main__":
    main()