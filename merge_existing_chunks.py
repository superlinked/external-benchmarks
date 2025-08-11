#!/usr/bin/env python3
"""
Recovery script to merge existing chunk files efficiently.
Uses PyArrow datasets API to handle large merges without memory issues.
"""

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import glob
import json
import os

def merge_chunks_streaming(chunk_pattern, output_file):
    """Merge chunk files using PyArrow datasets for memory efficiency"""
    
    # Find all chunk files
    chunk_files = sorted(glob.glob(chunk_pattern))
    
    if not chunk_files:
        print(f"❌ No chunk files found matching pattern: {chunk_pattern}")
        return False
        
    print(f"🔗 Found {len(chunk_files)} chunk files to merge")
    print(f"📁 Output file: {output_file}")
    
    try:
        # Check if normalized chunks already exist
        normalized_pattern = f"{chunk_pattern}.normalized"
        existing_normalized = sorted(glob.glob(normalized_pattern))
        
        if existing_normalized and len(existing_normalized) == len(chunk_files):
            print(f"\n🔄 Found {len(existing_normalized)} existing normalized chunks, skipping normalization...")
            normalized_chunks = existing_normalized
        else:
            # First normalize schemas by converting complex fields to strings
            print(f"\n📋 Normalizing chunk schemas (found {len(existing_normalized)} existing, need {len(chunk_files)})...")
            normalized_chunks = []
            
            for i, chunk_file in enumerate(chunk_files):
                temp_file = f"{chunk_file}.normalized"
                
                # Skip if normalized file already exists
                if Path(temp_file).exists():
                    if (i + 1) % 50 == 0 or i == 0:
                        print(f"   Skipping chunk {i+1}/{len(chunk_files)} (already normalized)...")
                    normalized_chunks.append(temp_file)
                    continue
                
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"   Processing chunk {i+1}/{len(chunk_files)}...")
                
                # Read chunk
                df = pd.read_parquet(chunk_file)
                
                # Convert complex fields to strings to avoid schema conflicts
                if 'details' in df.columns:
                    df['details'] = df['details'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
                if 'categories' in df.columns:
                    df['categories'] = df['categories'].apply(lambda x: json.dumps(x) if isinstance(x, (list, tuple)) else str(x))
                
                # Save normalized chunk
                df.to_parquet(temp_file, index=False, engine='pyarrow', compression='snappy')
                normalized_chunks.append(temp_file)
                
                del df
            
            print(f"✅ Normalized {len(normalized_chunks)} chunks")
        
        # Merge in smaller groups to avoid OOM (max 20 files at a time)
        print("\n🔗 Merging chunks in small groups to avoid OOM...")
        
        group_size = 20
        groups = [normalized_chunks[i:i+group_size] for i in range(0, len(normalized_chunks), group_size)]
        print(f"Created {len(groups)} groups of ~{group_size} files each")
        
        # Remove corrupted output if it exists
        if Path(output_file).exists():
            print(f"🗑️ Removing corrupted output file...")
            os.remove(output_file)
        
        total_records = 0
        writer = None
        
        for group_idx, group_files in enumerate(groups):
            print(f"\n📦 Processing group {group_idx + 1}/{len(groups)} ({len(group_files)} files)")
            
            # Create dataset for this group only
            group_dataset = ds.dataset(group_files, format='parquet')
            
            for batch_idx, batch in enumerate(group_dataset.to_batches(batch_size=25000)):
                if writer is None:
                    # Create writer with schema from first batch
                    writer = pq.ParquetWriter(output_file, batch.schema, compression='snappy')
                    print(f"📝 Created output writer")
                
                writer.write_batch(batch)
                total_records += len(batch)
                
                # Explicitly delete batch and force garbage collection every few batches
                del batch
                if (batch_idx + 1) % 5 == 0:
                    import gc
                    gc.collect()
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"   Group {group_idx + 1}: written {total_records:,} records...")
            
            # Force cleanup of group dataset
            del group_dataset
            import gc
            gc.collect()
            
            print(f"✅ Completed group {group_idx + 1}, total records: {total_records:,}")
        
        if writer:
            writer.close()
        
        print(f"\n✅ Merge completed!")
        print(f"📊 Total records: {total_records:,}")
        print(f"📁 Output file: {output_file}")
        
        # Check file size
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size / (1024**3)
            print(f"📊 File size: {file_size:.1f} GB")
        
        # Keep normalized temp files for future runs (they're expensive to create)
        print("\n💾 Keeping normalized temp files for future runs...")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to merge chunks: {e}")
        
        # Keep normalized files even on failure (they're expensive to recreate)
        print("💾 Normalized files preserved for next attempt")
        
        return False

def main():
    print("=== CHUNK RECOVERY TOOL ===")
    
    # Look for chunk files in data directory
    chunk_pattern = "data/large_dataset_with_embeddings.parquet.chunk_*.parquet"
    output_file = "data/large_dataset_with_embeddings.parquet"
    
    print(f"Looking for chunks: {chunk_pattern}")
    
    if merge_chunks_streaming(chunk_pattern, output_file):
        print("\n🎉 SUCCESS! Your dataset has been recovered.")
        
        # Ask about cleanup
        response = input("\nDo you want to clean up the chunk files? (y/N): ")
        if response.lower() == 'y':
            chunk_files = sorted(glob.glob(chunk_pattern))
            print(f"🗑️ Cleaning up {len(chunk_files)} chunk files...")
            for chunk_file in chunk_files:
                os.remove(chunk_file)
            print("✅ Cleanup completed!")
        else:
            print("💾 Chunk files preserved")
    else:
        print("\n❌ Recovery failed. Chunk files preserved.")

if __name__ == "__main__":
    main()