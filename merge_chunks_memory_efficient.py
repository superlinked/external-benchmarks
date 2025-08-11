#!/usr/bin/env python3
"""
Memory-efficient chunk merger using memory mapping and streaming.
Combines PyArrow's memory mapping with direct streaming to handle massive datasets.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pathlib import Path
import glob
import json
import os
import gc
import sys
import time
import traceback

def merge_chunks_memory_mapped(chunk_pattern, output_file, batch_size=1000):
    """
    Merge chunks using memory mapping and streaming for minimal memory usage.
    
    This approach:
    1. Memory-maps input files (no loading into RAM)
    2. Streams data directly from input to output
    3. Processes small batches at a time
    4. Never materializes the full dataset
    """
    
    # Find all chunk files
    chunk_files = sorted(glob.glob(chunk_pattern))
    
    if not chunk_files:
        print(f"❌ No chunk files found matching pattern: {chunk_pattern}")
        return False
        
    print(f"🔗 Found {len(chunk_files)} chunk files to merge")
    print(f"📁 Output file: {output_file}")
    print(f"📦 Batch size: {batch_size} rows")
    
    try:
        # Check for normalized chunks
        normalized_pattern = f"{chunk_pattern}.normalized"
        existing_normalized = sorted(glob.glob(normalized_pattern))
        
        if existing_normalized and len(existing_normalized) == len(chunk_files):
            print(f"\n✅ Found {len(existing_normalized)} normalized chunks, using those")
            files_to_merge = existing_normalized
        else:
            print(f"\n📋 Need to normalize schemas first...")
            print(f"   Found {len(existing_normalized)} normalized, need {len(chunk_files)} total")
            
            # Normalize schemas if needed (this part is quick and doesn't use much memory)
            normalized_chunks = []
            for i, chunk_file in enumerate(chunk_files):
                temp_file = f"{chunk_file}.normalized"
                
                if Path(temp_file).exists():
                    if (i + 1) % 50 == 0 or i == 0:
                        print(f"   Skipping chunk {i+1}/{len(chunk_files)} (already normalized)")
                    normalized_chunks.append(temp_file)
                    continue
                
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"   Normalizing chunk {i+1}/{len(chunk_files)}...")
                
                # Use memory mapping to read
                parquet_file = pq.ParquetFile(chunk_file, memory_map=True)
                
                # Process in small batches
                writer = None
                for batch in parquet_file.iter_batches(batch_size=batch_size):
                    # Convert to pandas for normalization (small batch only)
                    df_batch = batch.to_pandas()
                    
                    # Normalize complex fields
                    if 'details' in df_batch.columns:
                        df_batch['details'] = df_batch['details'].apply(
                            lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
                        )
                    if 'categories' in df_batch.columns:
                        df_batch['categories'] = df_batch['categories'].apply(
                            lambda x: json.dumps(x) if isinstance(x, (list, tuple)) else str(x)
                        )
                    
                    # Convert back to arrow and write
                    normalized_batch = pa.Table.from_pandas(df_batch)
                    
                    if writer is None:
                        writer = pq.ParquetWriter(temp_file, normalized_batch.schema, compression='snappy')
                    
                    writer.write_table(normalized_batch)
                    
                    # Clean up batch memory
                    del df_batch, normalized_batch, batch
                
                if writer:
                    writer.close()
                    
                normalized_chunks.append(temp_file)
                
                # Force garbage collection
                gc.collect()
            
            files_to_merge = normalized_chunks
            print(f"✅ Normalized all chunks")
        
        # Remove existing output file if present
        if Path(output_file).exists():
            print(f"\n🗑️ Removing existing output file...")
            os.remove(output_file)
        
        # Create a dataset (memory mapping is automatic for parquet files)
        print(f"\n🔄 Starting memory-mapped streaming merge...")
        print(f"📝 Processing {len(files_to_merge)} files with minimal memory usage")
        
        # Create dataset - PyArrow automatically uses memory mapping for parquet
        dataset = ds.dataset(
            files_to_merge, 
            format='parquet'
        )
        
        # Get schema from dataset
        schema = dataset.schema
        
        # Create output writer
        writer = pq.ParquetWriter(
            output_file, 
            schema, 
            compression='snappy',
            use_dictionary=True,  # Better compression
            write_statistics=True  # For query optimization
        )
        
        # Stream data with small batches
        total_rows = 0
        batch_count = 0
        start_time = time.time()
        last_report_time = start_time
        
        # Get memory usage function
        try:
            import psutil
            process = psutil.Process()
            def get_memory_gb():
                return process.memory_info().rss / (1024**3)
        except ImportError:
            def get_memory_gb():
                return 0
        
        initial_memory = get_memory_gb()
        print(f"📊 Initial memory usage: {initial_memory:.2f} GB")
        
        # Use scanner for efficient streaming with memory mapping
        # Try to use advanced options if available
        try:
            # Try to import ParquetFragmentScanOptions (available in newer PyArrow)
            from pyarrow.dataset import ParquetFragmentScanOptions
            scanner = dataset.scanner(
                batch_size=batch_size,
                use_threads=False,  # Single thread to minimize memory
                fragment_scan_options=ParquetFragmentScanOptions(
                    pre_buffer=False,  # Don't prebuffer
                    use_buffered_stream=False,  # Direct read
                    buffer_size=2**20  # 1MB buffer
                )
            )
            print(f"✅ Using advanced memory-mapped scanner options")
        except (ImportError, AttributeError, TypeError) as e:
            # Fallback for older PyArrow versions or if options not available
            print(f"ℹ️ Using basic scanner (PyArrow {pa.__version__})")
            scanner = dataset.scanner(
                batch_size=batch_size,
                use_threads=False
            )
        
        print(f"📝 Starting batch processing...")
        
        for batch in scanner.to_batches():
            try:
                # Write batch directly without materializing
                writer.write_batch(batch)
                
                batch_rows = len(batch)
                total_rows += batch_rows
                batch_count += 1
                
                # Progress reporting every 100 batches or 30 seconds
                current_time = time.time()
                if batch_count % 100 == 0 or (current_time - last_report_time) > 30:
                    elapsed = current_time - start_time
                    rows_per_sec = total_rows / elapsed if elapsed > 0 else 0
                    current_memory = get_memory_gb()
                    memory_delta = current_memory - initial_memory
                    
                    print(f"   📈 Progress: {total_rows:,} rows | {batch_count:,} batches | "
                          f"{rows_per_sec:.0f} rows/sec | "
                          f"Memory: {current_memory:.2f} GB (+{memory_delta:.2f} GB)")
                    
                    last_report_time = current_time
                    
                    # Force cleanup every 100 batches
                    del batch
                    gc.collect()
                else:
                    del batch
                
                # Even more aggressive memory management
                if batch_count % 1000 == 0:
                    print(f"   🧹 Running deep garbage collection...")
                    gc.collect(2)  # Full collection
                    
                    # Check memory usage
                    current_memory = get_memory_gb()
                    if current_memory > 50:  # Alert if using more than 50GB
                        print(f"   ⚠️ High memory usage: {current_memory:.2f} GB")
                        
            except Exception as e:
                print(f"\n❌ Error processing batch {batch_count}: {e}")
                traceback.print_exc()
                raise
        
        # Close writer
        writer.close()
        
        # Final statistics
        total_time = time.time() - start_time
        final_memory = get_memory_gb()
        peak_memory_delta = final_memory - initial_memory
        
        print(f"\n✅ Merge completed successfully!")
        print(f"📊 Total rows processed: {total_rows:,}")
        print(f"📦 Total batches: {batch_count:,}")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"🚀 Average speed: {total_rows/total_time:.0f} rows/sec")
        print(f"💾 Memory usage: {initial_memory:.2f} GB → {final_memory:.2f} GB (delta: +{peak_memory_delta:.2f} GB)")
        
        # Verify output file
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size / (1024**3)
            print(f"📁 Output file size: {file_size:.2f} GB")
            
            # Quick validation using memory mapping
            print(f"\n🔍 Validating output file...")
            output_pf = pq.ParquetFile(output_file, memory_map=True)
            print(f"✅ Output has {output_pf.metadata.num_rows:,} rows")
            print(f"✅ Output has {len(output_pf.schema)} columns")
        
        print("\n💾 Keeping normalized temp files for potential future use")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💾 Normalized files preserved for debugging")
        return False

def main():
    print("=== MEMORY-EFFICIENT CHUNK MERGER ===")
    print("Using memory mapping + streaming for minimal RAM usage")
    
    # Configuration
    chunk_pattern = "data/large_dataset_with_embeddings.parquet.chunk_*.parquet"
    output_file = "data/large_dataset_with_embeddings_merged.parquet"
    
    # You can adjust batch size based on available memory
    # Smaller = less memory, larger = faster processing
    batch_size = 500  # Very conservative for 250GB+ dataset
    
    print(f"\n📋 Configuration:")
    print(f"   Input pattern: {chunk_pattern}")
    print(f"   Output file: {output_file}")
    print(f"   Batch size: {batch_size} rows")
    
    # Check available memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"\n💾 System memory: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
    except ImportError:
        pass
    
    if merge_chunks_memory_mapped(chunk_pattern, output_file, batch_size):
        print("\n🎉 SUCCESS! Dataset merged with minimal memory usage")
        
        # Optional cleanup
        response = input("\nDo you want to remove the original chunk files? (y/N): ")
        if response.lower() == 'y':
            chunk_files = sorted(glob.glob(chunk_pattern))
            print(f"🗑️ Removing {len(chunk_files)} original chunk files...")
            for cf in chunk_files:
                os.remove(cf)
            print("✅ Cleanup completed")
        else:
            print("💾 Original chunk files preserved")
    else:
        print("\n❌ Merge failed. Check error messages above.")
        print("💡 Try reducing batch_size if you got OOM errors")

if __name__ == "__main__":
    main()