#!/usr/bin/env python3
"""
Process large combined Amazon dataset with crash recovery and checkpointing.
Downloads multiple datasets, combines them, and processes with periodic saves.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
import time
import subprocess
import os
import tempfile
warnings.filterwarnings('ignore')

def download_datasets(datasets, force_download=False, data_dir="data"):
    """Download datasets from GCS bucket"""
    print(f"=== DOWNLOADING {len(datasets)} DATASETS ===")
    
    # Ensure data directory exists
    Path(data_dir).mkdir(exist_ok=True)
    
    downloaded_files = []
    for dataset in datasets:
        filename = f"meta_{dataset}.jsonl"
        filepath = Path(data_dir) / filename
        
        if filepath.exists() and not force_download:
            print(f"✅ {filepath} already exists, skipping download")
            downloaded_files.append(str(filepath))
            continue
            
        print(f"📥 Downloading {filename}...")
        try:
            result = subprocess.run([
                'gsutil', 'cp', 
                f'gs://superlinked-benchmarks-external/{filename}', 
                str(data_dir)
            ], capture_output=True, text=True, check=True)
            downloaded_files.append(str(filepath))
            print(f"✅ Downloaded {filename}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download {filename}: {e}")
            continue
    
    return downloaded_files

def count_records_in_file(filepath):
    """Count total records in JSONL file efficiently"""
    print(f"Counting records in {filepath}...")
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            count += 1
    return count

def combine_datasets(downloaded_files, output_file="combined_dataset.jsonl", data_dir="data"):
    """Combine multiple JSONL files into one with dataset labels"""
    output_path = Path(data_dir) / output_file
    if output_path.exists():
        print(f"✅ Combined dataset {output_path} already exists")
        return str(output_path)
        
    print(f"=== COMBINING {len(downloaded_files)} DATASETS ===")
    
    # Count total records first
    total_records = 0
    file_counts = {}
    for filepath in downloaded_files:
        count = count_records_in_file(filepath)
        file_counts[filepath] = count
        total_records += count
        print(f"  {filepath}: {count:,} records")
    
    print(f"Total records to combine: {total_records:,}")
    
    # Combine all files
    with open(output_path, 'w') as outfile:
        progress_bar = tqdm(total=total_records, desc="Combining files")
        
        for filepath in downloaded_files:
            # Extract dataset name from filename
            dataset_name = Path(filepath).stem.replace('meta_', '')
            
            with open(filepath, 'r') as infile:
                for line in infile:
                    try:
                        record = json.loads(line.strip())
                        record['source_dataset'] = dataset_name  # Add source label
                        outfile.write(json.dumps(record) + '\n')
                        progress_bar.update(1)
                    except json.JSONDecodeError:
                        continue
        
        progress_bar.close()
    
    print(f"✅ Combined dataset saved as: {output_path}")
    return str(output_path)

def load_checkpoint(checkpoint_path):
    """Load processing checkpoint"""
    if not Path(checkpoint_path).exists():
        return None
    
    try:
        df = pd.read_parquet(checkpoint_path)
        print(f"📂 Loaded checkpoint: {len(df)} records already processed")
        return df
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None

def save_checkpoint(df, checkpoint_path, processed_count):
    """Save processing checkpoint"""
    try:
        df.to_parquet(checkpoint_path, index=False)
        print(f"💾 Checkpoint saved: {processed_count} records processed")
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")

def load_metadata(filepath, start_from=0, limit=None):
    """Load and parse JSONL metadata file with resume capability"""
    print(f"Loading metadata from {filepath} (starting from record {start_from})")
    records = []
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Loading records")):
            if i < start_from:
                continue
            if limit and len(records) >= limit:
                break
                
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(records)} records (skipped first {start_from})")
    return records

def prepare_dataframe(records):
    """Convert records to pandas DataFrame with relevant columns"""
    print("Converting to DataFrame...")
    
    data = []
    for record in records:
        # Combine description and features for embedding text
        description_text = ""
        if record.get('description'):
            description_text = " ".join(record['description'])
        
        features_text = ""
        if record.get('features'):
            features_text = " ".join(record['features'])
        
        # Combine title, description, and features for embedding
        combined_text = f"{record.get('title', '')} {description_text} {features_text}".strip()
        
        # Extract price as numeric value
        price = None
        if record.get('price'):
            try:
                price_str = str(record['price']).replace('$', '').replace(',', '')
                price = float(price_str) if price_str.replace('.', '').isdigit() else None
            except:
                price = None
        
        data.append({
            'parent_asin': record.get('parent_asin'),
            'title': record.get('title', ''),
            'description': description_text,
            'features': features_text,
            'combined_text': combined_text,
            'average_rating': record.get('average_rating'),
            'rating_number': record.get('rating_number', 0),
            'price': price,
            'main_category': record.get('main_category', ''),
            'categories': record.get('categories', []),
            'store': record.get('store', ''),
            'details': record.get('details', {}),
            'source_dataset': record.get('source_dataset', 'unknown')
        })
    
    df = pd.DataFrame(data)
    
    # Add derived columns for benchmarking
    df['has_price'] = df['price'].notna()
    df['rating_tier'] = pd.cut(df['average_rating'], 
                              bins=[0, 3.5, 4.0, 4.5, 5.0], 
                              labels=['low', 'medium', 'high', 'excellent'],
                              include_lowest=True)
    df['review_volume'] = pd.cut(df['rating_number'], 
                               bins=[0, 10, 100, 1000, float('inf')], 
                               labels=['few', 'moderate', 'many', 'popular'],
                               include_lowest=True)
    
    print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    return df

def load_embedding_model(force_cpu=False):
    """Load BGE-small-en-v1.5 model with advanced optimizations"""
    print("Loading BGE-small-en-v1.5 model...")
    
    model_name = "BAAI/bge-small-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Force CPU if requested
    if force_cpu:
        print("🖥️  Forced CPU mode - skipping GPU detection")
        model = AutoModel.from_pretrained(model_name)
        device = 'cpu'
        print("✅ Using CPU (forced)")
    else:
        # Try CUDA first, then MPS, fallback to CPU
        device = 'cpu'
        if torch.cuda.is_available():
            try:
                # Load with mixed precision for CUDA performance
                model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
                model = model.to('cuda')
                device = 'cuda'
                print(f"✅ Using CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            except Exception as e:
                print(f"CUDA failed ({e}), trying fallback options...")
                model = AutoModel.from_pretrained(model_name)
                device = 'cpu'
        elif torch.backends.mps.is_available():
            try:
                # Load with float32 for MPS stability 
                model = AutoModel.from_pretrained(model_name)
                model = model.to('mps')
                device = 'mps'
                print("✅ Using MPS with float32 for stability and performance")
            except Exception as e:
                print(f"MPS failed ({e}), falling back to CPU")
                model = AutoModel.from_pretrained(model_name)
                device = 'cpu'
                print("✅ Using CPU")
        else:
            model = AutoModel.from_pretrained(model_name)
            print("✅ Using CPU")
    
    model.eval()
    print(f"Model loaded: {model_name}")
    return model, tokenizer, device

def encode_texts(texts, model, tokenizer, device, batch_size=1024):
    """Encode texts with optimizations for GPU acceleration"""
    embeddings = []
    
    # Process in batches
    with torch.no_grad():
        if device == 'mps':
            torch.mps.empty_cache()
        elif device == 'cuda':
            torch.cuda.empty_cache()
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing"):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                             return_tensors="pt", max_length=512)
            
            if device in ['cuda', 'mps']:
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            hidden_states = outputs.last_hidden_state
            
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask
            
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            embeddings.extend(batch_embeddings.cpu().numpy())
            
            # Only clear cache occasionally to avoid overhead
            if i % 10 == 0:
                if device == 'mps':
                    torch.mps.empty_cache()
                elif device == 'cuda':
                    torch.cuda.empty_cache()
            del inputs, outputs, hidden_states, mask_expanded, sum_embeddings, sum_mask, batch_embeddings
    
    return np.array(embeddings)

def encode_field_worker(field_data):
    """Worker function for parallel field processing"""
    field_name, texts, model, tokenizer, device, batch_size = field_data
    print(f"Processing {field_name} ({len(texts)} texts)...")
    embeddings = encode_texts(texts, model, tokenizer, device, batch_size)
    return field_name, embeddings

def process_chunk_with_embeddings(df_chunk, model_tokenizer_device, batch_size=512, max_workers=8):
    """Process a chunk of data with embeddings"""
    model, tokenizer, device = model_tokenizer_device
    
    # Prepare all text fields
    titles = df_chunk['title'].fillna('').tolist()
    descriptions = df_chunk['description'].fillna('').tolist() 
    features = df_chunk['features'].fillna('').tolist()
    categories_text = df_chunk['main_category'].fillna('').tolist()
    stores = df_chunk['store'].fillna('').tolist()
    
    # Process categories array into text
    categories_list = []
    for cats in df_chunk['categories']:
        if isinstance(cats, (list, np.ndarray)) and len(cats) > 0:
            categories_list.append(' '.join([str(c) for c in cats]))
        else:
            categories_list.append('')
    
    # Process details dict into text  
    details_text = []
    for details in df_chunk['details']:
        if isinstance(details, dict):
            useful_details = []
            for key, value in details.items():
                if value is not None and str(value).strip():
                    useful_details.append(f"{key}: {value}")
            details_text.append(' | '.join(useful_details[:10]))
        else:
            details_text.append('')
    
    # Prepare field data for parallel processing
    field_tasks = [
        ('title', titles, model, tokenizer, device, batch_size),
        ('description', descriptions, model, tokenizer, device, batch_size),
        ('features', features, model, tokenizer, device, batch_size),
        ('main_category', categories_text, model, tokenizer, device, batch_size),
        ('store', stores, model, tokenizer, device, batch_size),
        ('categories', categories_list, model, tokenizer, device, batch_size),
        ('details', details_text, model, tokenizer, device, batch_size)
    ]
    
    # Process fields in parallel
    field_embeddings = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_field = {executor.submit(encode_field_worker, task): task[0] for task in field_tasks}
        
        for future in concurrent.futures.as_completed(future_to_field):
            field_name = future_to_field[future]
            try:
                field_name, embeddings = future.result()
                field_embeddings[field_name] = embeddings
                print(f"✅ Completed {field_name} embeddings: {embeddings.shape}")
            except Exception as exc:
                print(f"❌ {field_name} generated an exception: {exc}")
                raise exc
    
    # Concatenate embeddings
    concatenated_embeddings = np.concatenate([
        field_embeddings['title'],
        field_embeddings['description'],
        field_embeddings['features'],
        field_embeddings['main_category'],
        field_embeddings['store'],
        field_embeddings['categories'],
        field_embeddings['details']
    ], axis=1)
    
    # Add embeddings to dataframe
    df_chunk_with_embeddings = df_chunk.copy()
    df_chunk_with_embeddings['embedding'] = concatenated_embeddings.tolist()
    
    return df_chunk_with_embeddings

def process_with_checkpoints(combined_file, output_file, chunk_size=10000, checkpoint_interval=50000, data_dir="data", **kwargs):
    """Process large dataset with periodic checkpointing"""
    # Ensure output is in data directory
    if not str(output_file).startswith(data_dir):
        output_file = Path(data_dir) / output_file
    checkpoint_path = f"{output_file}.checkpoint"
    
    # Try to load existing checkpoint
    existing_df = load_checkpoint(checkpoint_path)
    start_from = 0 if existing_df is None else len(existing_df)
    
    print(f"=== PROCESSING WITH CHECKPOINTS ===")
    print(f"Chunk size: {chunk_size:,} records")
    print(f"Checkpoint interval: {checkpoint_interval:,} records")
    print(f"Starting from record: {start_from:,}")
    
    # Load embedding model
    model_tokenizer_device = load_embedding_model(force_cpu=kwargs.get('force_cpu', False))
    
    # Initialize results
    all_processed_dfs = []
    if existing_df is not None:
        all_processed_dfs.append(existing_df)
    
    processed_count = start_from
    total_start_time = time.time()
    
    # Process in chunks
    while True:
        print(f"\n--- LOADING CHUNK STARTING FROM {processed_count:,} ---")
        
        # Load next chunk
        try:
            records = load_metadata(combined_file, start_from=processed_count, limit=chunk_size)
            if not records:
                print("✅ No more records to process")
                break
                
            df_chunk = prepare_dataframe(records)
            print(f"Processing chunk: {len(df_chunk)} records")
            
            # Process chunk with embeddings
            chunk_start_time = time.time()
            df_processed = process_chunk_with_embeddings(
                df_chunk, model_tokenizer_device, 
                batch_size=kwargs.get('batch_size', 512),
                max_workers=kwargs.get('max_workers', 8)
            )
            chunk_time = time.time() - chunk_start_time
            
            all_processed_dfs.append(df_processed)
            processed_count += len(df_processed)
            
            print(f"✅ Chunk completed in {chunk_time:.1f}s ({len(df_processed)/chunk_time:.1f} records/sec)")
            
            # Save checkpoint if interval reached
            if processed_count % checkpoint_interval == 0 or len(records) < chunk_size:
                print(f"\n💾 SAVING CHECKPOINT AT {processed_count:,} RECORDS")
                combined_df = pd.concat(all_processed_dfs, ignore_index=True)
                save_checkpoint(combined_df, checkpoint_path, processed_count)
                
                # Clear memory
                all_processed_dfs = [combined_df]
                
        except Exception as e:
            print(f"❌ Error processing chunk: {e}")
            # Save emergency checkpoint
            if all_processed_dfs:
                emergency_checkpoint = f"{checkpoint_path}.emergency"
                combined_df = pd.concat(all_processed_dfs, ignore_index=True)
                save_checkpoint(combined_df, emergency_checkpoint, processed_count)
                print(f"💾 Emergency checkpoint saved: {emergency_checkpoint}")
            raise e
    
    # Final save
    print(f"\n=== FINAL SAVE ===")
    final_df = pd.concat(all_processed_dfs, ignore_index=True)
    final_df.to_parquet(output_file, index=False)
    
    total_time = time.time() - total_start_time
    print(f"✅ Final dataset saved: {output_file}")
    print(f"📊 Total records processed: {len(final_df):,}")
    print(f"📊 Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"📊 Average speed: {len(final_df)/total_time:.1f} records/second")
    print(f"📊 File size: {Path(output_file).stat().st_size / (1024*1024):.1f} MB")
    
    # Clean up checkpoint
    if Path(checkpoint_path).exists():
        os.remove(checkpoint_path)
        print("🗑️  Checkpoint file cleaned up")
    
    return final_df

def main():
    parser = argparse.ArgumentParser(description='Process large combined Amazon dataset with crash recovery')
    parser.add_argument('--datasets', nargs='+', 
                       default=['Automotive', 'Beauty_and_Personal_Care', 'Books', 'Electronics', 'Tools_and_Home_Improvement'],
                       help='List of datasets to download and process')
    parser.add_argument('--output', type=str, default='large_dataset_with_embeddings.parquet',
                       help='Output parquet file path')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for embedding generation')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Maximum number of parallel workers')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Number of records to process in each chunk')
    parser.add_argument('--checkpoint-interval', type=int, default=50000,
                       help='Save checkpoint every N records')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage instead of GPU acceleration')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download of datasets even if they exist')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download step and use existing files')
    parser.add_argument('--skip-combine', action='store_true',
                       help='Skip combine step and use existing combined file')
    
    args = parser.parse_args()
    
    print("=== LARGE DATASET PROCESSING WITH CRASH RECOVERY ===")
    print(f"Target datasets: {args.datasets}")
    print(f"Output file: {args.output}")
    print(f"Chunk size: {args.chunk_size:,}")
    print(f"Checkpoint interval: {args.checkpoint_interval:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max workers: {args.max_workers}")
    print(f"Force CPU: {args.force_cpu}")
    
    overall_start = time.time()
    
    # Step 1: Download datasets
    if not args.skip_download:
        downloaded_files = download_datasets(args.datasets, force_download=args.force_download, data_dir="data")
        if not downloaded_files:
            print("❌ No datasets downloaded, exiting")
            return
    else:
        downloaded_files = [f"data/meta_{dataset}.jsonl" for dataset in args.datasets]
        print(f"📂 Using existing files: {downloaded_files}")
    
    # Step 2: Combine datasets
    if not args.skip_combine:
        combined_file = combine_datasets(downloaded_files, data_dir="data")
    else:
        combined_file = "data/combined_dataset.jsonl"
        print(f"📂 Using existing combined file: {combined_file}")
    
    # Step 3: Process with embeddings and checkpointing
    final_df = process_with_checkpoints(
        combined_file, 
        args.output,
        chunk_size=args.chunk_size,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        force_cpu=args.force_cpu,
        data_dir="data"
    )
    
    # Final statistics
    overall_time = time.time() - overall_start
    print(f"\n=== PROCESS COMPLETE ===")
    print(f"🎯 Total records: {len(final_df):,}")
    print(f"⏱️  Overall time: {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
    print(f"🚀 Overall speed: {len(final_df)/overall_time:.1f} records/second")
    
    # Dataset composition
    if 'source_dataset' in final_df.columns:
        print(f"\n📊 DATASET COMPOSITION:")
        composition = final_df['source_dataset'].value_counts()
        for dataset, count in composition.items():
            print(f"  {dataset}: {count:,} records ({count/len(final_df)*100:.1f}%)")

if __name__ == "__main__":
    main()