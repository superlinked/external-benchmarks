#!/usr/bin/env python3
"""
Process Amazon dataset and generate vector embeddings.
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
warnings.filterwarnings('ignore')

def load_metadata(filepath):
    """Load and parse JSONL metadata file"""
    print(f"Loading metadata from {filepath}")
    records = []
    
    with open(filepath, 'r') as f:
        for line in tqdm(f, desc="Loading records"):
            record = json.loads(line.strip())
            records.append(record)
    
    print(f"Loaded {len(records)} records")
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
                # Remove $ and convert to float
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
            'details': record.get('details', {})
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
    
    print(f"  Processing {len(texts)} texts with massive batch size {batch_size}...")
    
    # Process entire dataset in single batch if small enough
    if len(texts) <= batch_size:
        print(f"  Single batch processing {len(texts)} texts...")
        with torch.no_grad():
            # Tokenize all at once
            inputs = tokenizer(texts, padding=True, truncation=True, 
                             return_tensors="pt", max_length=512)
            
            # Move to device
            if device in ['cuda', 'mps']:
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = model(**inputs)
            
            # Optimized pooling
            attention_mask = inputs['attention_mask']
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask
            
            # L2 normalize
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            embeddings = batch_embeddings.cpu().numpy()
            
            # Memory cleanup
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda':
                torch.cuda.empty_cache()
    else:
        # Standard batch processing for larger datasets
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
                if i % 10 == 0:  # Clear every 10 batches
                    if device == 'mps':
                        torch.mps.empty_cache()
                    elif device == 'cuda':
                        torch.cuda.empty_cache()
                del inputs, outputs, hidden_states, mask_expanded, sum_embeddings, sum_mask, batch_embeddings
        
        embeddings = np.array(embeddings)
    
    return embeddings

def encode_field_worker(field_data):
    """Worker function for parallel field processing"""
    field_name, texts, model, tokenizer, device, batch_size = field_data
    print(f"Processing {field_name} ({len(texts)} texts)...")
    embeddings = encode_texts(texts, model, tokenizer, device, batch_size)
    return field_name, embeddings

def generate_embeddings(df, model_tokenizer_device, batch_size=1024, max_workers=12):
    """Generate embeddings for 7 fields in parallel and concatenate to 2688 dimensions"""
    model, tokenizer, device = model_tokenizer_device
    print("Generating embeddings for 7 fields in parallel: title, description, features, category, store, cat_hierarchy, details...")
    
    # Prepare all text fields
    titles = df['title'].fillna('').tolist()
    descriptions = df['description'].fillna('').tolist() 
    features = df['features'].fillna('').tolist()
    categories_text = df['main_category'].fillna('').tolist()
    stores = df['store'].fillna('').tolist()
    
    # Process categories array into text
    categories_list = []
    for cats in df['categories']:
        if isinstance(cats, (list, np.ndarray)) and len(cats) > 0:
            categories_list.append(' '.join([str(c) for c in cats]))
        else:
            categories_list.append('')
    
    # Process details dict into text  
    details_text = []
    for details in df['details']:
        if isinstance(details, dict):
            # Extract key non-null details
            useful_details = []
            for key, value in details.items():
                if value is not None and str(value).strip():
                    useful_details.append(f"{key}: {value}")
            details_text.append(' | '.join(useful_details[:10]))  # Limit to top 10 details
        else:
            details_text.append('')
    
    print(f"Processing {len(df)} records with batch size {batch_size}")
    
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
    
    # Process fields in parallel using ThreadPoolExecutor
    print("Starting parallel embedding generation...")
    field_embeddings = {}
    
    # Use multiple threads for parallel processing
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
    
    # Concatenate embeddings in the correct order
    print("Concatenating embeddings...")
    concatenated_embeddings = np.concatenate([
        field_embeddings['title'],
        field_embeddings['description'],
        field_embeddings['features'],
        field_embeddings['main_category'],
        field_embeddings['store'],
        field_embeddings['categories'],
        field_embeddings['details']
    ], axis=1)
    
    print(f"Final embedding dimensions: {concatenated_embeddings.shape[1]}")
    dims = [field_embeddings[f].shape[1] for f in ['title', 'description', 'features', 'main_category', 'store', 'categories', 'details']]
    print(f"Dimensions breakdown: {dims} = {sum(dims)}")
    
    return concatenated_embeddings

def save_dataset(df, embeddings, output_path):
    """Save the dataset with embeddings to parquet format"""
    print(f"Saving dataset to {output_path}")
    
    # Convert embeddings to list format for parquet storage
    df_with_embeddings = df.copy()
    df_with_embeddings['embedding'] = embeddings.tolist()
    
    # Save to parquet
    df_with_embeddings.to_parquet(output_path, index=False)
    print(f"Saved {len(df_with_embeddings)} records to {output_path}")
    
    return df_with_embeddings

def main():
    parser = argparse.ArgumentParser(description='Process Amazon dataset and generate vector embeddings')
    parser.add_argument('--dataset', type=str, default='Appliances',
                       choices=['Appliances', 'Gift_Cards'],
                       help='Dataset to process (default: Appliances)')
    parser.add_argument('--input', type=str, default=None,
                       help='Input JSONL file path (default: meta_{dataset}.jsonl)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output parquet file path (default: {dataset_lower}_with_embeddings.parquet)')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size for embedding generation (default: 1024)')
    parser.add_argument('--max-workers', type=int, default=12,
                       help='Maximum number of parallel workers (default: 12)')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage instead of GPU acceleration')
    
    args = parser.parse_args()
    
    # Set default paths based on dataset if not provided
    if args.input is None:
        metadata_file = f"meta_{args.dataset}.jsonl"
    else:
        metadata_file = args.input
    
    if args.output is None:
        output_file = f"{args.dataset.lower()}_with_embeddings.parquet"
    else:
        output_file = args.output
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Input file: {metadata_file}")
    print(f"Output file: {output_file}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max workers: {args.max_workers}")
    print(f"Force CPU: {args.force_cpu}")
    
    # Check if input file exists
    if not Path(metadata_file).exists():
        print(f"Error: {metadata_file} not found. Please download it first.")
        print(f"\nTo download from GCS, run:")
        print(f"  gsutil cp gs://superlinked-benchmarks-external/{metadata_file} .")
        return
    
    # Track overall time
    start_time = time.time()
    
    # Load and process data
    records = load_metadata(metadata_file)
    df = prepare_dataframe(records)
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Total records: {len(df)}")
    print(f"Records with descriptions: {df['description'].str.len().gt(0).sum()}")
    print(f"Records with features: {df['features'].str.len().gt(0).sum()}")
    print(f"Records with ratings: {df['average_rating'].notna().sum()}")
    print(f"Records with prices: {df['has_price'].sum()}")
    print(f"Average rating: {df['average_rating'].mean():.2f}")
    print(f"Rating distribution:")
    print(df['rating_tier'].value_counts())
    
    # Load embedding model and generate embeddings
    model_tokenizer_device = load_embedding_model(force_cpu=args.force_cpu)
    embeddings = generate_embeddings(df, model_tokenizer_device, batch_size=args.batch_size, max_workers=args.max_workers)
    
    # Save the final dataset
    final_df = save_dataset(df, embeddings, output_file)
    
    # Calculate total time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nProcessing complete!")
    print(f"Dataset saved as: {output_file}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"File size: {Path(output_file).stat().st_size / (1024*1024):.1f} MB")
    print(f"Total processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

if __name__ == "__main__":
    main()