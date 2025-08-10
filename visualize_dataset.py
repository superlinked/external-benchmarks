#!/usr/bin/env python3
"""
Visualize and analyze parquet dataset files with embeddings using Apple's Embedding Atlas.
Generates both interactive embedding visualizations and static analysis charts.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
import subprocess
import sys
from embedding_atlas.atlas import Atlas
from embedding_atlas.data import Data
warnings.filterwarnings('ignore')

# Set matplotlib backend and style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def load_dataset(parquet_path):
    """Load the parquet dataset"""
    print(f"Loading dataset from {parquet_path}...")
    if parquet_path.suffix == '.csv':
        data = pd.read_csv(parquet_path)
    else:
        data = pd.read_parquet(parquet_path)
    print(f"Loaded {len(data)} records with {len(data.columns)} columns")
    return data

def setup_output_directory(parquet_path):
    """Create output directory based on parquet filename"""
    output_dir = Path(parquet_path.stem + "_analysis")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir

def analyze_basic_stats(df):
    """Generate basic statistics about the dataset"""
    stats = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'null_counts': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    # Analyze text columns
    text_cols = ['title', 'description', 'features']
    for col in text_cols:
        if col in df.columns:
            texts = df[col].fillna('')
            stats[f'{col}_avg_length'] = texts.str.len().mean()
            stats[f'{col}_max_length'] = texts.str.len().max()
            stats[f'{col}_non_empty'] = (texts.str.len() > 0).sum()
    
    return stats

def plot_rating_distribution(df, output_dir):
    """Plot distribution of ratings"""
    if 'average_rating' not in df.columns:
        return None
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Rating Analysis', fontsize=16, fontweight='bold')
    
    # Rating distribution
    ratings = df['average_rating'].dropna()
    axes[0,0].hist(ratings, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Average Rating Distribution')
    axes[0,0].set_xlabel('Average Rating')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axvline(ratings.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean: {ratings.mean():.2f}')
    axes[0,0].legend()
    
    # Rating tier distribution (if exists)
    if 'rating_tier' in df.columns:
        tier_counts = df['rating_tier'].value_counts()
        wedges, texts, autotexts = axes[0,1].pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('Rating Tier Distribution')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        axes[0,1].text(0.5, 0.5, 'No rating_tier column', ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Rating Tier Distribution (N/A)')
    
    # Rating vs Review Count scatter
    if 'rating_number' in df.columns:
        valid_data = df[['average_rating', 'rating_number']].dropna()
        if len(valid_data) > 0:
            axes[1,0].scatter(valid_data['average_rating'], np.log10(valid_data['rating_number'] + 1), 
                           alpha=0.6, s=20, color='coral')
            axes[1,0].set_xlabel('Average Rating')
            axes[1,0].set_ylabel('Log10(Review Count + 1)')
            axes[1,0].set_title('Rating vs Review Count')
            axes[1,0].grid(True, alpha=0.3)
    else:
        axes[1,0].text(0.5, 0.5, 'No rating_number column', ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Rating vs Review Count (N/A)')
    
    # Rating statistics box
    stats_text = f"""Rating Statistics:
    Mean: {ratings.mean():.2f}
    Median: {ratings.median():.2f}
    Std: {ratings.std():.2f}
    Min: {ratings.min():.1f}
    Max: {ratings.max():.1f}
    Count: {len(ratings):,}"""
    
    axes[1,1].text(0.1, 0.5, stats_text, transform=axes[1,1].transAxes, fontsize=12, 
                   verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    axes[1,1].set_title('Rating Statistics Summary')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rating_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 'rating_analysis.png'

def plot_price_analysis(df, output_dir):
    """Plot comprehensive price analysis"""
    if 'price' not in df.columns:
        return None
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Price Analysis', fontsize=16, fontweight='bold')
    
    prices = df['price'].dropna()
    
    if len(prices) > 0:
        # Price distribution (log scale)
        log_prices = np.log10(prices + 1)
        axes[0,0].hist(log_prices, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,0].set_title('Price Distribution (Log10)')
        axes[0,0].set_xlabel('Log10(Price + 1)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axvline(log_prices.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean: {log_prices.mean():.2f}')
        axes[0,0].legend()
        
        # Price box plot by price ranges
        price_ranges = pd.cut(prices, bins=[0, 10, 25, 50, 100, 500, float('inf')], 
                             labels=['$0-10', '$10-25', '$25-50', '$50-100', '$100-500', '$500+'])
        range_counts = price_ranges.value_counts()
        axes[0,1].bar(range_counts.index.astype(str), range_counts.values, color='lightcoral', alpha=0.7)
        axes[0,1].set_title('Products by Price Range')
        axes[0,1].set_xlabel('Price Range')
        axes[0,1].set_ylabel('Count')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
    
    # Price vs Rating scatter
    if 'average_rating' in df.columns and len(prices) > 0:
        valid_data = df[['price', 'average_rating']].dropna()
        if len(valid_data) > 0:
            axes[0,2].scatter(valid_data['average_rating'], np.log10(valid_data['price'] + 1), 
                           alpha=0.6, s=20, color='purple')
            axes[0,2].set_xlabel('Average Rating')
            axes[0,2].set_ylabel('Log10(Price + 1)')
            axes[0,2].set_title('Price vs Rating Correlation')
            axes[0,2].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = valid_data['price'].corr(valid_data['average_rating'])
            axes[0,2].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[0,2].transAxes,
                         bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # Price availability pie chart
    has_price = df['price'].notna().sum()
    no_price = df['price'].isna().sum()
    wedges, texts, autotexts = axes[1,0].pie([has_price, no_price], labels=['Has Price', 'No Price'], 
            autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    axes[1,0].set_title('Price Data Availability')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Price statistics
    if len(prices) > 0:
        stats_text = f"""Price Statistics:
        Mean: ${prices.mean():.2f}
        Median: ${prices.median():.2f}
        Std: ${prices.std():.2f}
        Min: ${prices.min():.2f}
        Max: ${prices.max():.2f}
        Q25: ${prices.quantile(0.25):.2f}
        Q75: ${prices.quantile(0.75):.2f}
        Count: {len(prices):,}"""
        
        axes[1,1].text(0.1, 0.5, stats_text, transform=axes[1,1].transAxes, fontsize=12, 
                       verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        axes[1,1].set_title('Price Statistics Summary')
        axes[1,1].axis('off')
        
        # Price trend by category (if available)
        if 'main_category' in df.columns:
            cat_prices = df.groupby('main_category')['price'].median().sort_values(ascending=False).head(10)
            axes[1,2].barh(range(len(cat_prices)), cat_prices.values, color='gold', alpha=0.7)
            axes[1,2].set_yticks(range(len(cat_prices)))
            axes[1,2].set_yticklabels(cat_prices.index, fontsize=10)
            axes[1,2].set_title('Top 10 Categories by Median Price')
            axes[1,2].set_xlabel('Median Price ($)')
            axes[1,2].grid(True, alpha=0.3)
        else:
            axes[1,2].text(0.5, 0.5, 'No category data available', ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Price by Category (N/A)')
    else:
        axes[1,1].text(0.5, 0.5, 'No price data available', ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Price Statistics (N/A)')
        axes[1,2].text(0.5, 0.5, 'No price data available', ha='center', va='center', transform=axes[1,2].transAxes)
        axes[1,2].set_title('Price by Category (N/A)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'price_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 'price_analysis.png'

def plot_text_analysis(df, output_dir):
    """Comprehensive text field analysis"""
    text_cols = ['title', 'description', 'features']
    available_cols = [col for col in text_cols if col in df.columns]
    
    if not available_cols:
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Text Content Analysis', fontsize=16, fontweight='bold')
    
    # Length distributions for each text field
    for i, col in enumerate(available_cols[:3]):
        lengths = df[col].fillna('').str.len()
        
        # Length histogram
        if i < 3:
            axes[0, i].hist(lengths, bins=50, alpha=0.7, edgecolor='black', color=plt.cm.Set3(i))
            axes[0, i].set_title(f'{col.title()} Length Distribution')
            axes[0, i].set_xlabel('Character Count')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_len = lengths.mean()
            max_len = lengths.max()
            axes[0, i].axvline(mean_len, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_len:.0f}')
            axes[0, i].text(0.7, 0.8, f'Mean: {mean_len:.1f}\nMax: {max_len}\nNon-empty: {(lengths > 0).sum():,}', 
                           transform=axes[0, i].transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
            axes[0, i].legend()
    
    # Fill empty subplots
    for i in range(len(available_cols), 3):
        axes[0, i].text(0.5, 0.5, f'Text field {i+1}\nnot available', ha='center', va='center', transform=axes[0, i].transAxes)
        axes[0, i].set_title(f'Text Field {i+1} (N/A)')
        axes[0, i].axis('off')
    
    # Text field completeness comparison
    completeness_data = []
    for col in available_cols:
        non_empty = (df[col].fillna('').str.len() > 0).sum()
        completeness_data.append((col.title(), non_empty, len(df) - non_empty))
    
    if completeness_data:
        fields, filled, empty = zip(*completeness_data)
        x = np.arange(len(fields))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, filled, width, label='Non-empty', alpha=0.8, color='lightgreen')
        axes[1, 0].bar(x + width/2, empty, width, label='Empty', alpha=0.8, color='lightcoral')
        axes[1, 0].set_xlabel('Text Fields')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Text Field Completeness')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(fields, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Word count analysis (for first available text field)
    if available_cols:
        col = available_cols[0]
        word_counts = df[col].fillna('').str.split().str.len()
        axes[1, 1].hist(word_counts, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 1].set_title(f'Word Count Distribution ({col.title()})')
        axes[1, 1].set_xlabel('Word Count')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(word_counts.mean(), color='red', linestyle='--', alpha=0.7, 
                          label=f'Mean: {word_counts.mean():.1f}')
        axes[1, 1].legend()
    
    # Text length comparison across fields
    if len(available_cols) > 1:
        length_data = []
        for col in available_cols:
            lengths = df[col].fillna('').str.len()
            length_data.extend([(col.title(), length) for length in lengths if length > 0])
        
        if length_data:
            fields_for_box, lengths_for_box = zip(*length_data)
            df_box = pd.DataFrame({'Field': fields_for_box, 'Length': lengths_for_box})
            
            # Create box plot
            field_names = df_box['Field'].unique()
            box_data = [df_box[df_box['Field'] == field]['Length'].values for field in field_names]
            
            box_plot = axes[1, 2].boxplot(box_data, labels=field_names, patch_artist=True)
            axes[1, 2].set_title('Text Length Distribution by Field')
            axes[1, 2].set_xlabel('Text Field')
            axes[1, 2].set_ylabel('Character Count (Log Scale)')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].tick_params(axis='x', rotation=45)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
    else:
        axes[1, 2].text(0.5, 0.5, 'Need multiple text fields\nfor comparison', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Text Length Comparison (N/A)')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'text_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 'text_analysis.png'

def plot_category_analysis(df, output_dir):
    """Comprehensive category analysis"""
    category_cols = ['main_category', 'store']
    available_cols = [col for col in category_cols if col in df.columns]
    
    if not available_cols:
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Category & Store Analysis', fontsize=16, fontweight='bold')
    
    plot_idx = 0
    for col in available_cols[:2]:
        if col in df.columns:
            # Top categories/stores
            top_items = df[col].value_counts().head(15)
            
            if plot_idx < 2:
                top_items.plot(kind='bar', ax=axes[0, plot_idx], color=plt.cm.Set3(plot_idx))
                axes[0, plot_idx].set_title(f'Top 15 {col.replace("_", " ").title()}')
                axes[0, plot_idx].set_xlabel(col.replace("_", " ").title())
                axes[0, plot_idx].set_ylabel('Count')
                axes[0, plot_idx].tick_params(axis='x', rotation=45)
                axes[0, plot_idx].grid(True, alpha=0.3)
                
                # Add percentage labels
                total = len(df)
                for i, v in enumerate(top_items.values):
                    axes[0, plot_idx].text(i, v + total * 0.01, f'{v/total*100:.1f}%', 
                                         ha='center', va='bottom', fontsize=8)
            
            plot_idx += 1
    
    # Fill empty subplot if needed
    for i in range(len(available_cols), 2):
        axes[0, i].text(0.5, 0.5, f'Category field {i+1}\nnot available', ha='center', va='center', transform=axes[0, i].transAxes)
        axes[0, i].set_title(f'Category Field {i+1} (N/A)')
        axes[0, i].axis('off')
    
    # Category diversity comparison
    diversity_stats = []
    for col in available_cols:
        unique_count = df[col].nunique()
        diversity_stats.append((col.replace("_", " ").title(), unique_count))
    
    if diversity_stats:
        cols, counts = zip(*diversity_stats)
        bars = axes[0, 2].bar(cols, counts, color=['skyblue', 'lightcoral'][:len(cols)], alpha=0.8)
        axes[0, 2].set_title('Category Diversity')
        axes[0, 2].set_ylabel('Unique Categories')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01, 
                          f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Categories by rating (if available)
    if 'average_rating' in df.columns and 'main_category' in df.columns:
        category_ratings = df.groupby('main_category').agg({
            'average_rating': ['mean', 'count']
        }).round(2)
        category_ratings.columns = ['avg_rating', 'count']
        category_ratings = category_ratings[category_ratings['count'] >= 10]  # Filter categories with <10 products
        category_ratings = category_ratings.sort_values('avg_rating', ascending=False).head(10)
        
        bars = axes[1, 0].barh(range(len(category_ratings)), category_ratings['avg_rating'].values, color='lightgreen', alpha=0.8)
        axes[1, 0].set_yticks(range(len(category_ratings)))
        axes[1, 0].set_yticklabels(category_ratings.index, fontsize=10)
        axes[1, 0].set_title('Top 10 Categories by Average Rating\n(Min 10 products)')
        axes[1, 0].set_xlabel('Average Rating')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add rating values to bars
        for i, (bar, rating) in enumerate(zip(bars, category_ratings['avg_rating'].values)):
            axes[1, 0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                          f'{rating:.2f}', va='center', fontsize=9)
    
    # Price by category (if available)
    if 'price' in df.columns and 'main_category' in df.columns:
        category_prices = df.groupby('main_category').agg({
            'price': ['median', 'count']
        }).round(2)
        category_prices.columns = ['median_price', 'count']
        category_prices = category_prices.dropna()
        category_prices = category_prices[category_prices['count'] >= 5]  # Filter categories with <5 priced products
        category_prices = category_prices.sort_values('median_price', ascending=False).head(10)
        
        if len(category_prices) > 0:
            bars = axes[1, 1].barh(range(len(category_prices)), category_prices['median_price'].values, color='gold', alpha=0.8)
            axes[1, 1].set_yticks(range(len(category_prices)))
            axes[1, 1].set_yticklabels(category_prices.index, fontsize=10)
            axes[1, 1].set_title('Top 10 Categories by Median Price\n(Min 5 priced products)')
            axes[1, 1].set_xlabel('Median Price ($)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add price values to bars
            for i, (bar, price) in enumerate(zip(bars, category_prices['median_price'].values)):
                axes[1, 1].text(bar.get_width() + max(category_prices['median_price']) * 0.01, 
                               bar.get_y() + bar.get_height()/2, f'${price:.0f}', va='center', fontsize=9)
        else:
            axes[1, 1].text(0.5, 0.5, 'Not enough price data\nby category', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Price by Category (Insufficient Data)')
    else:
        axes[1, 1].text(0.5, 0.5, 'Price or category data\nnot available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Price by Category (N/A)')
        axes[1, 1].axis('off')
    
    # Category statistics summary
    stats_text = "Category Statistics:\n\n"
    for col in available_cols:
        unique_count = df[col].nunique()
        most_common = df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else "N/A"
        most_common_count = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
        
        stats_text += f"{col.replace('_', ' ').title()}:\n"
        stats_text += f"  Unique: {unique_count:,}\n"
        stats_text += f"  Most common: {most_common}\n"
        stats_text += f"  Count: {most_common_count:,} ({most_common_count/len(df)*100:.1f}%)\n\n"
    
    axes[1, 2].text(0.1, 0.95, stats_text, transform=axes[1, 2].transAxes, fontsize=11, 
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.8))
    axes[1, 2].set_title('Category Statistics Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 'category_analysis.png'

def create_embedding_atlas(df, output_dir):
    """Create interactive embedding visualization using Apple's Embedding Atlas"""
    if 'embedding' not in df.columns:
        print("No embeddings found - skipping Atlas creation")
        return None
    
    print("Creating interactive embedding atlas...")
    
    # Prepare embeddings
    embeddings = np.array(df['embedding'].tolist())
    print(f"Processing embeddings with shape: {embeddings.shape}")
    
    # Prepare metadata for atlas
    metadata_columns = []
    for col in ['title', 'description', 'average_rating', 'price', 'main_category', 'store']:
        if col in df.columns:
            metadata_columns.append(col)
    
    # Create metadata dataframe
    atlas_df = df[metadata_columns].copy()
    
    # Clean and prepare text fields for atlas
    for col in ['title', 'description']:
        if col in atlas_df.columns:
            atlas_df[col] = atlas_df[col].fillna('').astype(str)
            # Truncate very long text
            atlas_df[col] = atlas_df[col].str[:200]
    
    # Handle missing values
    for col in atlas_df.columns:
        if atlas_df[col].dtype == 'object':
            atlas_df[col] = atlas_df[col].fillna('Unknown')
        else:
            atlas_df[col] = atlas_df[col].fillna(0)
    
    try:
        # Create Data object for Atlas
        data = Data(
            embeddings=embeddings,
            dataframe=atlas_df
        )
        
        # Create and export Atlas
        atlas = Atlas(data=data)
        atlas_path = output_dir / 'embedding_atlas'
        
        # Export the atlas
        atlas.export(path=str(atlas_path))
        print(f"Interactive Atlas created at: {atlas_path}")
        
        # Create a simple HTML file that links to the atlas
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Embedding Atlas - {output_dir.name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .atlas-link {{
            display: block;
            text-align: center;
            background-color: #007AFF;
            color: white;
            padding: 15px 30px;
            text-decoration: none;
            border-radius: 5px;
            margin: 20px 0;
            font-size: 18px;
        }}
        .atlas-link:hover {{
            background-color: #0056CC;
        }}
        .info {{
            background-color: #E8F4FD;
            padding: 15px;
            border-left: 4px solid #007AFF;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Interactive Embedding Atlas</h1>
        
        <div class="info">
            <strong>Dataset:</strong> {len(df):,} records with {embeddings.shape[1]}-dimensional embeddings<br>
            <strong>Metadata fields:</strong> {', '.join(metadata_columns)}
        </div>
        
        <p>This interactive visualization allows you to explore the high-dimensional embedding space using Apple's Embedding Atlas.</p>
        
        <a href="embedding_atlas/index.html" class="atlas-link">
            🚀 Launch Interactive Atlas
        </a>
        
        <h3>Features:</h3>
        <ul>
            <li>Interactive 2D/3D visualization of embeddings</li>
            <li>Search and filter by metadata</li>
            <li>Cluster analysis and exploration</li>
            <li>Similarity search</li>
            <li>Hover for detailed information</li>
        </ul>
        
        <p><em>Note: Make sure to serve this directory with a web server (e.g., python -m http.server) to view the interactive atlas properly.</em></p>
    </div>
</body>
</html>
"""
        
        with open(output_dir / 'atlas_viewer.html', 'w') as f:
            f.write(html_content)
            
        return 'atlas_viewer.html'
        
    except Exception as e:
        print(f"Error creating Atlas: {e}")
        return None

def generate_markdown_report(df, stats, charts, output_dir, parquet_path):
    """Generate comprehensive markdown report"""
    report_path = output_dir / 'README.md'
    
    with open(report_path, 'w') as f:
        f.write(f"# 📊 Dataset Analysis Report\n\n")
        f.write(f"**Dataset:** `{parquet_path.name}`  \n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Analysis Directory:** `{output_dir.name}/`\n\n")
        
        # Quick Stats Box
        f.write("## 🎯 Quick Overview\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| 📦 **Total Records** | {stats['total_records']:,} |\n")
        f.write(f"| 📋 **Columns** | {stats['total_columns']} |\n")
        f.write(f"| 💾 **Memory Usage** | {stats['memory_usage_mb']:.1f} MB |\n")
        f.write(f"| 📁 **File Size** | {parquet_path.stat().st_size / 1024**2:.1f} MB |\n")
        
        if 'embedding' in df.columns:
            embedding_dim = len(df['embedding'].iloc[0]) if len(df) > 0 else 0
            f.write(f"| 🧠 **Embedding Dimension** | {embedding_dim} |\n")
        
        f.write("\n")
        
        # Interactive Atlas Section
        if 'atlas_viewer.html' in charts.values():
            f.write("## 🌟 Interactive Embedding Visualization\n\n")
            f.write("🚀 **[Launch Interactive Atlas](atlas_viewer.html)** - Explore embeddings in 2D/3D space with Apple's Embedding Atlas\n\n")
            f.write("**Features:**\n")
            f.write("- Interactive scatter plots with zoom and pan\n")
            f.write("- Search and filter by metadata\n")
            f.write("- Cluster analysis and exploration\n")
            f.write("- Similarity search functionality\n")
            f.write("- Hover tooltips with detailed information\n\n")
            f.write("> **Note:** Serve this directory with a web server (`python -m http.server`) for best experience.\n\n")
        
        # Dataset Schema
        f.write("## 📋 Dataset Schema\n\n")
        f.write("| Column | Type | Non-Null | Missing % | Description |\n")
        f.write("|--------|------|----------|-----------|-------------|\n")
        
        for col, dtype in stats['dtypes'].items():
            null_count = stats['null_counts'][col]
            null_pct = (null_count / stats['total_records']) * 100
            non_null = stats['total_records'] - null_count
            
            # Add description based on column name
            descriptions = {
                'parent_asin': 'Product identifier',
                'title': 'Product title/name',
                'description': 'Product description',
                'features': 'Product features list',
                'average_rating': 'Average customer rating (1-5)',
                'rating_number': 'Number of reviews',
                'price': 'Product price in USD',
                'main_category': 'Primary product category',
                'store': 'Seller/brand name',
                'categories': 'Full category hierarchy',
                'details': 'Additional product details',
                'embedding': 'Vector embedding representation'
            }
            description = descriptions.get(col, 'Data field')
            
            f.write(f"| `{col}` | {dtype} | {non_null:,} | {null_pct:.1f}% | {description} |\n")
        
        # Text Field Analysis
        text_stats_found = False
        for col in ['title', 'description', 'features']:
            if f'{col}_avg_length' in stats:
                if not text_stats_found:
                    f.write("\n## 📝 Text Content Analysis\n\n")
                    f.write("| Field | Avg Length | Max Length | Non-Empty | Completeness |\n")
                    f.write("|-------|------------|------------|-----------|---------------|\n")
                    text_stats_found = True
                
                non_empty = stats[f'{col}_non_empty']
                completeness = (non_empty / stats['total_records']) * 100
                f.write(f"| **{col.title()}** | {stats[f'{col}_avg_length']:.0f} chars | {stats[f'{col}_max_length']:,} chars | {non_empty:,} | {completeness:.1f}% |\n")
        
        # Key Insights
        f.write("\n## 🔍 Key Insights\n\n")
        
        # Rating insights
        if 'average_rating' in df.columns:
            ratings = df['average_rating'].dropna()
            f.write("### ⭐ Rating Analysis\n")
            f.write(f"- **{len(ratings):,}** products have ratings ({len(ratings)/len(df)*100:.1f}% coverage)\n")
            f.write(f"- **Average rating:** {ratings.mean():.2f} out of 5.0\n")
            f.write(f"- **Rating distribution:** {ratings.min():.1f} to {ratings.max():.1f}\n")
            f.write(f"- **Most common rating tier:** {df['rating_tier'].mode().iloc[0] if 'rating_tier' in df.columns and len(df['rating_tier'].mode()) > 0 else 'N/A'}\n\n")
        
        # Price insights
        if 'price' in df.columns:
            prices = df['price'].dropna()
            f.write("### 💰 Pricing Analysis\n")
            f.write(f"- **{len(prices):,}** products have pricing ({len(prices)/len(df)*100:.1f}% coverage)\n")
            if len(prices) > 0:
                f.write(f"- **Price range:** ${prices.min():.2f} - ${prices.max():.2f}\n")
                f.write(f"- **Median price:** ${prices.median():.2f}\n")
                f.write(f"- **Average price:** ${prices.mean():.2f}\n\n")
        
        # Category insights
        if 'main_category' in df.columns:
            categories = df['main_category'].value_counts()
            f.write("### 📦 Category Distribution\n")
            f.write(f"- **{len(categories)}** unique categories\n")
            f.write(f"- **Largest category:** {categories.index[0]} ({categories.iloc[0]:,} products, {categories.iloc[0]/len(df)*100:.1f}%)\n")
            if 'store' in df.columns:
                stores = df['store'].value_counts()
                f.write(f"- **{len(stores)}** unique stores/brands\n")
                f.write(f"- **Largest store:** {stores.index[0]} ({stores.iloc[0]:,} products)\n\n")
        
        # Embedding insights
        if 'embedding' in df.columns:
            embeddings = np.array(df['embedding'].tolist())
            norms = np.linalg.norm(embeddings, axis=1)
            f.write("### 🧠 Embedding Analysis\n")
            f.write(f"- **Embedding dimensions:** {embeddings.shape[1]}\n")
            f.write(f"- **Vector norms:** {norms.mean():.3f} ± {norms.std():.3f}\n")
            f.write(f"- **Suitable for:** Semantic search, clustering, similarity analysis\n\n")
        
        # Data Quality Assessment
        f.write("## ✅ Data Quality Assessment\n\n")
        
        completeness = {}
        for col in df.columns:
            if col != 'embedding':
                completeness[col] = (1 - stats['null_counts'][col] / stats['total_records']) * 100
        
        high_quality = [col for col, score in completeness.items() if score >= 90]
        medium_quality = [col for col, score in completeness.items() if 50 <= score < 90]
        low_quality = [col for col, score in completeness.items() if score < 50]
        
        f.write("| Quality Level | Fields | Count |\n")
        f.write("|---------------|--------|-------|\n")
        f.write(f"| 🟢 **High (≥90%)** | {', '.join([f'`{col}`' for col in high_quality]) if high_quality else 'None'} | {len(high_quality)} |\n")
        f.write(f"| 🟡 **Medium (50-89%)** | {', '.join([f'`{col}`' for col in medium_quality]) if medium_quality else 'None'} | {len(medium_quality)} |\n")
        f.write(f"| 🔴 **Low (<50%)** | {', '.join([f'`{col}`' for col in low_quality]) if low_quality else 'None'} | {len(low_quality)} |\n")
        f.write("\n")
        
        # Visualizations Section
        f.write("## 📈 Static Analysis Charts\n\n")
        
        chart_descriptions = {
            'rating_analysis': '⭐ **Rating Analysis** - Distribution of product ratings and review counts',
            'price_analysis': '💰 **Price Analysis** - Price distributions, ranges, and correlations',
            'text_analysis': '📝 **Text Analysis** - Content length distributions and completeness metrics',
            'category_analysis': '📦 **Category Analysis** - Product categories, stores, and market distribution'
        }
        
        for chart_name, chart_file in charts.items():
            if chart_file and chart_file.endswith('.png'):
                description = chart_descriptions.get(chart_name, f'📊 **{chart_name.replace("_", " ").title()}**')
                f.write(f"### {description}\n\n")
                f.write(f"![{chart_name}]({chart_file})\n\n")
        
        # Usage Recommendations
        f.write("## 🚀 Usage Recommendations\n\n")
        
        recommendations = []
        
        if 'embedding' in df.columns:
            recommendations.append("🔍 **Semantic Search**: Use embeddings for similarity-based product search and recommendations")
            recommendations.append("🎯 **Clustering**: Group similar products using embedding vectors for market analysis")
        
        if 'average_rating' in df.columns and 'price' in df.columns:
            recommendations.append("📊 **Market Analysis**: Analyze price-rating relationships for competitive insights")
        
        if text_stats_found:
            recommendations.append("🏷️ **Text Mining**: Extract insights from product descriptions and features")
        
        if 'main_category' in df.columns:
            recommendations.append("📈 **Category Analytics**: Perform category-wise performance analysis")
        
        if low_quality:
            recommendations.append(f"⚠️ **Data Enhancement**: Improve data collection for: {', '.join([f'`{col}`' for col in low_quality])}")
        
        recommendations.extend([
            "🤖 **ML Applications**: Build recommendation systems, price prediction models",
            "🔗 **Vector Database**: Load embeddings into vector databases (Pinecone, Weaviate, Qdrant)",
            "📱 **Search Applications**: Implement semantic search for e-commerce platforms"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
        
        f.write("\n")
        
        # Technical Notes
        f.write("## 🔧 Technical Notes\n\n")
        f.write("- **Analysis Framework**: Python with pandas, matplotlib, seaborn, Apple Embedding Atlas\n")
        f.write("- **Embedding Visualization**: Interactive 2D/3D projections using UMAP and t-SNE\n")
        f.write("- **Statistical Analysis**: Comprehensive descriptive statistics and data quality metrics\n")
        f.write("- **Export Formats**: Static PNG charts + interactive HTML visualizations\n\n")
        
        f.write("---\n")
        f.write(f"*Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*\n")
    
    print(f"Comprehensive report generated: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Comprehensive dataset visualization with Apple Embedding Atlas')
    parser.add_argument('parquet_file', type=str, help='Path to the parquet file to analyze')
    parser.add_argument('--sample', type=int, default=None, 
                       help='Sample N records for faster analysis (recommended for >50k records)')
    parser.add_argument('--skip-atlas', action='store_true', 
                       help='Skip creating interactive atlas (faster for large datasets)')
    
    args = parser.parse_args()
    
    # Validate input file
    parquet_path = Path(args.parquet_file)
    if not parquet_path.exists():
        print(f"❌ Error: File {parquet_path} not found")
        sys.exit(1)
    
    print(f"🔍 Analyzing dataset: {parquet_path.name}")
    print(f"📁 File size: {parquet_path.stat().st_size / 1024**2:.1f} MB")
    
    # Setup output directory
    output_dir = setup_output_directory(parquet_path)
    
    # Load data
    df = load_dataset(parquet_path)
    
    # Sample data if requested
    if args.sample and len(df) > args.sample:
        print(f"📊 Sampling {args.sample:,} records from {len(df):,} total records")
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
    
    # Generate basic statistics
    print("📈 Computing dataset statistics...")
    stats = analyze_basic_stats(df)
    
    # Generate static visualizations
    print("🎨 Creating static visualizations...")
    charts = {}
    
    chart_functions = [
        ('rating_analysis', plot_rating_distribution),
        ('price_analysis', plot_price_analysis), 
        ('text_analysis', plot_text_analysis),
        ('category_analysis', plot_category_analysis)
    ]
    
    for chart_name, chart_func in chart_functions:
        try:
            result = chart_func(df, output_dir)
            if result:
                charts[chart_name] = result
                print(f"  ✅ {chart_name.replace('_', ' ').title()}")
            else:
                print(f"  ⚠️  {chart_name.replace('_', ' ').title()} - insufficient data")
        except Exception as e:
            print(f"  ❌ {chart_name.replace('_', ' ').title()} - error: {e}")
    
    # Create interactive embedding atlas
    if not args.skip_atlas:
        print("🌟 Creating interactive embedding atlas...")
        try:
            atlas_result = create_embedding_atlas(df, output_dir)
            if atlas_result:
                charts['interactive_atlas'] = atlas_result
                print("  ✅ Interactive Atlas created successfully")
            else:
                print("  ⚠️  Atlas creation skipped - no embeddings found")
        except Exception as e:
            print(f"  ❌ Atlas creation failed: {e}")
    else:
        print("⚠️  Skipping interactive atlas creation (--skip-atlas flag)")
    
    # Generate comprehensive markdown report
    print("📝 Generating comprehensive report...")
    report_path = generate_markdown_report(df, stats, charts, output_dir, parquet_path)
    
    # Final summary
    print(f"\n🎉 Analysis Complete!")
    print(f"📊 Dataset: {len(df):,} records analyzed")
    print(f"📈 Charts: {len([c for c in charts.values() if c and c.endswith('.png')])} static visualizations")
    print(f"🌟 Atlas: {'✅ Created' if 'interactive_atlas' in charts else '❌ Not created'}")
    print(f"📁 Output: {output_dir}/")
    print(f"📖 Report: {report_path}")
    
    if 'interactive_atlas' in charts:
        print(f"🚀 Interactive: Open {output_dir / 'atlas_viewer.html'} in browser")
        print("   (Serve with: python -m http.server)")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Open {report_path} for detailed insights")
    print(f"   2. Review static charts in {output_dir}/")
    if 'interactive_atlas' in charts:
        print(f"   3. Explore embeddings interactively via atlas_viewer.html")

if __name__ == "__main__":
    main()