#!/usr/bin/env python3
"""
Simple dataset visualization tool that creates static charts and analysis reports.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_report_directory(parquet_path):
    """Create report directory based on parquet filename"""
    parquet_name = Path(parquet_path).stem
    report_dir = Path(parquet_name)
    report_dir.mkdir(exist_ok=True)
    return report_dir

def load_and_analyze_data(parquet_path, sample_size=None):
    """Load parquet data and perform basic analysis"""
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size} records from {len(df)} total records...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"Dataset shape: {df.shape}")
    return df

def create_rating_analysis(df, report_dir):
    """Create rating analysis charts"""
    if 'average_rating' not in df.columns or df['average_rating'].isna().all():
        print("No rating data available for analysis")
        return None
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Rating Analysis', fontsize=16, fontweight='bold')
    
    # Rating distribution
    axes[0,0].hist(df['average_rating'].dropna(), bins=20, edgecolor='black', alpha=0.7)
    axes[0,0].set_title('Rating Distribution')
    axes[0,0].set_xlabel('Average Rating')
    axes[0,0].set_ylabel('Count')
    
    # Rating vs Number of Reviews
    if 'rating_number' in df.columns:
        valid_data = df.dropna(subset=['average_rating', 'rating_number'])
        if len(valid_data) > 0:
            axes[0,1].scatter(valid_data['rating_number'], valid_data['average_rating'], alpha=0.5)
            axes[0,1].set_title('Rating vs Number of Reviews')
            axes[0,1].set_xlabel('Number of Reviews')
            axes[0,1].set_ylabel('Average Rating')
            axes[0,1].set_xscale('log')
    
    # Rating tiers
    if 'rating_tier' in df.columns:
        rating_counts = df['rating_tier'].value_counts()
        axes[1,0].bar(rating_counts.index, rating_counts.values)
        axes[1,0].set_title('Rating Tiers')
        axes[1,0].set_xlabel('Rating Tier')
        axes[1,0].set_ylabel('Count')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # Review volume
    if 'review_volume' in df.columns:
        volume_counts = df['review_volume'].value_counts()
        axes[1,1].pie(volume_counts.values, labels=volume_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Review Volume Distribution')
    
    plt.tight_layout()
    rating_path = report_dir / 'rating_analysis.png'
    plt.savefig(rating_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mean_rating': df['average_rating'].mean(),
        'median_rating': df['average_rating'].median(),
        'rating_std': df['average_rating'].std(),
        'total_reviews': df['rating_number'].sum() if 'rating_number' in df.columns else None
    }

def create_price_analysis(df, report_dir):
    """Create price analysis charts"""
    if 'price' not in df.columns or df['price'].isna().all():
        print("No price data available for analysis")
        return None
        
    price_data = df['price'].dropna()
    if len(price_data) == 0:
        return None
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Price Analysis', fontsize=16, fontweight='bold')
    
    # Price distribution
    axes[0,0].hist(price_data, bins=30, edgecolor='black', alpha=0.7)
    axes[0,0].set_title('Price Distribution')
    axes[0,0].set_xlabel('Price ($)')
    axes[0,0].set_ylabel('Count')
    
    # Log price distribution
    log_prices = np.log10(price_data[price_data > 0])
    axes[0,1].hist(log_prices, bins=30, edgecolor='black', alpha=0.7)
    axes[0,1].set_title('Price Distribution (Log Scale)')
    axes[0,1].set_xlabel('Log10(Price)')
    axes[0,1].set_ylabel('Count')
    
    # Price vs Rating
    if 'average_rating' in df.columns:
        valid_data = df.dropna(subset=['price', 'average_rating'])
        if len(valid_data) > 0:
            axes[1,0].scatter(valid_data['price'], valid_data['average_rating'], alpha=0.5)
            axes[1,0].set_title('Price vs Rating')
            axes[1,0].set_xlabel('Price ($)')
            axes[1,0].set_ylabel('Average Rating')
            axes[1,0].set_xscale('log')
    
    # Price range distribution
    price_ranges = pd.cut(price_data, bins=[0, 10, 50, 100, 500, float('inf')], 
                         labels=['<$10', '$10-50', '$50-100', '$100-500', '>$500'])
    range_counts = price_ranges.value_counts()
    axes[1,1].bar(range_counts.index, range_counts.values)
    axes[1,1].set_title('Price Range Distribution')
    axes[1,1].set_xlabel('Price Range')
    axes[1,1].set_ylabel('Count')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    price_path = report_dir / 'price_analysis.png'
    plt.savefig(price_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mean_price': price_data.mean(),
        'median_price': price_data.median(),
        'price_std': price_data.std(),
        'min_price': price_data.min(),
        'max_price': price_data.max(),
        'priced_items': len(price_data),
        'price_coverage': len(price_data) / len(df) * 100
    }

def create_text_analysis(df, report_dir):
    """Create text analysis charts"""
    text_stats = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Text Analysis', fontsize=16, fontweight='bold')
    
    # Title length distribution
    if 'title' in df.columns:
        title_lengths = df['title'].fillna('').str.len()
        axes[0,0].hist(title_lengths, bins=30, edgecolor='black', alpha=0.7)
        axes[0,0].set_title('Title Length Distribution')
        axes[0,0].set_xlabel('Characters')
        axes[0,0].set_ylabel('Count')
        text_stats['avg_title_length'] = title_lengths.mean()
    
    # Description length distribution
    if 'description' in df.columns:
        desc_lengths = df['description'].fillna('').str.len()
        axes[0,1].hist(desc_lengths, bins=30, edgecolor='black', alpha=0.7)
        axes[0,1].set_title('Description Length Distribution')
        axes[0,1].set_xlabel('Characters')
        axes[0,1].set_ylabel('Count')
        text_stats['avg_desc_length'] = desc_lengths.mean()
    
    # Features availability
    if 'features' in df.columns:
        has_features = df['features'].fillna('').str.len() > 0
        feature_counts = ['Has Features', 'No Features']
        feature_values = [has_features.sum(), (~has_features).sum()]
        axes[1,0].pie(feature_values, labels=feature_counts, autopct='%1.1f%%')
        axes[1,0].set_title('Features Availability')
        text_stats['features_coverage'] = has_features.sum() / len(df) * 100
    
    # Store distribution (top 10)
    if 'store' in df.columns:
        store_counts = df['store'].value_counts().head(10)
        if len(store_counts) > 0:
            axes[1,1].barh(range(len(store_counts)), store_counts.values)
            axes[1,1].set_yticks(range(len(store_counts)))
            axes[1,1].set_yticklabels(store_counts.index)
            axes[1,1].set_title('Top 10 Stores')
            axes[1,1].set_xlabel('Product Count')
    
    plt.tight_layout()
    text_path = report_dir / 'text_analysis.png'
    plt.savefig(text_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return text_stats

def create_category_analysis(df, report_dir):
    """Create category analysis charts"""
    if 'main_category' not in df.columns:
        print("No category data available for analysis")
        return None
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Category Analysis', fontsize=16, fontweight='bold')
    
    # Main categories
    category_counts = df['main_category'].value_counts().head(15)
    axes[0].barh(range(len(category_counts)), category_counts.values)
    axes[0].set_yticks(range(len(category_counts)))
    axes[0].set_yticklabels(category_counts.index, fontsize=10)
    axes[0].set_title('Top Categories')
    axes[0].set_xlabel('Product Count')
    
    # Category vs Rating
    if 'average_rating' in df.columns:
        category_ratings = df.groupby('main_category')['average_rating'].mean().sort_values(ascending=False).head(10)
        axes[1].bar(range(len(category_ratings)), category_ratings.values)
        axes[1].set_xticks(range(len(category_ratings)))
        axes[1].set_xticklabels(category_ratings.index, rotation=45, ha='right', fontsize=8)
        axes[1].set_title('Average Rating by Category (Top 10)')
        axes[1].set_ylabel('Average Rating')
    
    plt.tight_layout()
    category_path = report_dir / 'category_analysis.png'
    plt.savefig(category_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'unique_categories': df['main_category'].nunique(),
        'top_category': category_counts.index[0] if len(category_counts) > 0 else None,
        'top_category_count': category_counts.iloc[0] if len(category_counts) > 0 else None
    }

def create_embedding_analysis(df, report_dir):
    """Create embedding analysis"""
    if 'embedding' not in df.columns:
        print("No embedding data available for analysis")
        return None
    
    # Convert embeddings to numpy array
    embeddings = np.array(df['embedding'].tolist())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Embedding Analysis', fontsize=16, fontweight='bold')
    
    # Embedding dimension analysis
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    axes[0,0].hist(embedding_norms, bins=30, edgecolor='black', alpha=0.7)
    axes[0,0].set_title('Embedding Norm Distribution')
    axes[0,0].set_xlabel('L2 Norm')
    axes[0,0].set_ylabel('Count')
    
    # Embedding component variance
    component_vars = np.var(embeddings, axis=0)
    axes[0,1].plot(component_vars[:100])  # Plot first 100 components
    axes[0,1].set_title('Embedding Component Variance (First 100 dims)')
    axes[0,1].set_xlabel('Component Index')
    axes[0,1].set_ylabel('Variance')
    
    # PCA visualization (2D)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    scatter = axes[1,0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=df['average_rating'] if 'average_rating' in df.columns else None,
                               alpha=0.6, s=10)
    axes[1,0].set_title('PCA Visualization (colored by rating)')
    axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    if 'average_rating' in df.columns:
        plt.colorbar(scatter, ax=axes[1,0], label='Average Rating')
    
    # Explained variance
    pca_full = PCA(n_components=min(50, embeddings.shape[1]), random_state=42)
    pca_full.fit(embeddings)
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    axes[1,1].plot(range(1, len(cumsum_var)+1), cumsum_var)
    axes[1,1].set_title('PCA Explained Variance (First 50 components)')
    axes[1,1].set_xlabel('Number of Components')
    axes[1,1].set_ylabel('Cumulative Explained Variance')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    embedding_path = report_dir / 'embedding_analysis.png'
    plt.savefig(embedding_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'embedding_dimensions': embeddings.shape[1],
        'mean_norm': embedding_norms.mean(),
        'std_norm': embedding_norms.std(),
        'pca_variance_explained_2d': pca.explained_variance_ratio_.sum(),
        'total_variance_50_components': cumsum_var[-1] if len(cumsum_var) > 0 else None
    }

def generate_markdown_report(df, report_dir, rating_stats, price_stats, text_stats, category_stats, embedding_stats):
    """Generate comprehensive markdown report"""
    report_path = report_dir / 'README.md'
    
    with open(report_path, 'w') as f:
        f.write(f"# Dataset Analysis Report: {report_dir.name}\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset Overview
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total Records**: {len(df):,}\n")
        f.write(f"- **Total Columns**: {len(df.columns)}\n")
        f.write(f"- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n\n")
        
        # Column Information
        f.write("### Column Information\n\n")
        f.write("| Column | Type | Non-Null Count | Unique Values |\n")
        f.write("|--------|------|----------------|---------------|\n")
        for col in df.columns:
            if col == 'embedding':  # Special handling for embedding column
                dtype = 'embedding (array)'
                non_null = df[col].notna().sum()
                f.write(f"| {col} | {dtype} | {non_null:,} | N/A |\n")
            else:
                dtype = str(df[col].dtype)
                non_null = df[col].notna().sum()
                try:
                    unique = df[col].nunique()
                    f.write(f"| {col} | {dtype} | {non_null:,} | {unique:,} |\n")
                except (TypeError, ValueError):
                    f.write(f"| {col} | {dtype} | {non_null:,} | N/A |\n")
        f.write("\n")
        
        # Rating Analysis
        if rating_stats:
            f.write("## Rating Analysis\n\n")
            f.write(f"- **Average Rating**: {rating_stats['mean_rating']:.2f}\n")
            f.write(f"- **Median Rating**: {rating_stats['median_rating']:.2f}\n")
            f.write(f"- **Rating Standard Deviation**: {rating_stats['rating_std']:.2f}\n")
            if rating_stats['total_reviews']:
                f.write(f"- **Total Reviews**: {rating_stats['total_reviews']:,}\n")
            f.write("\n![Rating Analysis](rating_analysis.png)\n\n")
        
        # Price Analysis
        if price_stats:
            f.write("## Price Analysis\n\n")
            f.write(f"- **Average Price**: ${price_stats['mean_price']:.2f}\n")
            f.write(f"- **Median Price**: ${price_stats['median_price']:.2f}\n")
            f.write(f"- **Price Range**: ${price_stats['min_price']:.2f} - ${price_stats['max_price']:.2f}\n")
            f.write(f"- **Items with Price**: {price_stats['priced_items']:,} ({price_stats['price_coverage']:.1f}%)\n")
            f.write("\n![Price Analysis](price_analysis.png)\n\n")
        
        # Text Analysis
        if text_stats:
            f.write("## Text Analysis\n\n")
            if 'avg_title_length' in text_stats:
                f.write(f"- **Average Title Length**: {text_stats['avg_title_length']:.0f} characters\n")
            if 'avg_desc_length' in text_stats:
                f.write(f"- **Average Description Length**: {text_stats['avg_desc_length']:.0f} characters\n")
            if 'features_coverage' in text_stats:
                f.write(f"- **Items with Features**: {text_stats['features_coverage']:.1f}%\n")
            f.write("\n![Text Analysis](text_analysis.png)\n\n")
        
        # Category Analysis
        if category_stats:
            f.write("## Category Analysis\n\n")
            f.write(f"- **Unique Categories**: {category_stats['unique_categories']}\n")
            if category_stats['top_category']:
                f.write(f"- **Top Category**: {category_stats['top_category']} ({category_stats['top_category_count']} items)\n")
            f.write("\n![Category Analysis](category_analysis.png)\n\n")
        
        # Embedding Analysis
        if embedding_stats:
            f.write("## Embedding Analysis\n\n")
            f.write(f"- **Embedding Dimensions**: {embedding_stats['embedding_dimensions']}\n")
            f.write(f"- **Average L2 Norm**: {embedding_stats['mean_norm']:.3f} ± {embedding_stats['std_norm']:.3f}\n")
            f.write(f"- **PCA Variance Explained (2D)**: {embedding_stats['pca_variance_explained_2d']:.1%}\n")
            if embedding_stats['total_variance_50_components']:
                f.write(f"- **PCA Variance Explained (50D)**: {embedding_stats['total_variance_50_components']:.1%}\n")
            f.write("\n![Embedding Analysis](embedding_analysis.png)\n\n")
        
        # Interactive Visualization
        f.write("## Interactive Visualization\n\n")
        f.write("For interactive exploration of embeddings and data relationships, ")
        f.write("the dataset is also available through Apple's Embedding Atlas visualization tool.\n\n")
        f.write("The interactive visualization provides:\n")
        f.write("- 2D/3D embedding projections using UMAP\n")
        f.write("- Interactive filtering and search\n")
        f.write("- Nearest neighbor exploration\n")
        f.write("- Real-time clustering analysis\n\n")
    
    print(f"Report generated: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive dataset visualization and analysis')
    parser.add_argument('parquet_file', help='Path to parquet file to analyze')
    parser.add_argument('--sample', type=int, default=None, help='Sample size (default: use all data)')
    
    args = parser.parse_args()
    
    # Create report directory
    report_dir = create_report_directory(args.parquet_file)
    print(f"Creating report in directory: {report_dir}")
    
    # Load data
    df = load_and_analyze_data(args.parquet_file, args.sample)
    
    # Generate all analyses
    print("Generating rating analysis...")
    rating_stats = create_rating_analysis(df, report_dir)
    
    print("Generating price analysis...")
    price_stats = create_price_analysis(df, report_dir)
    
    print("Generating text analysis...")
    text_stats = create_text_analysis(df, report_dir)
    
    print("Generating category analysis...")
    category_stats = create_category_analysis(df, report_dir)
    
    print("Generating embedding analysis...")
    embedding_stats = create_embedding_analysis(df, report_dir)
    
    # Generate markdown report
    print("Generating markdown report...")
    generate_markdown_report(df, report_dir, rating_stats, price_stats, text_stats, category_stats, embedding_stats)
    
    print(f"\n✅ Analysis complete! Report saved to: {report_dir}/")
    print(f"📊 View the full report: {report_dir}/README.md")

if __name__ == "__main__":
    main()