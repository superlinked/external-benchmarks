# Benchmark Dataset Analysis Report - 1M (Fast Analysis)

**Generated:** 2025-08-10 23:46:13

## Dataset Overview

| Metric | Value |
|--------|--------|
| Dataset Size | 1M |
| Total Records | 1,044,500 |
| Total Columns | 17 |
| File Size | 23365.2 MB |

## Schema Information

- **parent_asin** (`VARCHAR`)
- **title** (`VARCHAR`)
- **description** (`VARCHAR`)
- **features** (`VARCHAR`)
- **combined_text** (`VARCHAR`)
- **average_rating** (`DOUBLE`)
- **rating_number** (`BIGINT`)
- **price** (`DOUBLE`)
- **main_category** (`VARCHAR`)
- **categories** (`VARCHAR`)
- **store** (`VARCHAR`)
- **details** (`VARCHAR`)
- **source_dataset** (`VARCHAR`)
- **has_price** (`BOOLEAN`)
- **rating_tier** (`VARCHAR`)
- **review_volume** (`VARCHAR`)
- **embedding** (`DOUBLE[]`)

## Categorical Field Analysis

### main_category
- Books: 423,884 (40.6%)
- Automotive: 164,275 (15.7%)
- Tools & Home Improvement: 112,316 (10.8%)
- All Beauty: 68,136 (6.5%)
- Computers: 43,777 (4.2%)

### rating_tier
- excellent: 400,592 (38.4%)
- high: 329,076 (31.5%)
- medium: 169,513 (16.2%)
- low: 145,319 (13.9%)

### review_volume
- few: 498,637 (47.7%)
- moderate: 364,187 (34.9%)
- many: 140,955 (13.5%)
- popular: 40,721 (3.9%)

### source_dataset
- Books: 462,000 (44.2%)
- Automotive: 187,500 (17.9%)
- Electronics: 164,000 (15.7%)
- Tools_and_Home_Improvement: 138,000 (13.2%)
- Beauty_and_Personal_Care: 93,000 (8.9%)

## Numerical Field Analysis

### average_rating
- Non-null values: 1,044,500
- Missing values: 0 (0.0%)
- Mean: 4.230
- Range: 1.0 - 5.0
- Std Dev: 0.763

### rating_number
- Non-null values: 1,044,500
- Missing values: 0 (0.0%)
- Mean: 268.979
- Range: 1 - 609879
- Std Dev: 2674.573

### price
- Non-null values: 591,320
- Missing values: 453,180 (43.4%)
- Mean: 45.672
- Range: 0.01 - 75108.94
- Std Dev: 189.380

## Text Field Analysis

### title
- Non-null values: 1,044,500
- Missing values: 0 (0.0%)
- Average length: 83.9 characters
- Length range: 0 - 1424

### description
- Non-null values: 1,044,500
- Missing values: 0 (0.0%)
- Average length: 734.6 characters
- Length range: 0 - 449865

### features
- Non-null values: 1,044,500
- Missing values: 0 (0.0%)
- Average length: 589.9 characters
- Length range: 0 - 41354

### combined_text
- Non-null values: 1,044,500
- Missing values: 0 (0.0%)
- Average length: 1410.1 characters
- Length range: 0 - 451150


## Summary

This fast analysis of the 1M dataset confirms:
- Consistent schema with smaller dataset samples  
- Similar distribution patterns in categorical fields
- Maintained data quality metrics across scale
- Memory-efficient analysis completed using DuckDB

*Note: This is a statistical validation report. For comprehensive visualizations, see the 10k dataset report.*
