# Benchmark Dataset Analysis Report - 100K (Fast Analysis)

**Generated:** 2025-08-10 23:46:12

## Dataset Overview

| Metric | Value |
|--------|--------|
| Dataset Size | 100k |
| Total Records | 98,500 |
| Total Columns | 17 |
| File Size | 2265.8 MB |

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
- Books: 39,926 (40.5%)
- Automotive: 19,281 (19.6%)
- Tools & Home Improvement: 14,261 (14.5%)
- All Beauty: 5,109 (5.2%)
- Buy a Kindle: 2,520 (2.6%)

### rating_tier
- excellent: 38,767 (39.4%)
- high: 30,788 (31.3%)
- medium: 15,678 (15.9%)
- low: 13,267 (13.5%)

### review_volume
- few: 48,078 (48.8%)
- moderate: 33,769 (34.3%)
- many: 12,866 (13.1%)
- popular: 3,787 (3.8%)

### source_dataset
- Books: 43,500 (44.2%)
- Automotive: 22,000 (22.3%)
- Tools_and_Home_Improvement: 18,000 (18.3%)
- Electronics: 8,000 (8.1%)
- Beauty_and_Personal_Care: 7,000 (7.1%)

## Numerical Field Analysis

### average_rating
- Non-null values: 98,500
- Missing values: 0 (0.0%)
- Mean: 4.243
- Range: 1.0 - 5.0
- Std Dev: 0.765

### rating_number
- Non-null values: 98,500
- Missing values: 0 (0.0%)
- Mean: 258.685
- Range: 1 - 279862
- Std Dev: 2518.655

### price
- Non-null values: 57,114
- Missing values: 41,386 (42.0%)
- Mean: 46.247
- Range: 0.01 - 6969.95
- Std Dev: 124.894

## Text Field Analysis

### title
- Non-null values: 98,500
- Missing values: 0 (0.0%)
- Average length: 82.2 characters
- Length range: 0 - 1005

### description
- Non-null values: 98,500
- Missing values: 0 (0.0%)
- Average length: 725.5 characters
- Length range: 0 - 1470853

### features
- Non-null values: 98,500
- Missing values: 0 (0.0%)
- Average length: 584.7 characters
- Length range: 0 - 19153

### combined_text
- Non-null values: 98,500
- Missing values: 0 (0.0%)
- Average length: 1394.1 characters
- Length range: 0 - 1472774


## Summary

This fast analysis of the 100k dataset confirms:
- Consistent schema with smaller dataset samples  
- Similar distribution patterns in categorical fields
- Maintained data quality metrics across scale
- Memory-efficient analysis completed using DuckDB

*Note: This is a statistical validation report. For comprehensive visualizations, see the 10k dataset report.*
