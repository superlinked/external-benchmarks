# Benchmark Dataset Analysis Report - 10M (Fast Analysis)

**Generated:** 2025-08-10 23:46:26

## Dataset Overview

| Metric | Value |
|--------|--------|
| Dataset Size | 10M |
| Total Records | 10,564,046 |
| Total Columns | 17 |
| File Size | 242668.9 MB |

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
- Books: 4,080,671 (38.6%)
- Automotive: 1,757,073 (16.6%)
- Tools & Home Improvement: 1,194,237 (11.3%)
- All Beauty: 742,225 (7.0%)
- Computers: 422,915 (4.0%)

### rating_tier
- excellent: 4,022,421 (38.1%)
- high: 3,325,506 (31.5%)
- medium: 1,730,474 (16.4%)
- low: 1,485,645 (14.1%)

### review_volume
- few: 5,041,804 (47.7%)
- moderate: 3,688,431 (34.9%)
- many: 1,425,963 (13.5%)
- popular: 407,848 (3.9%)

### source_dataset
- Books: 4,448,181 (42.1%)
- Automotive: 2,003,129 (19.0%)
- Electronics: 1,610,012 (15.2%)
- Tools_and_Home_Improvement: 1,473,810 (13.9%)
- Beauty_and_Personal_Care: 1,028,914 (9.7%)

## Numerical Field Analysis

### average_rating
- Non-null values: 10,564,046
- Missing values: 0 (0.0%)
- Mean: 4.225
- Range: 1.0 - 5.0
- Std Dev: 0.766

### rating_number
- Non-null values: 10,564,046
- Missing values: 0 (0.0%)
- Mean: 263.813
- Range: 1 - 1034896
- Std Dev: 2650.980

### price
- Non-null values: 5,951,325
- Missing values: 4,612,721 (43.7%)
- Mean: 46.509
- Range: 0.01 - 123567.97
- Std Dev: 178.496

## Text Field Analysis

### title
- Non-null values: 10,564,046
- Missing values: 0 (0.0%)
- Average length: 85.0 characters
- Length range: 0 - 2000

### description
- Non-null values: 10,564,046
- Missing values: 0 (0.0%)
- Average length: 723.2 characters
- Length range: 0 - 1470853

### features
- Non-null values: 10,564,046
- Missing values: 0 (0.0%)
- Average length: 583.5 characters
- Length range: 0 - 190557

### combined_text
- Non-null values: 10,564,046
- Missing values: 0 (0.0%)
- Average length: 1393.4 characters
- Length range: 0 - 1472774


## Summary

This fast analysis of the 10M dataset confirms:
- Consistent schema with smaller dataset samples  
- Similar distribution patterns in categorical fields
- Maintained data quality metrics across scale
- Memory-efficient analysis completed using DuckDB

*Note: This is a statistical validation report. For comprehensive visualizations, see the 10k dataset report.*
