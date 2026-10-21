import os
import json
import pandas as pd
import numpy as np

# Standard YouTube Category Mapping for IDs
CATEGORY_ID_MAP = {
    1: 'Film & Animation',
    2: 'Autos & Vehicles',
    10: 'Music',
    15: 'Pets & Animals',
    17: 'Sports',
    18: 'Shorts',
    19: 'Travel & Events',
    20: 'Gaming',
    21: 'Videoblogging',
    22: 'People & Blogs',
    23: 'Comedy',
    24: 'Entertainment',
    25: 'News & Politics',
    26: 'Howto & Style',
    27: 'Education',
    28: 'Science & Technology',
    29: 'Nonprofits & Activism',
    30: 'Movies',
    31: 'Anime/Animation',
    32: 'Action/Adventure',
    33: 'Classics',
    34: 'Comedy',
    35: 'Documentary',
    36: 'Drama',
    37: 'Family',
    38: 'Foreign',
    39: 'Horror',
    40: 'Sci-Fi/Fantasy',
    41: 'Thriller',
    42: 'Short Shorts',
    43: 'Shows',
    44: 'Trailers'
}

def load_raw_data():
    raw_dir = os.path.join('data', 'raw')
    
    kaggle_path = os.path.join(raw_dir, 'INvideos_cleaned.csv')
    api_cat_path = os.path.join(raw_dir, 'youtube_api_by_category.csv')
    api_trend_path = os.path.join(raw_dir, 'youtube_api_trending_raw.csv')
    
    df_kaggle = pd.read_csv(kaggle_path, encoding='utf-8', encoding_errors='replace')
    df_api_cat = pd.read_csv(api_cat_path, encoding='utf-8', encoding_errors='replace')
    df_api_trend = pd.read_csv(api_trend_path, encoding='utf-8', encoding_errors='replace')
    
    return df_kaggle, df_api_cat, df_api_trend

def perform_quality_assessment(df_kaggle, df_api_cat, df_api_trend):
    report = {
        'kaggle_dataset': {
            'rows': int(len(df_kaggle)),
            'columns': int(len(df_kaggle.columns)),
            'missing_values': df_kaggle.isnull().sum().to_dict(),
            'duplicate_video_ids': int(df_kaggle['video_id'].duplicated().sum()),
            'duplicate_records': int(df_kaggle.duplicated().sum())
        },
        'api_category_dataset': {
            'rows': int(len(df_api_cat)),
            'columns': int(len(df_api_cat.columns)),
            'missing_values': df_api_cat.isnull().sum().to_dict(),
            'duplicate_video_ids': int(df_api_cat['video_id'].duplicated().sum()),
            'duplicate_records': int(df_api_cat.duplicated().sum())
        },
        'api_trending_dataset': {
            'rows': int(len(df_api_trend)),
            'columns': int(len(df_api_trend.columns)),
            'missing_values': df_api_trend.isnull().sum().to_dict(),
            'duplicate_video_ids': int(df_api_trend['video_id'].duplicated().sum()),
            'duplicate_records': int(df_api_trend.duplicated().sum())
        }
    }
    
    os.makedirs('reports', exist_ok=True)
    with open(os.path.join('reports', 'data_quality_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    print("Data Quality Assessment Report saved to reports/data_quality_report.json")
    return report

def clean_kaggle_data(df):
    df_clean = df.copy()
    
    # Fill null descriptions / titles
    df_clean['title'] = df_clean['title'].fillna('').astype(str)
    df_clean['description'] = df_clean['description'].fillna('').astype(str)
    df_clean['channel_title'] = df_clean['channel_title'].fillna('Unknown Channel').astype(str)
    
    # Convert numerical columns
    for col in ['views', 'likes', 'comment_count']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
        
    # Convert category_id
    df_clean['category_id'] = pd.to_numeric(df_clean['category_id'], errors='coerce').fillna(0).astype(int)
    
    # Map category names if missing
    if 'category_name' not in df_clean.columns or df_clean['category_name'].isnull().all():
        df_clean['category_name'] = df_clean['category_id'].map(CATEGORY_ID_MAP).fillna('Other')
    else:
        df_clean['category_name'] = df_clean['category_name'].fillna(df_clean['category_id'].map(CATEGORY_ID_MAP)).fillna('Other')
        
    # Parse published dates
    df_clean['published_at'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    
    # Select standardized columns
    std_cols = ['video_id', 'title', 'description', 'channel_title', 'category_id', 'category_name', 'published_at', 'views', 'likes', 'comment_count']
    df_clean = df_clean[std_cols].copy()
    df_clean['source'] = 'Kaggle'
    
    # Sort by views descending before dropping duplicates so we keep peak metrics
    df_clean = df_clean.sort_values('views', ascending=False).drop_duplicates(subset=['video_id'], keep='first')
    
    return df_clean

def clean_api_data(df_cat, df_trend):
    df_combined = pd.concat([df_cat, df_trend], ignore_index=True)
    df_clean = df_combined.copy()
    
    # Rename columns to standard names
    rename_dict = {
        'channelTitle': 'channel_title',
        'categoryId': 'category_id',
        'publishedAt': 'published_at',
        'viewCount': 'views',
        'likeCount': 'likes',
        'commentCount': 'comment_count'
    }
    df_clean = df_clean.rename(columns=rename_dict)
    
    # Fill missing fields
    df_clean['title'] = df_clean['title'].fillna('').astype(str)
    df_clean['description'] = df_clean['description'].fillna('').astype(str)
    df_clean['channel_title'] = df_clean['channel_title'].fillna('Unknown Channel').astype(str)
    
    # Numerical conversions
    for col in ['views', 'likes', 'comment_count']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
        
    df_clean['category_id'] = pd.to_numeric(df_clean['category_id'], errors='coerce').fillna(0).astype(int)
    
    if 'category_name' not in df_clean.columns:
        df_clean['category_name'] = df_clean['category_id'].map(CATEGORY_ID_MAP).fillna('Other')
    else:
        df_clean['category_name'] = df_clean['category_name'].fillna(df_clean['category_id'].map(CATEGORY_ID_MAP)).fillna('Other')
        
    df_clean['published_at'] = pd.to_datetime(df_clean['published_at'], errors='coerce')
    
    std_cols = ['video_id', 'title', 'description', 'channel_title', 'category_id', 'category_name', 'published_at', 'views', 'likes', 'comment_count']
    df_clean = df_clean[std_cols].copy()
    df_clean['source'] = 'YouTube API'
    
    # Sort by views descending and drop duplicates
    df_clean = df_clean.sort_values('views', ascending=False).drop_duplicates(subset=['video_id'], keep='first')
    
    return df_clean

def run_preprocessing_pipeline():
    print("--- Phase 1 & 2: Loading & Data Quality Assessment ---")
    df_kaggle, df_api_cat, df_api_trend = load_raw_data()
    report = perform_quality_assessment(df_kaggle, df_api_cat, df_api_trend)
    
    print("\n--- Phase 3: Data Cleaning ---")
    kaggle_clean = clean_kaggle_data(df_kaggle)
    api_clean = clean_api_data(df_api_cat, df_api_trend)
    
    os.makedirs('data/cleaned', exist_ok=True)
    kaggle_clean.to_csv('data/cleaned/kaggle_cleaned.csv', index=False, encoding='utf-8')
    api_clean.to_csv('data/cleaned/api_cleaned.csv', index=False, encoding='utf-8')
    
    print(f"Kaggle cleaned saved: {len(kaggle_clean)} unique records.")
    print(f"API cleaned saved: {len(api_clean)} unique records.")
    
    print("\n--- Phase 4: Dataset Consolidation ---")
    master_df = pd.concat([kaggle_clean, api_clean], ignore_index=True)
    # Deduplicate video_ids across sources (preferring peak metrics)
    master_df = master_df.sort_values('views', ascending=False).drop_duplicates(subset=['video_id'], keep='first')
    master_df = master_df.sort_values('published_at', ascending=False).reset_index(drop=True)
    
    master_df.to_csv('data/cleaned/youtube_master.csv', index=False, encoding='utf-8')
    print(f"Consolidated master dataset saved to data/cleaned/youtube_master.csv with {len(master_df)} unique records.")
    
    return master_df

if __name__ == '__main__':
    run_preprocessing_pipeline()
