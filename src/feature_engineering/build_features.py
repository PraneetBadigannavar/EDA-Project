import os
import pandas as pd
import numpy as np

def build_features(input_path='data/cleaned/youtube_master.csv', output_path='data/processed/youtube_features.csv'):
    print(f"Loading cleaned dataset from {input_path}...")
    df = pd.read_csv(input_path, encoding='utf-8')
    
    # Ensure published_at is datetime
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    
    # Drop rows where published_at could not be parsed
    df = df.dropna(subset=['published_at']).copy()
    
    print("Generating Content Features...")
    df['title_length'] = df['title'].fillna('').astype(str).str.len()
    df['title_word_count'] = df['title'].fillna('').astype(str).str.split().str.len()
    df['description_length'] = df['description'].fillna('').astype(str).str.len()
    
    print("Generating Time Features...")
    df['upload_hour'] = df['published_at'].dt.hour
    df['upload_day'] = df['published_at'].dt.day_name()
    df['upload_day_num'] = df['published_at'].dt.dayofweek
    df['upload_month'] = df['published_at'].dt.month
    df['upload_year'] = df['published_at'].dt.year
    df['is_weekend'] = df['upload_day_num'].apply(lambda x: 1 if x in [5, 6] else 0)
    
    print("Generating Video Age Features...")
    # Reference date is current max published date + 1 day
    max_date = df['published_at'].max()
    df['video_age_days'] = (max_date - df['published_at']).dt.total_seconds() / (24 * 3600)
    df['video_age_days'] = df['video_age_days'].clip(lower=0.1) # avoid 0 age
    
    print("Generating Engagement Features & Scores...")
    # Add epsilon to views to avoid division by zero
    views = df['views'].clip(lower=1)
    df['like_ratio'] = df['likes'] / views
    df['comment_ratio'] = df['comment_count'] / views
    
    # Engagement score: (likes + comments) / views * 100
    df['engagement_score'] = ((df['likes'] + df['comment_count']) / views) * 100
    
    # Log-transformed metrics for skewness handling in ML/EDA
    df['log_views'] = np.log1p(df['views'])
    df['log_likes'] = np.log1p(df['likes'])
    df['log_comment_count'] = np.log1p(df['comment_count'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Feature engineering completed! Saved {len(df)} records with {len(df.columns)} columns to {output_path}.")
    return df

if __name__ == '__main__':
    build_features()
