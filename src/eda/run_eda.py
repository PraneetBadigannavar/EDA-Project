import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

def run_eda(input_path='data/processed/youtube_features.csv', output_dir='results/eda'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    
    summary_findings = {}
    
    # -------------------------------------------------------------
    # 1. View Distribution Analysis
    # -------------------------------------------------------------
    print("Generating Analysis 1: View Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df['views'], bins=50, kde=True, ax=axes[0], color='#2b5c8f')
    axes[0].set_title('Raw View Count Distribution (Skewe`d)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Views')
    axes[0].set_ylabel('Frequency')
    
    sns.histplot(df['log_views'], bins=50, kde=True, ax=axes[1], color='#e056fd')
    axes[1].set_title('Log-Transformed View Count Distribution', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Log(Views + 1)')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_view_distribution.png'), dpi=300)
    plt.close()
    
    view_median = float(df['views'].median())
    view_mean = float(df['views'].mean())
    summary_findings['analysis_1_view_distribution'] = {
        'median_views': view_median,
        'mean_views': view_mean,
        'conclusion': 'View count exhibits extreme positive skewness; most videos are concentrated in lower view ranges (< 1M views) with a small percentage of viral outliers.'
    }
    
    # -------------------------------------------------------------
    # 2. Category Distribution Analysis
    # -------------------------------------------------------------
    print("Generating Analysis 2: Category Distribution...")
    cat_counts = df['category_name'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=cat_counts.values, y=cat_counts.index, hue=cat_counts.index, palette='viridis', legend=False)
    plt.title('Top 10 Dominant YouTube Video Categories', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Videos')
    plt.ylabel('Category')
    
    for i, v in enumerate(cat_counts.values):
        ax.text(v + 10, i, f"{v:,}", va='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_category_distribution.png'), dpi=300)
    plt.close()
    
    summary_findings['analysis_2_category_distribution'] = {
        'top_category': cat_counts.index[0],
        'top_category_count': int(cat_counts.values[0]),
        'conclusion': f"The '{cat_counts.index[0]}' category heavily dominates YouTube content, followed by '{cat_counts.index[1]}' and '{cat_counts.index[2]}'."
    }
    
    # -------------------------------------------------------------
    # 3. Engagement by Category
    # -------------------------------------------------------------
    print("Generating Analysis 3: Engagement by Category...")
    cat_eng = df.groupby('category_name')['engagement_score'].mean().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=cat_eng.values, y=cat_eng.index, hue=cat_eng.index, palette='magma', legend=False)
    plt.title('Top 10 Categories by Average Engagement Score (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Average Engagement Score (%)')
    plt.ylabel('Category')
    
    for i, v in enumerate(cat_eng.values):
        ax.text(v + 0.1, i, f"{v:.2f}%", va='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_engagement_by_category.png'), dpi=300)
    plt.close()
    
    summary_findings['analysis_3_engagement_by_category'] = {
        'highest_engagement_category': cat_eng.index[0],
        'highest_engagement_score': float(cat_eng.values[0]),
        'conclusion': f"'{cat_eng.index[0]}' yields the highest average user interaction rate relative to total views."
    }
    
    # -------------------------------------------------------------
    # 4. Upload Day Analysis
    # -------------------------------------------------------------
    print("Generating Analysis 4: Upload Day Analysis...")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_df = df.groupby('upload_day').agg(
        video_count=('video_id', 'count'),
        avg_engagement=('engagement_score', 'mean'),
        avg_views=('views', 'mean')
    ).reindex(day_order)
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    color1 = '#3498db'
    ax1.set_xlabel('Day of the Week', fontweight='bold')
    ax1.set_ylabel('Video Count', color=color1, fontweight='bold')
    bars = ax1.bar(day_df.index, day_df['video_count'], color=color1, alpha=0.7, width=0.4, label='Video Count')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    ax2.set_ylabel('Avg Engagement Score (%)', color=color2, fontweight='bold')
    line = ax2.plot(day_df.index, day_df['avg_engagement'], color=color2, marker='o', linewidth=3, label='Avg Engagement %')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Video Upload Frequency & Average Engagement by Day of Week', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_upload_day_analysis.png'), dpi=300)
    plt.close()
    
    summary_findings['analysis_4_upload_day'] = {
        'peak_upload_day': str(day_df['video_count'].idxmax()),
        'best_engagement_day': str(day_df['avg_engagement'].idxmax()),
        'conclusion': f"Peak upload frequency occurs on {day_df['video_count'].idxmax()}, while maximum audience engagement rate is achieved on {day_df['avg_engagement'].idxmax()}."
    }
    
    # -------------------------------------------------------------
    # 5. Upload Hour Analysis
    # -------------------------------------------------------------
    print("Generating Analysis 5: Upload Hour Analysis...")
    hour_df = df.groupby('upload_hour').agg(
        video_count=('video_id', 'count'),
        avg_views=('views', 'mean'),
        avg_engagement=('engagement_score', 'mean')
    )
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    ax1.plot(hour_df.index, hour_df['video_count'], color='#2ecc71', marker='s', linewidth=2.5, label='Upload Count')
    ax1.set_xlabel('Upload Hour (UTC)', fontweight='bold')
    ax1.set_ylabel('Number of Videos Uploaded', color='#2ecc71', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2ecc71')
    ax1.set_xticks(range(0, 24))
    
    ax2 = ax1.twinx()
    ax2.plot(hour_df.index, hour_df['avg_views'], color='#9b59b6', marker='^', linewidth=2.5, linestyle='--', label='Avg Views')
    ax2.set_ylabel('Average Views', color='#9b59b6', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#9b59b6')
    
    plt.title('Upload Hour vs. Upload Volume and Average Views', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_upload_hour_analysis.png'), dpi=300)
    plt.close()
    
    summary_findings['analysis_5_upload_hour'] = {
        'peak_upload_hour': int(hour_df['video_count'].idxmax()),
        'highest_views_hour': int(hour_df['avg_views'].idxmax()),
        'conclusion': f"Upload volume peaks around {hour_df['video_count'].idxmax()}:00 UTC, whereas videos uploaded around {hour_df['avg_views'].idxmax()}:00 UTC generate higher average view counts."
    }
    
    # -------------------------------------------------------------
    # 6. Correlation Analysis
    # -------------------------------------------------------------
    print("Generating Analysis 6: Correlation Heatmap...")
    corr_cols = ['views', 'likes', 'comment_count', 'engagement_score', 'title_length', 'title_word_count', 'video_age_days', 'like_ratio']
    corr_matrix = df[corr_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='Blues', linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_correlation_heatmap.png'), dpi=300)
    plt.close()
    
    summary_findings['analysis_6_correlation'] = {
        'views_likes_corr': float(corr_matrix.loc['views', 'likes']),
        'views_comments_corr': float(corr_matrix.loc['views', 'comment_count']),
        'title_len_views_corr': float(corr_matrix.loc['title_length', 'views']),
        'conclusion': f"Strong linear correlation between views and likes ({corr_matrix.loc['views', 'likes']:.2f}) and comments ({corr_matrix.loc['views', 'comment_count']:.2f}). Title length shows near-zero correlation with views."
    }
    
    # -------------------------------------------------------------
    # 7. Outlier & Viral Video Detection
    # -------------------------------------------------------------
    print("Generating Analysis 7: Outlier Detection...")
    p99_views = df['views'].quantile(0.99)
    df['is_viral'] = df['views'] > p99_views
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=df,
        x='views',
        y='engagement_score',
        hue='is_viral',
        palette={False: '#34495e', True: '#e74c3c'},
        alpha=0.6,
        s=40
    )
    plt.xscale('log')
    plt.axvline(x=p99_views, color='red', linestyle='--', label=f'99th Percentile ({int(p99_views):,} views)')
    plt.title('Viral Outlier Detection (Views vs. Engagement Score)', fontsize=14, fontweight='bold')
    plt.xlabel('Views (Log Scale)')
    plt.ylabel('Engagement Score (%)')
    plt.legend(title='Is Viral Outlier (Top 1%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_outlier_detection.png'), dpi=300)
    plt.close()
    
    summary_findings['analysis_7_outliers'] = {
        'p99_views_threshold': float(p99_views),
        'viral_video_count': int(df['is_viral'].sum()),
        'conclusion': f"The top 1% viral videos threshold is {int(p99_views):,} views. Highly viral videos often maintain high engagement, though extreme outlier view counts can exhibit lower percentage engagement due to massive passive reach."
    }
    
    # Save findings JSON
    with open(os.path.join(output_dir, 'eda_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_findings, f, indent=4)
        
    print(f"EDA successfully completed! All 7 plots and eda_summary.json saved in {output_dir}.")
    return summary_findings

if __name__ == '__main__':
    run_eda()
