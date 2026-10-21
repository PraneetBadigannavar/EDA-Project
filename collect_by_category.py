from src.api.fetch_by_category import get_videos_by_category
from src.api.category_map import get_category_map
from src.parsing.parse_videos import parse_videos
import pandas as pd
import time

categories = get_category_map(region_code="IN")

all_dfs = []
for cat_id, cat_name in categories.items():
    print(f"Fetching category: {cat_name} ({cat_id})")
    videos = get_videos_by_category(cat_id, region_code="IN", max_total=200)
    if videos:
        df = parse_videos(videos)
        df["category_name"] = cat_name
        all_dfs.append(df)
    time.sleep(1)  # polite pause between calls

combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df = combined_df.drop_duplicates(subset="video_id")

combined_df.to_csv("data/raw/youtube_api_by_category.csv", index=False)
print(f"Saved {len(combined_df)} unique videos across {len(all_dfs)} categories")