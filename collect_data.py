from src.api.fetch_trending import get_trending_videos
from src.parsing.parse_videos import parse_videos

videos = get_trending_videos(region_code="IN", max_total=200)
df = parse_videos(videos)

df.to_csv("data/raw/youtube_api_trending_raw.csv", index=False)
print(f"Saved {len(df)} videos to data/raw/youtube_api_trending_raw.csv")