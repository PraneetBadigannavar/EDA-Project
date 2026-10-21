import pandas as pd

def parse_videos(video_items):
    rows = []
    for v in video_items:
        snippet = v.get("snippet", {})
        stats = v.get("statistics", {})
        content = v.get("contentDetails", {})

        rows.append({
            "video_id": v.get("id"),
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "channelTitle": snippet.get("channelTitle"),
            "channelId": snippet.get("channelId"),
            "categoryId": snippet.get("categoryId"),
            "publishedAt": snippet.get("publishedAt"),
            "tags": snippet.get("tags", []),
            "viewCount": stats.get("viewCount"),
            "likeCount": stats.get("likeCount"),
            "commentCount": stats.get("commentCount"),
            "duration": content.get("duration"),
            "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url")
        })
    return pd.DataFrame(rows)