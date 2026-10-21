from src.api.youtube_client import youtube

def get_trending_videos(region_code="IN", max_total=200):
    all_videos = []
    next_page_token = None

    while len(all_videos) < max_total:
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            chart="mostPopular",
            regionCode=region_code,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()
        all_videos.extend(response["items"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return all_videos