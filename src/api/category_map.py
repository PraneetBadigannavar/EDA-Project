from src.api.youtube_client import youtube

def get_category_map(region_code="IN"):
    request = youtube.videoCategories().list(
        part="snippet", regionCode=region_code
    )
    response = request.execute()
    return {item["id"]: item["snippet"]["title"] for item in response["items"]}