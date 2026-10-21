from src.api.youtube_client import youtube

response = youtube.videos().list(
    part="snippet,statistics",
    chart="mostPopular",
    regionCode="IN",
    maxResults=5
).execute()

for item in response["items"]:
    print(item["snippet"]["title"])