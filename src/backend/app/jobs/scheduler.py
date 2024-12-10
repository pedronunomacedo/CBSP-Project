import requests
import hashlib
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
import json

# Database setup
engine = create_engine('sqlite:///songs.db')
metadata = MetaData()
songs = Table('songs', metadata,
              Column('id', Integer, primary_key=True),
              Column('title', String),
              Column('hash', String))
metadata.create_all(engine)

def get_spotify_token():
    client_id = 'fbd75e1af9034bc6b5987d3d12927e98'
    client_secret = '6ecbd072fc0e4be39d6e620ba57360e8'
    auth_url = 'https://accounts.spotify.com/api/token'

    # Post client credentials to get access token
    auth_response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    })

    auth_response_data = auth_response.json()
    access_token = auth_response_data['access_token']

    return access_token

def fetch_songs():
    access_token = get_spotify_token()
    playlist_id = "37i9dQZEVXbMDoHDwVN2tF" # Global top 50 (updated daily)

    # Use access token to access Spotify API
    headers = {
        'Authorization': f'Bearer {access_token}',
    }

    base_url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    response = requests.get(base_url, headers=headers)

    song_data = response.json()  # Adjust based on the actual JSON response structure
    print(json.dumps(song_data, indent=4))
    song_data = song_data['items']

    top_songs = []

    for song in song_data:
        # Process each song to extract details and store in database
        top_songs = {
            "name": song['track']['name'],
            "artists": [artist['name'] for artist in song['track']['artists']]
        }
        print(song['track']['name'])

    print("Fetching and processing songs...")

# Set up the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(fetch_songs, 'cron', minute='*')  # Run daily at midnight
scheduler.start()