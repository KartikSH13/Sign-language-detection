import cv2
import json
import numpy as np # linear algebra
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pafy
import requests
from sklearn.model_selection import train_test_split
from distutils.dir_util import copy_tree
from pytube import YouTube
main_path = "videos/"
batch_path="batch_data/"

wlasl_df = pd.read_json(batch_path+"batch_1.json")

def download_swf(video_url,video_id,save_path):
    return True

def download_yt(video_url, video_id, save_path):
    try:
        video = pafy.new(video_url, ydl_opts={'verbose': True}) 
        best_video_stream = video.getbest(preftype='mp4')
        video_file_path = os.path.join(save_path, f"{video_id}.mp4")
        best_video_stream.download(filepath=video_file_path)
        print("YouTube video downloaded and saved at:", video_file_path)
        return True
    except Exception as e:
        print("Error:", e, video_url)
        return False

def download_mp4(video_url,video_id,save_path):
    response = requests.get(video_url, stream=True)
    if response.status_code == 200:
        video_path = os.path.join(save_path, f'{video_id}.mp4')
        with open(video_path, 'wb') as video_file:
            for chunk in response.iter_content(chunk_size=8192):
                video_file.write(chunk)
        print(f'Video {video_id} downloaded and saved at: {video_path}')
        return True
    else:
        print(f'Failed to download mp4 video {video_id}')
        return False

def download_video(video_url, video_id, save_path):
    if video_url.endswith('.mp4'):
        return download_mp4(video_url,video_id,save_path)
    elif video_url.startswith('https://www.youtube.com'):
        return download_yt(video_url,video_id,save_path)
    elif video_url.endswith('.swf'):
        return download_swf(video_url,video_id,save_path)
    else :
        print("wrong format",video_url)
        return False     
    
def check(video_data):
    for row in video_data:
        video_path=f'{main_path}{row["video_id"]}.mp4'
        if os.path.exists(video_path):
            return
        else:
            download_success = download_video(row['url'], row['video_id'], main_path)
            if download_success:
                pass
            else:
                # Handle the case when the download fails
                print(f'Video {row["video_id"]} not found or download failed!')
     
wlasl_df['instances'].apply(check)