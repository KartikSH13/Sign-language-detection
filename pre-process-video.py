# pip install nqdm

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os
import cv2
import shutil
from sklearn.model_selection import train_test_split
# from nqdm import nqdm
# from IPython.display import FileLink
from distutils.dir_util import copy_tree

main_path = "videos/"

output_folder = 'split-data-frames/'
wlasl_df = pd.read_json("WLASL_v0.3.json")

print(len(wlasl_df))

# wlasl_df.head(5)

def get_videos_ids(json_list):
    """
    check if the video id is available in the dataset
    and return the viedos ids of the current instance

    Args:
        json_list: Instance of video metadata.

    Returns:
        List of video ids.
    """
    video_ids = []
    for ins in json_list:
        video_id = ins['video_id']
        if os.path.exists(f'{main_path}{video_id}.mp4'):
            video_ids.append(video_id)
    return video_ids

def get_json_features(json_list):
    """
    function to check if the video id is available in the dataset
    and return the viedos ids and url or any other featrue of the current instance

    input: instance json list
    output: list of videos_ids

    """
    videos_ids = []
    videos_urls = []
    videos_bbox = []
    videos_fps = []
    videos_frame_end = []
    videos_frame_start = []
    videos_signer_id = []
    videos_source = []
    videos_split = []
    videos_variation_id = []
    for ins in json_list:
        video_id = ins['video_id']
        video_url = ins['url']
        video_bbox = ins['bbox']
        video_fps = ins['fps']
        video_frame_end = ins['frame_end']
        video_frame_start = ins['frame_start']
        video_signer_id = ins['signer_id']
        video_source = ins['source']
        video_split = ins['split']
        video_variation_id = ins['variation_id']
        if os.path.exists(f'{main_path}{video_id}.mp4'):
            videos_ids.append(video_id)
            videos_urls.append(video_url)
            videos_bbox.append(video_bbox)
            videos_fps.append(video_fps)
            videos_frame_end.append(video_frame_end)
            videos_frame_start.append(video_frame_start)
            videos_signer_id.append(video_signer_id)
            videos_source.append(video_source)
            videos_split.append(video_split)
            videos_variation_id.append(video_variation_id)
    return videos_ids, videos_urls, videos_bbox, videos_fps, videos_frame_end, videos_frame_start, videos_signer_id, videos_source, videos_split,videos_variation_id

with open('WLASL_v0.3.json', 'r') as data_file:
    json_data = data_file.read()

instance_json = json.loads(json_data)
wlasl_df["video_ids"] = wlasl_df["instances"].apply(get_videos_ids)
# print(wlasl_df.head(5))

features_df = pd.DataFrame(columns=['gloss', 'video_id', 'urls', 'bbox', 'fps', 'frame_end', 'frame_start','signer_id', 'source', 'split', 'variation_id'])
for row in wlasl_df.iterrows():
    ids, urls, bbox, fps, frame_end, frame_start,signer_id, source, split, variation_id = get_json_features(row[1][1])
    word = [row[1][0]] * len(ids)
    df = pd.DataFrame(list(zip(word, ids, urls, bbox, fps, frame_end, frame_start, signer_id, source, split, variation_id)), columns=features_df.columns)
    # features_df = features_df.append(df)
    features_df = pd.concat([features_df, df])

print("features")
# print(features_df.head(10))

print(len(features_df))
print(len(features_df.gloss.unique()))
train_mask = features_df['split'] == 'train'
val_mask = features_df['split'] == 'val'
test_mask = features_df['split'] == 'test'

train_pos = np.flatnonzero(train_mask)
val_pos =  np.flatnonzero(val_mask)
test_pos = np.flatnonzero(test_mask)

train = features_df.iloc[train_pos]

val = features_df.iloc[val_pos]

test = features_df.iloc[test_pos]

x_train = train.loc[:, train.columns != 'gloss']
y_train = train['gloss']

x_val = val.loc[:, val.columns != 'gloss']
y_val = val['gloss']

x_test = test.loc[:, test.columns != 'gloss']
y_test = test['gloss']

print(len(x_train),len(x_test), len(x_val))

# print(x_train)

# shutil.rmtree("My Drive/kaggle/working/data")

def video_to_frames(from_directory, to_directory):
    
    # Create the output directory if it doesn't exist
    os.makedirs(to_directory, exist_ok=True)

    # Check if the from_directory is a video file
    if not from_directory.endswith(".mp4"):
        print("Error: The from_directory should point directly to a video file.")
        return

    # Open the video file
    cap = cv2.VideoCapture(from_directory)

    # Create a subdirectory within the output directory for this video
    video_name = os.path.splitext(os.path.basename(from_directory))[0]
    output_video_directory = os.path.join(to_directory, video_name)
    os.makedirs(output_video_directory, exist_ok=True)

    frame_count = 0

    while True:
        ret, frame = cap.read()  # Read a frame
        if not ret:
            break
        # Resize the frame to the target size (e.g., 224x224 pixels)
        frame = cv2.resize(frame, (224, 224))
            
        # Normalize pixel values to the range [0, 1]
        # frame = frame / 255.0
        
        # Save the frame as an image
        frame_filename = f'frame_{video_name}_{frame_count}.png'
        frame_path = os.path.join(output_video_directory, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()
    # print(f"{frame_count} Frames from {from_directory} saved to {output_video_directory}.")


def generateDatasplitFolder(series, folderName):
    video_count=0
    new_path = output_folder+str(folderName)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        for val in series.video_id:
            # print(str(val))
            from_directory = main_path+str(val)+'.mp4'
            to_directory = new_path
            video_to_frames(from_directory, to_directory)
            video_count+=1
            print(f"{video_count} video saved to {str(folderName)} ")

# generateDatasplitFolder(x_test, 'test')

# generateDatasplitFolder(x_val, 'val')
 
# generateDatasplitFolder(x_train, 'train')