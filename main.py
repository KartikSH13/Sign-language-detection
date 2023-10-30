import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os
import cv2
import shutil
from sklearn.model_selection import train_test_split
from distutils.dir_util import copy_tree
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

# Load and preprocess your sign language dataset
# X_train: Training data (sign language images)
# y_train: Training labels (corresponding words)

# Normalize pixel values between 0 and 1
main_path = "videos/"

output_folder = 'split-data/'
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

# with open('WLASL_v0.3.json', 'r') as data_file:
#     json_data = data_file.read()

# instance_json = json.loads(json_data)

wlasl_df["video_ids"] = wlasl_df["instances"].apply(get_videos_ids)

features_df = pd.DataFrame(columns=['gloss', 'video_id', 'urls', 'bbox', 'fps', 'frame_end', 'frame_start','signer_id', 'source', 'split', 'variation_id'])
for row in wlasl_df.iterrows():
    ids, urls, bbox, fps, frame_end, frame_start,signer_id, source, split, variation_id = get_json_features(row[1][1])
    word = [row[1][0]] * len(ids)
    df = pd.DataFrame(list(zip(word, ids, urls, bbox, fps, frame_end, frame_start, signer_id, source, split, variation_id)), columns=features_df.columns)
    # features_df = features_df.append(df)
    features_df = pd.concat([features_df, df])

print("features")
print(features_df)

train_mask = features_df['split'] == 'train'
val_mask = features_df['split'] == 'val'
test_mask = features_df['split'] == 'test'

train_pos = np.flatnonzero(train_mask)
val_pos =  np.flatnonzero(val_mask)
test_pos = np.flatnonzero(test_mask)

train = features_df.iloc[train_pos]

val = features_df.iloc[val_pos]

test = features_df.iloc[test_pos]

print("train")

x_train = train.loc[:, train.columns != 'gloss']
y_train = train['gloss']
# x_train=x_train.iloc[:100]
x_val = val.loc[:, val.columns != 'gloss']
y_val = val['gloss']

x_test = test.loc[:, test.columns != 'gloss']
y_test = test['gloss']

count=0
def loadImageFileAsArray(path,target_size=(64, 64)):
    global count
    file_path=f'split-data/train/{path}'
    frames=[]
    if(os.path.isdir(file_path)):
        for frameF in os.listdir(file_path):
            frame_path=os.path.join(file_path,frameF)
            frame=cv2.imread(frame_path)
            if frame is not None:
                frame = cv2.resize(frame, target_size)
                frames.append(frame)
    #img=cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
    count+=1
    print(count)
    return (np.array(frames))

print(len(x_train),len(x_test), len(x_val))
x_train['img_data']= x_train['video_id'].apply(loadImageFileAsArray)
print(x_train['img_data'])

# Define the model architecture
model = tf.keras.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 10)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, epochs=10, batch_size=30)

# Save the trained model
model.save('sign_language_model.h5')

# Load the trained model
# model = tf.keras.models.load_model('sign_language_model.h5')

# Preprocess your test data
# X_test: Test data (sign language images)

# Normalize pixel values between 0 and 1
# X_test = X_test / 255.0

# # Make predictions
# predictions = model.predict(X_test)

# # Get the predicted word for each sign language image
# predicted_words = [word_list[prediction.argmax()] for prediction in predictions]