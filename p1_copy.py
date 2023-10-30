from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, LSTM, Dense, Flatten,TimeDistributed,Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os
import cv2
from sklearn.model_selection import train_test_split
from distutils.dir_util import copy_tree
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# Initialize the LabelEncoder

main_path = "videos/"
batch_path="batch_data/"

wlasl_df = pd.read_json(batch_path+"batch_1.json")

print(len(wlasl_df))

def get_videos_ids(json_list):
    video_ids = []
    for ins in json_list:
        video_id = ins['video_id']
        if os.path.exists(f'{main_path}{video_id}.mp4'):
            video_ids.append(video_id)
    return video_ids

def get_json_features(json_list):
    videos_ids = []
    videos_bbox = []
    for ins in json_list:
        # print(ins)
        video_id = ins['video_id']
        video_bbox = ins['bbox']
        if os.path.exists(f'{main_path}{video_id}.mp4'):
            videos_ids.append(video_id)
            videos_bbox.append(video_bbox)
    return videos_ids, videos_bbox

with open(batch_path+'batch_1.json', 'r') as data_file:
    json_data = data_file.read()

instance_json = json.loads(json_data)
wlasl_df["video_ids"] = wlasl_df["instances"].apply(get_videos_ids)

features_df = pd.DataFrame(columns=['gloss', 'video_id', 'bbox'])

for row in wlasl_df.iterrows():
    ids, bbox = get_json_features(row[1].iloc[1])
    word = [row[1].iloc[0]] * len(ids)
    df = pd.DataFrame(list(zip(word, ids, bbox)), columns=features_df.columns)
    features_df = pd.concat([features_df, df])

print("features")
print(len(features_df))
print(features_df.head(10))


# Splitting
unique_words = features_df['gloss'].unique()

# Initialize empty train, test, and validation DataFrames
train_data = pd.DataFrame(columns=['gloss', 'video_id', 'bbox'])
test_data = pd.DataFrame(columns=['gloss', 'video_id', 'bbox'])
val_data = pd.DataFrame(columns=['gloss', 'video_id', 'bbox'])

# Iterate through unique words and split the data
for word in unique_words:
    # Get rows where the gloss matches the current word
    word_data = features_df[features_df['gloss'] == word]
    
    if len(word_data) < 3:
        print(f"Skipping word '{word}' due to insufficient samples.")
        continue
    
    # Split the word_data into train, test, and validation sets
    train_word, temp = train_test_split(word_data, test_size=0.4, random_state=42)
    test_word, val_word = train_test_split(temp, test_size=0.5, random_state=42)
    
    # Append the split data to respective DataFrames
    train_data = pd.concat([train_data, train_word])
    test_data = pd.concat([test_data, test_word])
    val_data = pd.concat([val_data, val_word])

# Splitting
print("train len :",len(set(train_data['gloss'])))
print("test len :",len(set(test_data['gloss'])))
print("val len :",len(set(val_data['gloss'])))

print(train_data.head(10))
print(test_data.head(10))
print(val_data.head(10))


# Constants
LEARNING_RATE = 0.001
TARGET_FRAMES = 25
TARGET_WIDTH = 64
TARGET_HEIGHT = 64

# Initialize the model
model = Sequential()

# CNN layers for processing video frames
model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu'), input_shape=(TARGET_FRAMES, TARGET_HEIGHT, TARGET_WIDTH, 3)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))

# LSTM layer for sequential modeling
model.add(LSTM(64, return_sequences=False))

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(53, activation='softmax'))  # num_classes represents the number of action classes

# Compile the model
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Function to preprocess video frames and bounding boxes
def preprocess_video(video_path, target_frames=25, target_width=64, target_height=64):
    print("working for : ",video_path)
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)
    cap.release()
    cv2.destroyAllWindows()

    # Padding or truncating frames to target_frames
    if len(frames) > target_frames:
        frames = frames[:target_frames]
    elif len(frames) < target_frames:
        frames.extend([np.zeros((target_height, target_width, 3), dtype=np.uint8)] * (target_frames - len(frames)))

    return np.array(frames)  # Return frames as a sequence


def prepare_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels,num_classes=53)
    return categorical_labels

# train
x_train = np.array([preprocess_video(main_path + video_id + '.mp4') for video_id in train_data['video_id']])
y_train = prepare_labels(train_data['gloss'])
# test
x_test = np.array([preprocess_video(main_path + video_id + '.mp4') for video_id in test_data['video_id']])
y_test = prepare_labels(test_data['gloss'])

# val
x_val = np.array([preprocess_video(main_path + video_id + '.mp4') for video_id in val_data['video_id']])
y_val = prepare_labels(val_data['gloss'])


print(x_train[0]) 
print(y_train[0]) 
print(y_train[100]) 


model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=16)

# Example usage of preprocess_video function
video_path = main_path+"17710.mp4"

processed_frames = preprocess_video(video_path)

# Reshape frames for the model input (add batch dimension)
processed_frames = np.expand_dims(processed_frames, axis=0)

# Predict using the model
predictions = model.predict(processed_frames)
print(predictions)

model.save('sign_language_model.h5')