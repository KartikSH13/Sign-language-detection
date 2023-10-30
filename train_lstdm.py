import cv2
import json
import numpy as np
import os
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling3D, Dense, Flatten, TimeDistributed, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical


warnings.filterwarnings("ignore")


# Define your constants
LEARNING_RATE = 0.001
TARGET_FRAMES = 25
TARGET_WIDTH = 64
TARGET_HEIGHT = 64

main_path = "videos/"
batch_path = "batch_data/"
model_save_path = "models/wlasl_lstm.h5"
label_save_path = "labels/labels_lstm.npy"

# Load your data
wlasl_df = pd.read_json(batch_path + "batch_1.json")

# Preprocess your data
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

unique_words = features_df['gloss'].unique()
labels_array = np.array(unique_words)

# Initialize empty train, test, and validation DataFrames
train_data = pd.DataFrame(columns=['gloss', 'video_id', 'bbox'])
test_data = pd.DataFrame(columns=['gloss', 'video_id', 'bbox'])
val_data = pd.DataFrame(columns=['gloss', 'video_id', 'bbox'])

from sklearn.model_selection import train_test_split
import pandas as pd

train_data = pd.DataFrame()  # Initialize an empty DataFrame for training data
test_data = pd.DataFrame()   # Initialize an empty DataFrame for testing data
val_data = pd.DataFrame()    # Initialize an empty DataFrame for validation data

for word in unique_words:
    word_data = features_df[features_df['gloss'] == word]

    if len(word_data) < 3:
        print(f"Skipping word '{word}' due to insufficient samples.")
        continue

    # Ensure there are at least 3 samples for testing and validation
    if len(word_data) < 6:
        print(f"Skipping word '{word}' due to insufficient samples for testing and validation.")
        continue

    train_word, temp = train_test_split(word_data, test_size=0.33, random_state=42)
    test_word, val_word = train_test_split(temp, test_size=0.22, random_state=42)

    train_data = pd.concat([train_data, train_word])
    test_data = pd.concat([test_data, test_word])
    val_data = pd.concat([val_data, val_word])


print("train len :",len(set(train_data['gloss'])))
print("test len :",len(set(test_data['gloss'])))
print("val len :",len(set(val_data['gloss'])))

num_classes = len(set(train_data['gloss']))

# Define your model
def model_init():
    model = Sequential()
    
    model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh' ,data_format ="channels_last",recurrent_dropout=0.2, return_sequences=True, input_shape = (TARGET_FRAMES,TARGET_HEIGHT, TARGET_WIDTH,3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh' ,data_format ="channels_last",recurrent_dropout=0.2, return_sequences=True, input_shape = (TARGET_FRAMES,TARGET_HEIGHT, TARGET_WIDTH,3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(ConvLSTM2D(filters=14, kernel_size=(3, 3), activation='tanh' ,data_format ="channels_last",recurrent_dropout=0.2, return_sequences=True, input_shape = (TARGET_FRAMES,TARGET_HEIGHT, TARGET_WIDTH,3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh' ,data_format ="channels_last",recurrent_dropout=0.2, return_sequences=True, input_shape = (TARGET_FRAMES,TARGET_HEIGHT, TARGET_WIDTH,3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

model=model_init()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Model created successfully")
model.summary()

# Function to preprocess video frames and bounding boxes
def preprocess_video(video_path,bounding_box, target_frames=25, target_width=64, target_height=64):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x, y, w, h = bounding_box
        cropped_frame = frame[y:y+h, x:x+w]
        # Resize cropped frame to target dimensions
        cropped_frame = cv2.resize(cropped_frame, (target_width, target_height))
        normalized_frame=cropped_frame / 255
        frames.append(normalized_frame)
    cap.release()

    if len(frames) > target_frames:
        frames = frames[:target_frames]
    elif len(frames) < target_frames:
        frames.extend([np.zeros((target_height, target_width, 3), dtype=np.uint8)] * (target_frames - len(frames)))

    return np.array(frames)

def prepare_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)
    return categorical_labels


# Train
x_train = np.array([preprocess_video(main_path + video_id + '.mp4', bounding_box=(x, y, width, height)) for video_id, (x, y, width, height) in zip(train_data['video_id'], train_data['bbox'])])

y_train = prepare_labels(train_data['gloss'])

# Test
x_test = np.array([preprocess_video(main_path + video_id + '.mp4', bounding_box=(x, y, width, height)) for video_id, (x, y, width, height) in zip(test_data['video_id'], test_data['bbox'])])

y_test = prepare_labels(test_data['gloss'])

# Validation
x_val = np.array([preprocess_video(main_path + video_id + '.mp4', bounding_box=(x, y, width, height)) for video_id, (x, y, width, height) in zip(val_data['video_id'], val_data['bbox'])])

y_val = prepare_labels(val_data['gloss'])

# Train your model with data augmentation using the custom generator
history = model.fit(x_train,y_train, validation_data=(x_val, y_val), epochs=20)

# Save the trained model
model.save(model_save_path)

# Save the labels/classes to a file
np.save(label_save_path, labels_array)


