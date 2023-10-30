from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, LSTM, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os
import cv2
from sklearn.model_selection import train_test_split
# from nqdm import nqdm
# from IPython.display import FileLink
from distutils.dir_util import copy_tree
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# Initialize the LabelEncoder

main_path = "videos/"
batch_path="batch_data/"

wlasl_df = pd.read_json(batch_path+"batch_1.json")

print(len(wlasl_df))

# print(wlasl_df.head(5))

def get_videos_ids(json_list):
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

with open(batch_path+'batch_1.json', 'r') as data_file:
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

# print("features")

# print(len(features_df))

# print(features_df.head(10))


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


def preprocess_video(video_path, target_frames=30, target_width=64, target_height=64):
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

num_classes = len(set(y_train))

LEARNING_RATE = 0.001

# Preprocess the video data
x_train_processed = np.array([preprocess_video(main_path + video_id + '.mp4') for video_id in x_train['video_id']])
x_val_processed = np.array([preprocess_video(main_path + video_id + '.mp4') for video_id in x_val['video_id']])
x_test_processed = np.array([preprocess_video(main_path + video_id + '.mp4') for video_id in x_test['video_id']])

# Reshape the processed frames for LSTM input
x_train_reshaped = x_train_processed.reshape(x_train_processed.shape[0], x_train_processed.shape[1], -1)
x_val_reshaped = x_val_processed.reshape(x_val_processed.shape[0], x_val_processed.shape[1], -1)
x_test_reshaped = x_test_processed.reshape(x_test_processed.shape[0], x_test_processed.shape[1], -1)

# Check the reshaped data shape
print(x_train_reshaped.shape)  # It should be (333, 30, 12288)

# Encode the target labels to numeric values
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

y_train_encoded = to_categorical(y_train, num_classes=333)
y_val_encoded = to_categorical(y_val, num_classes=69)
y_test_encoded = to_categorical(y_test, num_classes=38)

print("Ytest len",len(y_train_encoded),y_train_encoded)
print("Yval len",len(y_val_encoded),y_val_encoded)
print("Ytest len",len(y_val_encoded),y_test_encoded)

# Define the CNN-LSTM model
model = Sequential()

# LSTM layer
model.add(LSTM(64, input_shape=(30, 64*64*3), return_sequences=False))

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(57, activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train_reshaped, y_train_encoded,epochs=10, batch_size=16)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test_reshaped, y_test_encoded, batch_size=16)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

model.save('sign_language_model.h5')

# # Define the CNN-LSTM model
# model = Sequential()

# # CNN layers
# model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(None, 64, 64, 3), padding='same', activation='relu'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))

# model.add(Flatten())

# # LSTM layer
# model.add(LSTM(64, return_sequences=False))

# # Fully connected layers
# model.add(Dense(128, activation='relu'))
# model.add(Dense(NUM_CLASSES, activation='softmax'))

# # Compile the model
# optimizer = Adam(lr=LEARNING_RATE)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=16)

# # Example usage of preprocess_video function
# video_path = main_path+"17710.mp4"

# processed_frames = preprocess_video(video_path)

# # Padding sequences if necessary
# processed_frames = pad_sequences([processed_frames], maxlen=30, padding='post', truncating='post')[0]

# # Reshape frames for the model input (add batch dimension)
# processed_frames = np.expand_dims(processed_frames, axis=0)

# # Predict using the model
# predictions = model.predict(processed_frames)
# print(predictions)
