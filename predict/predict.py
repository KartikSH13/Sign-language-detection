import cv2
import numpy as np
import tensorflow as tf

# Load the saved TensorFlow model
model = tf.keras.models.load_model('../models/wlasl.h5')

print("Model loaded")

# Constants
TARGET_FRAMES = 25
TARGET_WIDTH = 64
TARGET_HEIGHT = 64

# Load class labels from 'labels.npy'
class_labels = np.load('../labels/labels.npy', allow_pickle=True)

def preprocess_video(frame):
    # Resize and preprocess the input frame to match the training data
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    frame = frame / 255.0  # Normalize pixel values to the range [0, 1]
    return frame

# Initialize the video capture object (0 represents the default camera)
cap = cv2.VideoCapture(0)

frame_sequence = []  # Initialize an empty list for the frame sequence

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    processed_frame = preprocess_video(frame)

    # Add the processed frame to the sequence
    frame_sequence.append(processed_frame)

    # Maintain the frame sequence's length
    if len(frame_sequence) > TARGET_FRAMES:
        frame_sequence.pop(0)

    if len(frame_sequence) == TARGET_FRAMES:
        # Reshape the frame sequence for model input (add batch and time dimensions)
        processed_sequence = np.expand_dims(frame_sequence, axis=0)

        # Make predictions using your model
        predictions = model.predict(processed_sequence)

        # Convert predictions to human-readable labels
        predicted_label_indices = np.argmax(predictions, axis=1)
        predicted_labels = [class_labels[i] for i in predicted_label_indices]

        print(predicted_labels[0])
        # Display the frame with the predicted label
        cv2.putText(frame, predicted_labels[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Live Prediction', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
