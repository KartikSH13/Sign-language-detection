import cv2
import numpy as np
import tensorflow as tf
import argparse

def preprocess_video(video_path, target_frames=25, target_width=64, target_height=64):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_width, target_height))
        frame = frame /255.0
        frames.append(frame)

    if len(frames) > target_frames:
        frames = frames[:target_frames]
    elif len(frames) < target_frames:
        frames.extend([np.zeros((target_height, target_width, 3), dtype=np.uint8)] * (target_frames - len(frames)))

    cap.release()

    return np.array(frames)

def predict_video(video_path, model,class_labels):
    # Load the saved TensorFlow model
    model = tf.keras.models.load_model(model)

    # Initialize the frame sequence
    frame_sequence = []

    # Process the video frames
    for frame in preprocess_video(video_path):
        frame_sequence.append(frame)

        if len(frame_sequence) > 25:
            frame_sequence.pop(0)

        if len(frame_sequence) == 25:
            # Reshape the frame sequence for model input (add batch and time dimensions)
            processed_sequence = np.expand_dims(frame_sequence, axis=0)

            # Make predictions using your model
            predictions = model.predict(processed_sequence)

            print(predictions)
            # Convert predictions to human-readable labels
            predicted_label_indices = np.argmax(predictions, axis=1)
            predicted_labels = [class_labels[i] for i in predicted_label_indices]

            # Display the frame with the predicted label
            print(predicted_labels)  # Print the predicted labels

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict action in a video')
    parser.add_argument('video_path', help='Path to the video to predict')
    parser.add_argument('model', help='Path to the trained model')
    parser.add_argument('label',help='Path to the trained labels')

    args = parser.parse_args()
    video_path = args.video_path
    model = args.model
    class_labels=np.load(args.label, allow_pickle=True)
    
    predict_video(video_path, model,class_labels)
