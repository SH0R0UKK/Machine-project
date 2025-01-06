import mediapipe as mp
import cv2
import numpy as np
import os
import csv
from sklearn.preprocessing import LabelEncoder

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Gait types
gait_types = ['ataxic_gait', 'choreiform_gait', 'hemiplegic_gait', 'normal_gait', 'parkinsonian_gait', 'steppage_gait']
label_encoder.fit(gait_types)  # Fit encoder to gait types

# Map of body parts from shoulders down
body_part_indices = list(range(11, 33))  # Landmarks from 'left_shoulder' (11) to 'right_foot_index' (32)

def extract_keypoints_from_video(video_path, view_type):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            keypoints = []
            # Extract only keypoints from shoulders down
            for i in body_part_indices:
                landmark = results.pose_landmarks.landmark[i]
                keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            # Add the view type as an additional feature
            keypoints.append(view_type)  # Add "front" or "side"
            keypoints_sequence.append(keypoints)
    
    cap.release()
    return np.array(keypoints_sequence)

# Paths
data_dir = r"D:\gait_videos"  # Use raw string for Windows path
output_csv = r"D:\keypoints_labels.csv"  # Use raw string for Windows path

# Open a CSV file for writing keypoints with labels
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row (video name + label + keypoints for each frame + view type)
    header = ['video_name'] + ['label'] + [f'{mp_pose.PoseLandmark(i).name}_x' for i in body_part_indices] + \
             [f'{mp_pose.PoseLandmark(i).name}_y' for i in body_part_indices] + \
             [f'{mp_pose.PoseLandmark(i).name}_z' for i in body_part_indices] + \
             [f'{mp_pose.PoseLandmark(i).name}_visibility' for i in body_part_indices] + ['view_type']
    writer.writerow(header)

    # Process each gait type and its corresponding videos
    for gait_type in gait_types:
        label = label_encoder.transform([gait_type])[0]  # Encode gait type to numeric label
        gait_path = os.path.join(data_dir, gait_type)
        
        for view in ['front', 'side']:  # Loop through both 'front' and 'side' views
            view_path = os.path.join(gait_path, view)
            
            for video_file in os.listdir(view_path):
                if video_file.endswith(".mp4"):
                    video_path = os.path.join(view_path, video_file)
                    keypoints_sequence = extract_keypoints_from_video(video_path, view)
                    
                    # Store each frame's keypoints with the gait label and view type
                    for keypoints in keypoints_sequence:
                        row = [video_file] + [label] + keypoints.tolist()
                        writer.writerow(row)
                    print(f"Processed {video_file} (View: {view}) and added keypoints with label {label}")

print("Keypoints extraction and labeling completed successfully.")
