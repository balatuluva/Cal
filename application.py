from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
mp_pose = mp.solutions.pose

@app.route('/api/calculate', methods=['POST'])
def calculate_dimensions():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Load image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Initialize MediaPipe Pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return jsonify({'error': 'No body detected'}), 400

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Example calculation (these need scaling based on reference height)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x) * image.shape[1]
        hip_width = abs(left_hip.x - right_hip.x) * image.shape[1]
        height = abs(nose.y - left_hip.y) * image.shape[0]

        dimensions = {
            'height': f'{height:.2f} pixels',
            'shoulder_width': f'{shoulder_width:.2f} pixels',
            'hip_width': f'{hip_width:.2f} pixels',
        }

        return jsonify(dimensions)

if __name__ == '__main__':
    app.run(debug=True)
