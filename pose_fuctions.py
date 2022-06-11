import cv2
import numpy as np
import os
import time
import mediapipe as mp
import time
import tensorflow as tf
import collections
import os

from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

actions = np.array(['non_cheating', 'cheating'])
mp_pose = mp.solutions.pose

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 191)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.load_weights('pose_weights.h5')

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def get_scaled_landmarks(landmarks, dimenson):
    landmarks_2d = []
    landmarks_3d = []
    if dimenson == '2d':
        for landmark in landmarks:
            x, y = int(landmark.x*1280), int(landmark.y*720)
            landmarks_2d.append([x, y])
        return landmarks_2d
    if dimenson == 'both':
        for landmark in landmarks:
            x, y = int(landmark.x*1280), int(landmark.y*720)
            landmarks_2d.append([x, y])
            landmarks_3d.append([x, y, landmark.z])
        return landmarks_2d, landmarks_3d


def draw_landmarks(image, results):
    lmks = results.pose_landmarks.landmark
    pose_landmarks = [lmks[0], lmks[11], lmks[12], lmks[13], lmks[14], lmks[15], lmks[16], lmks[23], lmks[24], lmks[19],
                      lmks[20]]
    pose_landmarks = get_scaled_landmarks(pose_landmarks, '2d')

    cv2.line(image, tuple(pose_landmarks[1]), tuple(pose_landmarks[2]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[1]), tuple(pose_landmarks[3]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[3]), tuple(pose_landmarks[5]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[2]), tuple(pose_landmarks[4]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[4]), tuple(pose_landmarks[6]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[1]), tuple(pose_landmarks[7]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[2]), tuple(pose_landmarks[8]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[7]), tuple(pose_landmarks[8]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[5]), tuple(pose_landmarks[9]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[6]), tuple(pose_landmarks[10]), (255, 255, 255), 2)
    for lm in pose_landmarks:
        cv2.circle(image, (int(lm[0]), int(lm[1])), 4, (0, 0, 255), -1)

def show_fps(image, prev_frame_time):
    new_frame_time = time.time()
    fps = int(1/(new_frame_time-prev_frame_time))
    cv2.putText(image, f"fps: {fps}", (1000, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    return new_frame_time

def get_joint_angle(a, b, c):
    angle = np.abs(np.arctan2(c.y-b.y, c.x-b.x) - np.arctan2(a.y-b.y, a.x-b.x))
    if angle > np.pi:
        angle = 2*np.pi-angle
    return angle

def get_all_angles(landmarks):
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

    right_elbow_angle = get_joint_angle(right_shoulder, right_elbow, right_wrist)
    righ_shoulders_angle = get_joint_angle(right_elbow, right_shoulder, left_shoulder)
    left_elbow_angle = get_joint_angle(left_shoulder, left_elbow, left_wrist)
    left_shoulders_angle = get_joint_angle(left_elbow, left_shoulder, right_shoulder)
    nose_angle = get_joint_angle(left_shoulder, nose, right_shoulder)
    left_ear_angle = get_joint_angle(left_shoulder, left_ear, right_shoulder)
    right_ear_angle = get_joint_angle(left_shoulder, right_ear, right_shoulder)
    angles = [right_elbow_angle, righ_shoulders_angle, left_elbow_angle, left_shoulders_angle, nose_angle, left_ear_angle, right_ear_angle]
    return angles

def get_frame_landmarks(results):
    size_landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark[:23]]).flatten() if results.pose_landmarks else np.zeros(4*23)
    world_landmarks =  np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark[:23]]).flatten() if results.pose_world_landmarks else np.zeros(4*23)
    angles = np.array(get_all_angles(results.pose_landmarks.landmark)) if results.pose_landmarks else np.zeros(7)
    landmarks = np.concatenate([size_landmarks, world_landmarks, angles])
    return landmarks