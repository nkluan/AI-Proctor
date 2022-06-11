import cv2
import mediapipe as mp
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
iw, ih = 1280, 720


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def show_fps(image, prev_frame_time):
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    cv2.putText(image, f"fps: {fps}", (1000, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    return new_frame_time


def draw_face_landmarks(image, face_landmarks):
    # mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, None, mp_drawing_styles.get_default_face_mesh_contours_style())
    # mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, None, mp_drawing_styles.get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_LIPS, None, mp_drawing_styles.get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_IRISES, None,
                              mp_drawing_styles.get_default_face_mesh_iris_connections_style())


def draw_hand_landmarks(image, hand_landmarks):
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                              mp_drawing_styles.get_default_hand_landmarks_style(),
                              mp_drawing_styles.get_default_hand_connections_style())


def get_scaled_landmarks(landmarks, dimenson):
    landmarks_2d = []
    landmarks_3d = []
    if dimenson == '2d':
        for landmark in landmarks:
            x, y = int(landmark.x * iw), int(landmark.y * ih)
            landmarks_2d.append([x, y])
        return landmarks_2d
    if dimenson == 'both':
        for landmark in landmarks:
            x, y = int(landmark.x * iw), int(landmark.y * ih)
            landmarks_2d.append([x, y])
            landmarks_3d.append([x, y, landmark.z])
        return landmarks_2d, landmarks_3d


def get_eyes_movement(eyes_landmarks):
    eyes_landmarks = get_scaled_landmarks(eyes_landmarks, '2d')
    right_most, right_iris, right_inner, left_inner, left_iris, left_most = eyes_landmarks
    try:
        right_ratio = abs(right_iris[0] - right_most[0]) / abs(right_most[0] - right_inner[0])
        left_ratio = abs(left_iris[0] - left_inner[0]) / abs(left_inner[0] - left_most[0])
    except:
        return 0
    # look right
    if right_ratio < 0.35 and left_ratio < 0.35:
        return "peeking right"
    # look left
    elif right_ratio > 0.65 and left_ratio > 0.65:
        return "peeking left"
    else:
        return ""


def warning_3s(movements):
    movements = np.array(movements).reshape(3, 7)
    warning_count = 0
    warning = ""
    for second in movements:
        for movement in second:
            if movement != "":
                warning_count += 1
                warning = "Warning: " + movement
                break
    return warning if warning_count == 3 else ""


def get_head_movement(image, face_keypoints):
    face_2d, face_3d = get_scaled_landmarks(face_keypoints, 'both')
    nose_2d = face_2d[0]
    face_2d = np.array(face_2d, dtype=np.float64)[1:]

    face_3d = np.array(face_3d, dtype=np.float64)[1:]
    focal_length = 1 * iw
    cam_matrix = np.array([[focal_length, 0, ih / 2],
                           [0, focal_length, iw / 2],
                           [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    if y < -10:
        text = "turning left"
    elif y > 10:
        text = "turning right"
    elif x < -5:
        text = "turning down"
    else:
        text = ""

    # for point in face_2d:
    #     cv2.circle(image, (int(point[0]), int(point[1])), radius=2, color=(255, 0, 0), thickness=2)
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
    cv2.line(image, p1, p2, (255, 0, 0), 3)
    # Add the text on the image
    cv2.putText(image, "Vertical angle: " + str(np.round(x, 2)), (900, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, "Horizontal angle: " + str(np.round(y, 2)), (900, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return text


def get_limit_hand_coordinate(hand_landmarks):
    x = []
    y = []
    for lm in hand_landmarks:
        x.append(lm.x)
        y.append(lm.y)

    hand_limit = int(min(x) * iw), int(max(x) * iw), int(min(y) * ih), int(max(y) * ih)
    return hand_limit


def get_hand_movement(face_hands, hand_limit):
    x_min, x_max, y_min, y_max = hand_limit
    face_hands = get_scaled_landmarks(face_hands, '2d')
    warning = ""
    for lm in face_hands:
        if lm[0] > x_min and lm[0] < x_max and lm[1] > y_min and lm[1] < y_max:
            warning = "face occlusion"
            break
    return warning


def get_mouth_movement(upper_mouth, bottom_mouth):
    distance = abs(int(upper_mouth.y * ih) - int(bottom_mouth.y * ih))
    if distance > 15:
        return "talking"
    else:
        return ""


def warning_display(warnings):
    for warning in warnings:
        if warning != "":
            warnings = [""]
            return warning, warnings
    return "", [""]
