import threading

import imutils

from pose_fuctions import *
from object_detection import *
from face_functions import *

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap2 = cv2.VideoCapture(1)
cap2.set(3, 1280)
cap2.set(4, 720)


def face_system():
    prev_frame_time = 0
    output = None
    codec = cv2.VideoWriter_fourcc(*'XVID')
    frame_rec = 0
    recording = False
    new_dir = True
    eyes_movements, head_movements, mouth_movements, hand_movements = [], [], [], []
    warnings = [""]
    saved_model_loaded = tf.saved_model.load('yolov4-tiny-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    warning_list = []
    with mp_hands.Hands(model_complexity=0) as hands:
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print('Ignore empty camera frame!')
                    break

                image = cv2.flip(frame, 1)
                image, results = mediapipe_detection(image, face_mesh)
                image, hand_results = mediapipe_detection(image, hands)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        lmks = face_landmarks.landmark

                        eyes_landmarks = [lmks[263], lmks[473], lmks[362], lmks[133], lmks[468], lmks[33]]
                        eyes_movement = get_eyes_movement(eyes_landmarks)
                        eyes_movements.append(eyes_movement)
                        eyes_movements = eyes_movements[-21:]

                        face_keypoints = [lmks[1], lmks[33], lmks[263], lmks[61], lmks[291], lmks[199]]
                        head_movement = get_head_movement(image, face_keypoints)
                        head_movements.append(head_movement)
                        head_movements = head_movements[-21:]

                        mouth_movement = get_mouth_movement(lmks[13], lmks[14])
                        mouth_movements.append(mouth_movement)
                        mouth_movements = mouth_movements[-21:]

                        if len(eyes_movements) == 21:
                            warn_eyes = warning_3s(eyes_movements)
                            warn_head = warning_3s(head_movements)
                            warn_mouth = warning_3s(mouth_movements)
                            warnings.extend([warn_eyes, warn_head, warn_mouth])

                        draw_face_landmarks(image, face_landmarks)
                        face_occlusion_points = [lmks[13], lmks[8]]

                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                hand_lmks = hand_landmarks.landmark
                                hand_limit = get_limit_hand_coordinate(hand_lmks)
                                hand_movement = get_hand_movement(face_occlusion_points, hand_limit)
                                hand_movements.append(hand_movement)
                                hand_movements = hand_movements[-21:]

                                if len(hand_movements) == 21:
                                    warn_hand = warning_3s(hand_movements)
                                    warnings.append(warn_hand)
                                draw_hand_landmarks(image, hand_landmarks)
                        else:
                            hand_movements.append("")
                            hand_movements = hand_movements[-21:]

                warning_info, warnings = warning_display(warnings)
                if warning_info != "":
                    cv2.putText(image, warning_info, (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (45, 255, 255), 2,
                                cv2.LINE_AA)
                    recording = True
                    if new_dir:
                        vidOutDir = f"Log Video/{time.strftime('%Y%m%d-%H%M%S')}-{warning_info[9:]}.avi"
                        output = cv2.VideoWriter(vidOutDir, codec, 15, (1280, 720))
                        new_dir = False
                prev_frame_time = show_fps(image, prev_frame_time)
                if recording:
                    if frame_rec <= 600:
                        output.write(image)
                        frame_rec += 1
                    else:
                        frame_rec = 0
                        recording = False
                        output.release()
                        new_dir = True

                image_data = cv2.resize(cv2.flip(frame, 1), (416, 416))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)
                start_time = time.time()

                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=0.25,
                    score_threshold=0.2
                )

                # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
                original_h, original_w, _ = frame.shape
                bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

                pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

                # custom allowed classes (uncomment line below to allow detections for only people)
                # allowed_classes = ['person', ' ', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book']
                allowed_classes = ['person', 'cell phone']

                # count objects found
                counted_classes = count_objects(pred_bbox, by_class=True, allowed_classes=allowed_classes)

                warning_list.append(warning(counted_classes))
                warning_list = warning_list[-15:]
                if len(warning_list) == 15:
                    warning_list_dict = dict(collections.Counter(warning_list))
                    if '' not in warning_list_dict or warning_list_dict[''] < 5:
                        cv2.putText(image, warning(counted_classes), (7, 700), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    (45, 255, 255),
                                    2, cv2.LINE_AA)

                utils.draw_bbox(image, pred_bbox, False, counted_classes, allowed_classes=allowed_classes)
                cv2.imshow("Front camera", imutils.resize(image, 1000))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


def pose_system():
    prev_frame_time = 0
    input_sequence = []
    predictions = []

    frame_rec = 0
    output = None
    codec = cv2.VideoWriter_fourcc(*'XVID')

    recording = False
    new_dir = True

    with mp_pose.Pose() as pose:
        while cap2.isOpened():

            # Read feed
            ret, frame = cap2.read()
            if not ret:
                print("Can't get frame!")
                # break

            # Make detections
            image, results = mediapipe_detection(frame, pose)

            # Draw landmarks
            if results.pose_landmarks:
                draw_landmarks(image, results)

            frame_landmarks = get_frame_landmarks(results)
            input_sequence.append(frame_landmarks)
            input_sequence = input_sequence[-30:]
            if len(input_sequence) == 30:
                res = model.predict(np.expand_dims(input_sequence, axis=0))[0]
                cheating_prob = round(res[1], 2)

                if cheating_prob > 0.8:
                    predictions.append(1)
                    cv2.putText(image, "Cheating probs: " + str(cheating_prob), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                ((0, 0, 255)), 2, cv2.LINE_AA)
                else:
                    predictions.append(0)
                    cv2.putText(image, "Cheating probs: " + str(cheating_prob), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                ((255, 0, 0)), 2, cv2.LINE_AA)

                predictions = predictions[-20:]
                prediction_dict = dict(collections.Counter(predictions))
                if 1 in prediction_dict and prediction_dict[1] > 15:
                    cv2.putText(image, "Warning: suspicous behavior", (7, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (45, 255, 255),
                                2, cv2.LINE_AA)
                    recording = True
                    if new_dir:
                        vidOutDir = f"Log Video/{time.strftime('%Y%m%d-%H%M%S')}-suspicious action.avi"
                        output = cv2.VideoWriter(vidOutDir, codec, 12, (1280, 720))
                        new_dir = False
            # Show fps
            prev_frame_time = show_fps(image, prev_frame_time)

            if recording:
                if frame_rec <= 600:
                    output.write(image)
                    frame_rec += 1
                else:
                    frame_rec = 0
                    recording = False
                    new_dir = True
                    output.release()
            # Show to screen
            cv2.imshow('OpenCV Feed', imutils.resize(image, 1000))
            # Break gracefully
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


t1 = threading.Thread(target=face_system)
# t2 = threading.Thread(target=pose_system)
t1.start()
# t2.start()

t1.join()
# t2.join()

cap.release()
cap2.release()
cv2.destroyAllWindows()
