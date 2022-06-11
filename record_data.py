import cv2
import numpy as np
import time

capture = cv2.VideoCapture("cheating.mp4")
codec = cv2.VideoWriter_fourcc(*'XVID')

recording_flag = False
frame_count = 0
wait_time = 20
count_vid = 0
while True:
    ret, frame_temp = capture.read()
    key = cv2.waitKey(wait_time)
    if not ret:
        print("Can't get frame!")
    if key % 256 == ord('q'):
        break
    elif key % 256 == 32:
        if not recording_flag:
            # we are transitioning from not recording to recording
            vidOutDir = f"video_data/cheating/{time.strftime('%Y%m%d-%H%M%S')}.avi"
            output = cv2.VideoWriter(vidOutDir, codec, 30, (1280, 720))
            count_vid += 1
            recording_flag = True
            print(f"Start writing frame to {vidOutDir}")
        else:
            # transitioning from recording to not recording
            recording_flag = False
            output.release()
            print(f"Stop writing frame to {vidOutDir}")
            frame_count = 0

    if recording_flag:
        wait_time = 20
        frame_count += 1
        if frame_count > 30:
            frame_count = 0
            recording_flag = False
        else:
            output.write(frame_temp)
            print(f"write {frame_count} frames to {vidOutDir}")
            cv2.putText(frame_temp, f"Capturing {frame_count} frames",
                        (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    else:
        wait_time = 20
        cv2.putText(frame_temp, f"Press space to capture 30 frames of the {count_vid}th videos",
                    (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('FRAME', frame_temp)
capture.release()
cv2.destroyAllWindows()
