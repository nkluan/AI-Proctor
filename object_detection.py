
import os
import time
import tensorflow as tf
import collections
import core.utils as utils
import cv2
import numpy as np

from core.functions import *
from tensorflow.python.saved_model import tag_constants

# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def warning(counted_class):
    if 'cell phone' in counted_class:
        return 'Warning: phone detected'
    if 'person' not in counted_class:
        return 'Warning: no person detected'
    elif counted_class['person'] > 1:
        return "Warning: " + str(counted_class['person']) + " person detected"
    else:
        return ''


def show_fps(image, prev_frame_time):
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    cv2.putText(image, f"fps: {fps}", (1000, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 2, cv2.LINE_AA)
    return new_frame_time
