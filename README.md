# head-pointer

Mouse pointer movement driven by head movement, mouse clicks simulated by eyes' blinks.

## Installation

- [Install dlib](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/)
- `pip install -r requirements.txt`
- [Install PYAutoGUI](http://pyautogui.readthedocs.io/en/latest/install.html)

## How to run

`python video_facial_landmarks.py -p shape_predictor_68_face_landmarks.dat`

where `shape_predictor_68_face_landmarks.dat` is a file containing dlib's pre-trained facial landmarsk detector. It can be downloaded from a tutorial linked below.

## Overview

For the first ~6 seconds, keep your head still in the same position - that's when the initial position of your head is calculated, and head movements will be detected relative to this position.
To move the mouse pointer, slightly move your head in a desired direction.
To simulate mouse click, blink for about 1.5 seconds.

We used [Eye blink detection with OpenCV, Python and dlib](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib) tutorial for blink detection
