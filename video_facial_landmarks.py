# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat

import argparse
import time

import cv2
import dlib
import pyautogui
from imutils import face_utils
# import the necessary packages
from imutils.video import VideoStream

pyautogui.FAILSAFE = False

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
time.sleep(2.0)

frame_count = 0
vert_zero = 0
horiz_zero = 0
vert_acc = 0
horiz_acc = 0

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame_count += 1
    # frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    for rect in rects:

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEyeCorner = shape[39]
        leftEyebrowCorner = shape[21]
        noseTop = shape[27]

        vertical_diff = leftEyeCorner[1] - leftEyebrowCorner[1]
        horizontal_diff = noseTop[0] - leftEyeCorner[0]
        v_diff = vert_zero - vertical_diff
        h_diff = horiz_zero - horizontal_diff

        if frame_count < 10:
            vert_acc += vertical_diff
            horiz_acc += horizontal_diff
        else:
            horiz_zero = int(horiz_acc / 10)
            vert_zero = int(vert_acc / 10)
            pyautogui.moveRel(h_diff, v_diff)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        cv2.putText(frame, "vertical diff: {}".format(vertical_diff), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "horiz diff {}".format(horizontal_diff), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "vert zero: {}".format(vert_zero), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "horiz zero {}".format(horiz_zero), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "v_diff: {}".format(v_diff), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "h_diff {}".format(h_diff), (300, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
