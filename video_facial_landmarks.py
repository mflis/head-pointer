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
import imutils
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    return (A + B) / (2.0 * C)


pyautogui.FAILSAFE = False
BLINK_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

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

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
time.sleep(2.0)

frame_count = 0
vert_zero = 0
horiz_zero = 0
vert_acc = 0
horiz_acc = 0

blink_frame_counter = 0
total_blinks = 0


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame,width= 400)
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
        if frame_count < 100:
            vert_acc += vertical_diff
            horiz_acc += horizontal_diff
        else:
            horiz_zero = int(horiz_acc / 100)
            vert_zero = int(vert_acc / 100)
            h_movement = 0 if abs(h_diff) < 7 else h_diff
            v_movement = 0 if abs(v_diff) < 3 else v_diff
            pyautogui.moveRel(h_movement, v_movement)

        left_eye = shape[left_eye_start:left_eye_end]
        right_eye = shape[right_eye_start:right_eye_end]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        if ear < BLINK_AR_THRESH:
            blink_frame_counter += 1
        else:
            if blink_frame_counter > EYE_AR_CONSEC_FRAMES:
                # blink is detected
                total_blinks += 1
                pyautogui.click()
            blink_frame_counter = 0

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

        cv2.putText(frame, "Blinks: {}".format(total_blinks), (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "left eye: {:.2f}".format(eye_aspect_ratio(left_eye)), (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "right eye: {:.2f}".format(eye_aspect_ratio(right_eye)), (300, 150),
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
