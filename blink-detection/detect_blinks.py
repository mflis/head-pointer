import argparse
import time

import cv2
import dlib
# import the necessary packages
import imutils
import numpy as np
from imutils import face_utils
from helpers.webcam import WebcamVideoStream

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = WebcamVideoStream().start()
fileStream = False
time.sleep(1.0)

# loop over frames from the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
        break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    # frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEyeCorner = shape[39]
        leftEyebrowCorner = shape[21]
        noseTop = shape[27]
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        x, y, w, h = cv2.boundingRect(rightEyeHull)
        cropped = gray[y - 5:y + h + 5, x - 1:x + w + 1]
        cropped_color = frame[y - 5:y + h + 5, x - 1:x + w + 1]
        inverted_crop = cv2.bitwise_not(cropped)
        blurred = cv2.GaussianBlur(inverted_crop, (5, 5), 5)
        # binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY, 9, 0)
        ret, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # print( contours)
        # eye_ball_hull = cv2.convexHull(np_contours)
        # ret, thresh = cv2.threshold(inverted_crop, 127, 255, 0)
        # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # np_contours  = np.array(contours)
        img_seq = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in [inverted_crop, blurred, binary]]
        img = np.vstack(img_seq)
        # cv2.imshow("Frame", cropped_color)
        cv2.namedWindow("Frame2",cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame2',400,400)
        cv2.imshow("Frame2", np.vstack((cropped_color, img)))

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [eye_ball_hull], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    # show the frame
    # cv2.imshow("Frame", masked_image)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        import sys

        sys.exit(0)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
