#!/usr/bin/python3

"""
Detect any motion in the frame.

YunFei Robotics Labrotary
Twitter: @yfrobotics
Website: https://www.yfrl.org
"""
# 160 x 120
# 320 x 240
# 640 x 480 (480p)
# 1280 x 720 (720p)
# 1920 x 1080 (1080p; make sure your camera supports this high resolution.)

import cv2
import time
import numpy as np

CAMERA_DEVICE_ID = 0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
MOTION_BLUR = True

cnt_frame = 0


def mse(image_a, image_b):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


if __name__ == "__main__":
    try:
        # create video capture
        cap = cv2.VideoCapture(CAMERA_DEVICE_ID)

        # set resolution to 320x240 to reduce latency 
        cap.set(3, IMAGE_WIDTH)
        cap.set(4, IMAGE_HEIGHT)

        while True:
            # ----------------------------------------------------------------------
            # record start time
            start_time = time.time()
            # ----------------------------------------------------------------------
            # Read the frames from a camera
            _, frame_raw = cap.read()

            if MOTION_BLUR:
                # Denoise the frame
                frame = cv2.GaussianBlur(frame_raw, (3,3),0)
            else:
                frame = frame_raw

            # Convert to gray image
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find edges
            edges = cv2.Canny(frame_gray,100,200)

            # Show the original and processed image
            cv2.imshow('gray', frame_gray)
            cv2.imshow('edge', edges)

            # Calculate MSE
            if cnt_frame > 0:
                if mse(frame_gray, frame_gray_p) > 100:
                    print('Frame{0}: Motion Detected!'.format(cnt_frame))

            # ----------------------------------------------------------------------
            # record end time
            end_time = time.time()

            # calculate FPS
            seconds = end_time - start_time
            fps = 1.0 / seconds
            # print("Estimated fps:{0:0.1f}".format(fps));

            cnt_frame = cnt_frame + 1
            edges_p = edges
            frame_gray_p = frame_gray
            # ----------------------------------------------------------------------

            # if key pressed is 'Esc' then exit the loop
            if cv2.waitKey(1)== 27:
                break
    except Exception as e:
        print(e)
    finally:
        # Clean up and exit the program
        cv2.destroyAllWindows()
        cap.release()
