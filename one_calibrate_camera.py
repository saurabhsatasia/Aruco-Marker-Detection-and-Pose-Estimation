import cv2
import streamlit as st
from cv2 import aruco
from cv2.aruco import CharucoBoard_create, Dictionary_get
import pickle
import glob
import pandas as pd

class Camera_Calibration:

    def __init__(self):
        self.CHARUCOBOARD_ROWCOUNT = 7
        self.CHARUCOBOARD_COLCOUNT = 5
        self.ARUCO_DICT = Dictionary_get(aruco.DICT_4X4_1000)
        self.CHARUCO_BOARD = aruco.CharucoBoard_create(
            squaresX=self.CHARUCOBOARD_COLCOUNT,
            squaresY=self.CHARUCOBOARD_ROWCOUNT,
            squareLength=0.04,
            markerLength=0.02,
            dictionary=self.ARUCO_DICT)

    def show_images(self, image_path:str):
        corners_all = []  # Corners discovered in all images processed
        ids_all = []  # Aruco ids corresponding to corners discovered
        image_size = None  # Determined at runtime

        # Images for calibration with the naming scheme
        images = glob.glob(image_path) # image_path='./calib-*.jpg'
        imgs = []
        for iname in images:
            img = cv2.imread(iname) # Open the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale the image
            corners, ids, _ = aruco.detectMarkers(image=gray, dictionary=self.ARUCO_DICT) # Find aruco markers in the query image
            img = aruco.drawDetectedMarkers(image=img, corners=corners) # Outline the aruco markers found in our query image

            # Get charuco corners and ids from detected aruco markers
            response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners=corners,markerIds=ids,
                                                                                    image=gray, board=self.CHARUCO_BOARD)

            # If a Charuco board was found, let's collect image/corner points which requires at least 20 squares
            if response > 20:
                corners_all.append(charuco_corners) # Append these corners calibration array `corners_all`
                ids_all.append(charuco_ids) # Append ids to our calibration array `ids_all`

                # Draw the Charuco board we've detected to show our calibrator the board was properly detected
                img = aruco.drawDetectedCornersCharuco(image=img, charucoCorners=charuco_corners, charucoIds=charuco_ids)
                if not image_size:
                    image_size = gray.shape[::-1] # If our image size is unknown, set it now

                # Reproportion the image, maxing width or height at 1000
                proportion = max(img.shape) / 1000.0
                img = cv2.resize(img, (int(img.shape[1] / proportion), int(img.shape[0] / proportion)))
                # Pause to display each image, waiting for key press
                # cv2.imshow('Charuco board', img)
                imgs.append(imgs)
                # cv2.waitKey(0)
            else:
                print("Not able to detect a charuco board in image: {}".format(iname))
                st.write("Not able to detect a charuco board in image: {}".format(iname))
        cv2.destroyAllWindows()
        if len(images) < 1:
            print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
            st.write("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
            exit()# Exit for failure
        # Make sure we were able to calibrate on at least one charucoboard by checking
        # if we ever determined the image size
        if not image_size:
            # Calibration failed because we didn't see any charucoboards of the PatternSize used
            print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
            st.write("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
            # Exit for failure
            exit()
        return corners_all, ids_all, image_size, imgs

    def calibration(self, corners_all, ids_all, image_size):
        calibration, cameraMatrix, distCoeffs, rot_vecs, trans_vecs = aruco.calibrateCameraCharuco(charucoCorners=corners_all,
                                                                                           charucoIds=ids_all,
                                                                                           board=self.CHARUCO_BOARD,
                                                                                           imageSize=image_size,
                                                                                           cameraMatrix=None,
                                                                                           distCoeffs=None)




        return cameraMatrix, distCoeffs, rot_vecs, trans_vecs

    def save_model(self, cameraMatrix, distCoeffs, rot_vecs, trans_vecs):
        file = open('calibration_final.pckl', 'wb')
        pickle.dump((cameraMatrix, distCoeffs, rot_vecs, trans_vecs), file)
        file.close()
        print(f"Calibration successful. Calibration file used: {'calibration_final.pckl'}")
        st.success(f"Calibration successful. Calibration file used: {'calibration_final.pckl'}")