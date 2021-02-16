import cv2
import cv2.aruco as aruco
import os
import pickle
import streamlit as st

class Pose_Estimation:
    def __init__(self):
        # self.pickle_path = file_path  #'./calibration.pckl'
        # Constant parameters used in Aruco methods
        self.ARUCO_PARAMETERS = aruco.DetectorParameters_create()
        self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_1000)


    def load_check_pickle(self, pickle_path):
        if not os.path.exists(pickle_path):  #'./calibration.pckl'
            print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
            st.warning("You need to calibrate the camera you'll be using. See calibration project directory for details.")
            exit()
        else:
            f = open('calibration.pckl', 'rb')  # 'calibration.pckl', 'rb'--encoding="utf8"
            (cameraMatrix, distCoeffs, _, _) = pickle.load(f)
            f.close()
            if cameraMatrix is None or distCoeffs is None:
                print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
                st.warning("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
                exit()

            return cameraMatrix, distCoeffs

    def all_pose_estimation(self, cameraMatrix, distCoeffs):
        self.CHARUCOBOARD_ROWCOUNT = 7
        self.CHARUCOBOARD_COLCOUNT = 5

        # Create grid board object we're using in our stream
        self.CHARUCO_BOARD = aruco.CharucoBoard_create(squaresX=self.CHARUCOBOARD_COLCOUNT,
                                                       squaresY=self.CHARUCOBOARD_ROWCOUNT,
                                                       squareLength=0.04, markerLength=0.02, dictionary=self.ARUCO_DICT)
        rot_vecs, trans_vecs = None, None
        cam = cv2.VideoCapture(0) # cam = cv2.VideoCapture('video6.mp4')

        while (cam.isOpened()):
            ret, QueryImg = cam.read() # Capturing each frame of our video stream
            if ret == True:
                gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY) # grayscale image
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.ARUCO_DICT, parameters=self.ARUCO_PARAMETERS) # Detect Aruco markers

                # Refine detected markers
                # Eliminates markers not part of our board, adds missing markers to the board
                corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(image=gray,
                                                                                            board=self.CHARUCO_BOARD,
                                                                                            detectedCorners=corners,
                                                                                            detectedIds=ids,
                                                                                            rejectedCorners=rejectedImgPoints,
                                                                                            cameraMatrix=cameraMatrix,
                                                                                            distCoeffs=distCoeffs)

                QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 255, 0)) # Outline all of the markers detected in our image

                if ids is not None and len(ids) > 10: # Only try to find CharucoBoard if we found markers

                    # Get charuco corners and ids from detected aruco markers
                    response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners=corners,
                                                                                             markerIds=ids,
                                                                                             image=gray,
                                                                                             board=self.CHARUCO_BOARD)

                    if response is not None and response > 20: # Require more than 20 squares
                        # Estimate the posture of the charuco board, which is a construction of 3D space based on the 2D video
                        pose, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners=charuco_corners,
                                                                          charucoIds=charuco_ids,
                                                                          board=self.CHARUCO_BOARD,
                                                                          cameraMatrix=cameraMatrix,
                                                                          distCoeffs=distCoeffs, rvec=None, tvec=None)
                        if pose:
                            # Draw the camera posture calculated from the gridboard
                            QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 0.3)

                cv2.imshow('QueryImage', QueryImg) # Display our image
            if cv2.waitKey(1) & 0xFF == ord('q'): # Exit at the end of the video on the 'q' keypress
                break

        cv2.destroyAllWindows()

    def single_pose_estimation(self, cameraMatrix, distCoeffs):
        self.CHARUCOBOARD_ROWCOUNT = 2
        self.CHARUCOBOARD_COLCOUNT = 2

        # Create grid board object we're using in our stream
        self.CHARUCO_BOARD = aruco.GridBoard_create(markersX=self.CHARUCOBOARD_COLCOUNT, markersY=self.CHARUCOBOARD_ROWCOUNT,
                                                    markerLength=0.09,
                                                    markerSeparation=0.01,
                                                    dictionary=self.ARUCO_DICT)
        rot_vecs, trans_vecs = None, None
        cam = cv2.VideoCapture(0) # cam = cv2.VideoCapture('video6.mp4')

        while (cam.isOpened()):
            ret, QueryImg = cam.read() # Capturing each frame of our video/real-time feed
            if ret == True:
                gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY) # grayscale image
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.ARUCO_DICT, parameters=self.ARUCO_PARAMETERS) # Detect Aruco markers

                # Refine detected markers
                # Eliminates markers not part of our board, adds missing markers to the board
                corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(image=gray,
                                                                                            board=self.CHARUCO_BOARD,
                                                                                            detectedCorners=corners,
                                                                                            detectedIds=ids,
                                                                                            rejectedCorners=rejectedImgPoints,
                                                                                            cameraMatrix=cameraMatrix,
                                                                                            distCoeffs=distCoeffs)

                # Outline all of the markers detected in our image
                QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

                # Estimate the posture per each Aruco marker
                if ids is not None and len(ids) >= 1:
                    # Estimate the posture of the gridboard, which is a construction of 3D space based on the 2D video
                    pose, rvec, tvec = aruco.estimatePoseBoard(corners, ids, self.CHARUCO_BOARD, cameraMatrix, distCoeffs, None, None)
                    if pose:
                        #    # Draw the camera posture calculated from the gridboard
                        QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 0.3)
                    # rvecs, tvecs = aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)
                    # for rvec, tvec in zip(rvecs, tvecs):
                    # QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 1)
                cv2.imshow('QueryImage', QueryImg) # Display our image

            if cv2.waitKey(1) & 0xFF == ord('q'): # Exit at the end of the video on the 'q' keypress
                break

        cv2.destroyAllWindows()