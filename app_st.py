import streamlit as st
from one_calibrate_camera import Camera_Calibration
from two_board_pose_estimation import Pose_Estimation

def main():
    # st.sidebar(st.header("ARUco Detection"))
    st.header("Research Project")
    st.subheader("Topic: Image Aquisition and Processing with the Camera for Object Tracking")

    st.write("Please hit the CALIBRATION button to Calibrate the Camera")
    if st.button("CALIBRATE Camera"):
        cam_calibrate = Camera_Calibration()
        corners, ids, img_size = cam_calibrate.show_images(image_path='./calib-*.jpg')
        st.success("Charuco Board Detected Successfully")
        cameraMatrix, distCoeffs, rot_vecs, trans_vecs = cam_calibrate.calibration(corners_all=corners, ids_all=ids, image_size=img_size)

        cam_calibrate.save_model(cameraMatrix, distCoeffs, rot_vecs, trans_vecs)
        st.success("All the Matrices Saved to pickle file")


    pose_est = Pose_Estimation()
    cameraMatrix, distCoeffs = pose_est.load_check_pickle(pickle_path='./calibration_final.pckl')
    if st.button("All Pose Estimation"):
        pose_est.all_pose_estimation(cameraMatrix, distCoeffs)
    if st.button("Single Pose Estimation"):
        pose_est.single_pose_estimation(cameraMatrix, distCoeffs)

if __name__ == '__main__':
    main()
