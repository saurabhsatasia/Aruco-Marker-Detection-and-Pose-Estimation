import streamlit as st
from one_calibrate_camera import Camera_Calibration
from two_board_pose_estimation import Pose_Estimation

def main():
    imgs = None
    st.sidebar.image('./220px-Poznań_University_of_Technology.png', width=275)
    st.sidebar.header("Automatic Control and Robotics")
    # st.sidebar.header("ARUco Detection for Object Tracking")

    st.sidebar.markdown("<h3>Project Supervisor</h3>",  unsafe_allow_html=True)
    st.sidebar.markdown("<h4>Dr. Eng. Marcin Kielczewski<br><br><br></h4>",  unsafe_allow_html=True)

    st.sidebar.markdown('<h3>Project Authors<br></h3>', unsafe_allow_html=True)
    st.sidebar.markdown('<h4>Saurabh Satasia</h4>', unsafe_allow_html=True)
    st.sidebar.markdown('<h4>Manohar Shamanna</h4>', unsafe_allow_html=True)
    st.sidebar.markdown('<h4>Nimeshsinh Desai</h4>', unsafe_allow_html=True)


    st.header("Research Project : Design of Control System")
    st.subheader("Topic: Image Aquisition and Processing with the Camera for Object Tracking")

    st.write("Please hit the CALIBRATION button to Calibrate the Camera.")
    if st.button("CALIBRATE Camera"):
        cam_calibrate = Camera_Calibration()
        corners, ids, img_size, imgs = cam_calibrate.show_images(image_path='./calib-*.jpg')
        st.success("Charuco Board Detected Successfully")
        cameraMatrix, distCoeffs, rot_vecs, trans_vecs = cam_calibrate.calibration(corners_all=corners, ids_all=ids, image_size=img_size)

        cam_calibrate.save_model(cameraMatrix, distCoeffs, rot_vecs, trans_vecs)
        st.success("All the Matrices Saved to pickle file")
        # for img in imgs:
        #     st.image(img)
        print(imgs)
        # calibration results
        print("\nCAM-Matrix = \n", cameraMatrix)
        st.subheader("CAM-Matrix =")
        st.write(cameraMatrix)

        print("\nDistortion coefficients = \n", distCoeffs)
        st.subheader("Distortion coefficients =")
        # st.success(distCoeffs)
        st.write(distCoeffs)

        print("\nRotation = \n", rot_vecs)
        st.subheader("Rotation Vectors =")
        col1, col2 = st.beta_columns([1, 1])
        col3, col4 = st.beta_columns([1, 1])
        col5, col6 = st.beta_columns([1, 1])
        col7, col8 = st.beta_columns([1, 1])
        col9, col10 = st.beta_columns([1, 1])

        col1.write(rot_vecs[0])
        col2.write(rot_vecs[1])
        col3.write(rot_vecs[2])
        col4.write(rot_vecs[3])
        col5.write(rot_vecs[4])
        col6.write(rot_vecs[5])
        col7.write(rot_vecs[6])
        col8.write(rot_vecs[7])
        col9.write(rot_vecs[8])
        col10.write(rot_vecs[9])

        print("\nTranslation = \n", trans_vecs)
        st.subheader("Translation Vectors =")
        col1, col2 = st.beta_columns([1, 1])
        col3, col4 = st.beta_columns([1, 1])
        col5, col6 = st.beta_columns([1, 1])
        col7, col8 = st.beta_columns([1, 1])
        col9, col10 = st.beta_columns([1, 1])

        col1.write(trans_vecs[0])
        col2.write(trans_vecs[1])
        col3.write(trans_vecs[2])
        col4.write(trans_vecs[3])
        col5.write(trans_vecs[4])
        col6.write(trans_vecs[5])
        col7.write(trans_vecs[6])
        col8.write(trans_vecs[7])
        col9.write(trans_vecs[8])
        col10.write(trans_vecs[9])

        # arr = []
        # for lst in trans_vecs:
        #     # arr.append(lst)
        #     st.write(lst)

    st.write("Please press the Pose Estimation button to Perform a respective Pose Estimation.")
    pose_est = Pose_Estimation()
    cameraMatrix, distCoeffs = pose_est.load_check_pickle(pickle_path='./calibration_final.pckl')
    if st.button("All Pose Estimation"):
        pose_est.all_pose_estimation(cameraMatrix, distCoeffs)
    if st.button("Single Pose Estimation"):
        pose_est.single_pose_estimation(cameraMatrix, distCoeffs)

if __name__ == '__main__':
    main()
