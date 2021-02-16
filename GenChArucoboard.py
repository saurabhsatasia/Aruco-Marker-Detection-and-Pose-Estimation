import cv2
import cv2.aruco as aruco

# the following call gets a ChArUco board of tiles 5 wide X 7 tall
gridboard = aruco.CharucoBoard_create(
        squaresX=5, 
        squaresY=7, 
        squareLength=0.04, 
        markerLength=0.02, 
        dictionary=aruco.Dictionary_get(aruco.DICT_4X4_1000))

# Create an image from the gridboard
img = gridboard.draw(outSize=(988, 1400))
cv2.imwrite("test_charuco_4X4.jpg", img)

# Display the image to us
cv2.imshow('Gridboard_4X4', img)
# Exit on any key
cv2.waitKey(0)
cv2.destroyAllWindows()

