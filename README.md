# 3D Reconstruction
*	Used calibrated camera to take 6 images from an object then reconstructed it by OpenCV.

*	Programed to track the shared interesting points among 6 images using SIFT algorithm.

*	Estimated the camera position by interesting points and triangulated it to get 3D information. Used bundle adjustment in Matlab VLFeat to refine it and got the sparse 3D reconstruction.

# AR
*	Insert a wireframe “cubic” into image on the chessboard by using OpenCV with C++

*	Map an image to one of a surface of wireframe “cubic” on the chessboard
