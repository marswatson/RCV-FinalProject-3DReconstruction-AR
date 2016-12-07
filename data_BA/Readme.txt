This 5 file are the output of Bundle Adjustment matlab code
K_ - calibration parameters (4xm)		Te_- translation (3xm)
w_ - rotation (3xm)	Xe_- 3d points (4xn) in homogeneous form
error_ - errors for iterations  (1x#iter)


CameraMatrixBefore.txt is the outcome of the openCV. It store the camera matrix 
information before bundle adjustment.

ImagePoints.txt is the outcome of openCV, it store the keypoints 2Dcoordinates 
in every images.

Rotation.xml and Translation.xml is also the output of the openCV. It store
the 6 camera rotation translation vector.