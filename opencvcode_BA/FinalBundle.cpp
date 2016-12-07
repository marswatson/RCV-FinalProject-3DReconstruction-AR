#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <vector>
#include <iostream>  
#include <fstream> 
#include <string> 
#include <math.h> 
#include "sba.h"
using namespace std;
using namespace cv;

bool CheckCoherentRotation(Mat&);
vector<int> InterPolation(vector<Point2f> pixelp2f, Mat image);
float distance(float x, float y, float x1, float y1);
void FindMatchesIndex(Mat descriptor, Mat descriptor1, vector<KeyPoint> kp, vector<KeyPoint> kp1,
	vector<DMatch>& matches, vector<DMatch>::const_iterator& itM, vector<DMatch>& outMatches,
	vector<Point2f>& points1, vector<Point2f>& points2);
void Find3Dto2DMatch(vector<int>& ThreeDRecord, vector<int>& TwoDRecord,
	vector<DMatch> outMatches1, vector<DMatch> outMatches2);
void TrackPoints(vector<DMatch> outMatches12, vector<DMatch> outMatches23, vector<DMatch> outMatches34, 
	vector<DMatch> outMatches45, vector<DMatch> outMatches56, vector< vector <int> >&  Record);
void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps);

int main(){
	//read Picture image
	vector<Mat> Images;
	Mat temp;
	temp = imread("Images/Im1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Images.push_back(temp);
	temp = imread("Images/Im2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Images.push_back(temp);
	temp = imread("Images/Im3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Images.push_back(temp);
	temp = imread("Images/Im4.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Images.push_back(temp);
	temp = imread("Images/Im5.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Images.push_back(temp);
	temp = imread("Images/Im6.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Images.push_back(temp);
	Mat Intrinsic_Matrix, Distortion_Matrix;
	Intrinsic_Matrix = (Mat_<double>(3, 3) << 5.9275660408893941e+002, 0, 326, 0, 5.9275660408893941e+002,
		2.4450000000000000e+002, 0, 0, 1); /*296.4665107165597, 0.0, 120.97596801682067,
		0.0, 295.99214838920051, 164.29958079860066,
		0.0, 0.0, 1.0);*/
	Distortion_Matrix = (Mat_<double>(5, 1) << -2.1982242153765506e-002, 1.4992239174059787e+000, 0, 0,
											   -4.8912910339827480e+000); /*0.119660327263156, -0.217737668962057, 0.001858597481326, 0.000404599385077, 0.0);*/

	vector<Mat> ColorImages;
	temp = imread("Images/Im1.jpg");
	ColorImages.push_back(temp);
	temp = imread("Images/Im2.jpg");
	ColorImages.push_back(temp);
	temp = imread("Images/Im3.jpg");
	ColorImages.push_back(temp);
	temp = imread("Images/Im4.jpg");
	ColorImages.push_back(temp);
	temp = imread("Images/Im5.jpg");
	ColorImages.push_back(temp);
	temp = imread("Images/Im6.jpg");
	ColorImages.push_back(temp);
	

	//check if the images have been read
	for (int i = 0; i < Images.size(); i++){
		if (Images[i].empty()){
			cout << "Can't read the images.\n";
		}
	}

	//display image 
	//imshow("First Image", Images[0]);
	//imshow("Second Image", Images[1]);

	//sift detect
	SiftFeatureDetector  siftdtc;
	vector < vector<KeyPoint> >kp(6);

	vector<Mat> Outimg(6);
	for (int i = 0; i < Images.size(); i++){
		siftdtc.detect(Images[i], kp[i]);
	}
	

	//extrace descriptor for img1 and img2
	SiftDescriptorExtractor extractor;
	vector<Mat> descriptor(6);
	BruteForceMatcher<L2<float>> matcher;
	for (int i = 0; i < Images.size(); i++){
		extractor.compute(Images[i], kp[i], descriptor[i]);
	}
	
	vector<DMatch> matches12;
	matcher.match(descriptor[0], descriptor[1], matches12);

	//Transnform "KeyPoint" to "Point2f" in the next step we will use "point2f" compute F
	vector<Point2f> points12a, points12b;
	Point2f pt;
	for (int i = 0; i<kp[0].size(); i++)
	{
		pt = kp[0][matches12[i].queryIdx].pt;
		points12a.push_back(pt);

		pt = kp[1][matches12[i].trainIdx].pt;
		points12b.push_back(pt);
	}

	// Compute F matrix using RANSAC
	vector<uchar> inliers(points12a.size(), 0);//inlier or outliers
	Mat fundamental_matrix = findFundamentalMat(
		points12a, points12b, // matching points
		inliers,      // match status (inlier ou outlier)  
		FM_RANSAC, // RANSAC method
		1,     // distance to epipolar line
		0.99);  // confidence probability

	// extract the surviving (inliers) matches
	// this code I reference opencv cookbook chap09 matcher.hpp
	vector<uchar>::const_iterator itIn = inliers.begin();
	vector<DMatch>::const_iterator itM12 = matches12.begin();
	vector<DMatch> outMatches12;//good correspondense points matches
	// for all matches
	for (; itIn != inliers.end(); ++itIn, ++itM12)
	{
		if (*itIn)
		{
			// it is a valid match
			outMatches12.push_back(*itM12);
		}
	}
	//because there still lots of dismatch Picture point so we compute F again
	points12a.clear();
	points12b.clear();
	for (int i = 0; i<outMatches12.size(); i++)
	{
		pt = kp[0][outMatches12[i].queryIdx].pt;
		points12a.push_back(pt);

		pt = kp[1][outMatches12[i].trainIdx].pt;
		points12b.push_back(pt);
	}

	//fundamental_matrix = findFundamentalMat(points12a, points12b, inliers, CV_FM_8POINT);
	cout << "fundamental_matrix :\n" << fundamental_matrix << endl;

	//output matchimage
	Mat img_matches_RANSAC;
	drawMatches(Images[0], kp[0], Images[1], kp[1], outMatches12, img_matches_RANSAC);
	imshow("12matches points RANSAC", img_matches_RANSAC);

	//Using partial camera calibration to find camera matrix
	Mat Essential_matrix = Intrinsic_Matrix.t() * fundamental_matrix * Intrinsic_Matrix;
	SVD svd(Essential_matrix);
	Matx33d W(0, -1, 0,//HZ 9.13
		1, 0, 0,
		0, 0, 1);
	Mat_<double> M2_R = svd.u * Mat(W) * svd.vt;
	if (!CheckCoherentRotation(M2_R))
		return 0;
	Mat_<double> u3 = svd.u.col(2); //u3
	Mat M2_t = u3;
	Matx34d P1(1,0,0,0,0,1,0,0,0,0,1,0);
	Mat M1_R = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	Mat M1_t = (Mat_<double>(3, 1) <<  0, 0, 0);
	Matx34d P2(M2_R(0, 0), M2_R(0, 1), M2_R(0, 2), u3(0),
		M2_R(1, 0), M2_R(1, 1), M2_R(1, 2), u3(1),
		M2_R(2, 0), M2_R(2, 1), M2_R(2, 2), u3(2));
	Mat M1 = Intrinsic_Matrix * Mat(P1);
	Mat M2 = Intrinsic_Matrix * Mat(P2);
	Mat M1_r, M2_r;
	Rodrigues(M1_R,M1_r);
	Rodrigues(M2_R,M2_r);
	/************************************************************************************************************/
	
	/******************************************************************/
	vector<DMatch> matches23,matches34,matches45,matches56;
	vector<DMatch>::const_iterator itM23, itM34, itM45, itM56;
	vector<DMatch> outMatches23, outMatches34, outMatches45, outMatches56;//good correspondense points matches
	vector<Point2f> points23a, points23b, points34a, points34b, points45a, points45b, points56a, points56b;
	FindMatchesIndex(descriptor[1], descriptor[2], kp[1], kp[2], matches23, itM23, outMatches23, points23a, points23b);
	FindMatchesIndex(descriptor[2], descriptor[3], kp[2], kp[3], matches34, itM34, outMatches34, points34a, points34b);
	FindMatchesIndex(descriptor[3], descriptor[4], kp[3], kp[4], matches45, itM45, outMatches45, points45a, points45b);
	FindMatchesIndex(descriptor[4], descriptor[5], kp[4], kp[5], matches56, itM56, outMatches56, points56a, points56b);
	//draw matches
	drawMatches(Images[1], kp[1], Images[2], kp[2], outMatches23, img_matches_RANSAC);
	imshow("Matches23", img_matches_RANSAC);
	drawMatches(Images[2], kp[2], Images[3], kp[3], outMatches34, img_matches_RANSAC);
	imshow("Matches34", img_matches_RANSAC);
	drawMatches(Images[3], kp[3], Images[4], kp[4], outMatches45, img_matches_RANSAC);
	imshow("Matches45", img_matches_RANSAC);
	drawMatches(Images[4], kp[4], Images[5], kp[5], outMatches56, img_matches_RANSAC);
	imshow("Matches56", img_matches_RANSAC);
	//use vector to record outmaches number
	vector< vector <int> >  Record;
	//Track Points
	TrackPoints(outMatches12, outMatches23, outMatches34, outMatches45, outMatches56, Record);
	//Use Record to get track points
	vector < vector<KeyPoint> > TrackKP(6);
	for (int i = 0; i < Record[0].size(); i++){
		TrackKP[0].push_back(kp[0][ outMatches12[ Record[0][i] ].queryIdx ]);
	}
	for (int i = 0; i < Record[1].size(); i++){
		TrackKP[1].push_back(kp[1][outMatches23[Record[1][i]].queryIdx]);
	}
	for (int i = 0; i < Record[2].size(); i++){
		TrackKP[2].push_back(kp[2][outMatches34[Record[2][i]].queryIdx]);
	}
	for (int i = 0; i < Record[3].size(); i++){
		TrackKP[3].push_back(kp[3][outMatches45[Record[3][i]].queryIdx]);
	}
	for (int i = 0; i < Record[4].size(); i++){
		TrackKP[4].push_back(kp[4][outMatches56[Record[4][i]].queryIdx]);
	}
	for (int i = 0; i < Record[4].size(); i++){
		TrackKP[5].push_back(kp[5][outMatches56[Record[4][i]].trainIdx]);
	}
	//convert keypoints to point2f
	vector < vector<Point2f> > TrackPoint2f(6);
	for (int i = 0; i < 6; i++){
		KeyPointsToPoints(TrackKP[i], TrackPoint2f[i]);
	}

	/************************************************Triangulation********************************************/
	Mat Points4D;
	triangulatePoints(M1,M2,TrackPoint2f[0],TrackPoint2f[1],Points4D);
	Mat M3_r, M3_t, M3_R, M4_r, M4_t, M4_R, M5_r, M5_t, M5_R, M6_r, M6_t, M6_R;
	//get the 3D points;
	vector<Point3f> Points3D;
	for (int i = 0; i < Points4D.cols; i++){
		Point3f pt;
		pt.x = Points4D.at<float>(0, i) / Points4D.at<float>(3, i);
		pt.y = Points4D.at<float>(1, i) / Points4D.at<float>(3, i);
		pt.z = Points4D.at<float>(2, i) / Points4D.at<float>(3, i);
		Points3D.push_back(pt);
	}
	//get the camera pose from track points
	solvePnPRansac(Points3D, TrackPoint2f[2], Intrinsic_Matrix, Distortion_Matrix, M3_r, M3_t);
	solvePnPRansac(Points3D, TrackPoint2f[3], Intrinsic_Matrix, Distortion_Matrix, M4_r, M4_t);
	solvePnPRansac(Points3D, TrackPoint2f[4], Intrinsic_Matrix, Distortion_Matrix, M5_r, M5_t);
	solvePnPRansac(Points3D, TrackPoint2f[5], Intrinsic_Matrix, Distortion_Matrix, M6_r, M6_t);
	Rodrigues(M3_r, M3_R);
	Rodrigues(M4_r, M4_R);
	Rodrigues(M5_r, M5_R);
	Rodrigues(M6_r, M6_R);
	//get camera matrix
	Matx34d P3(M3_R.at<double>(0, 0), M3_R.at<double>(0, 1), M3_R.at<double>(0, 2), M3_t.at<double>(0, 0),
		M3_R.at<double>(1, 0), M3_R.at<double>(1, 1), M3_R.at<double>(1, 2), M3_t.at<double>(1, 0),
		M3_R.at<double>(2, 0), M3_R.at<double>(2, 1), M3_R.at<double>(2, 2), M3_t.at<double>(2, 0));
	Matx34d P4(M4_R.at<double>(0, 0), M4_R.at<double>(0, 1), M4_R.at<double>(0, 2), M4_t.at<double>(0, 0),
		M4_R.at<double>(1, 0), M4_R.at<double>(1, 1), M4_R.at<double>(1, 2), M4_t.at<double>(1, 0),
		M4_R.at<double>(2, 0), M4_R.at<double>(2, 1), M4_R.at<double>(2, 2), M4_t.at<double>(2, 0));
	Matx34d P5(M5_R.at<double>(0, 0), M5_R.at<double>(0, 1), M5_R.at<double>(0, 2), M5_t.at<double>(0, 0),
		M5_R.at<double>(1, 0), M5_R.at<double>(1, 1), M5_R.at<double>(1, 2), M5_t.at<double>(1, 0),
		M5_R.at<double>(2, 0), M5_R.at<double>(2, 1), M5_R.at<double>(2, 2), M5_t.at<double>(2, 0));
	Matx34d P6(M6_R.at<double>(0, 0), M6_R.at<double>(0, 1), M6_R.at<double>(0, 2), M6_t.at<double>(0, 0),
		M6_R.at<double>(1, 0), M6_R.at<double>(1, 1), M6_R.at<double>(1, 2), M6_t.at<double>(1, 0),
		M6_R.at<double>(2, 0), M6_R.at<double>(2, 1), M6_R.at<double>(2, 2), M6_t.at<double>(2, 0));
	Mat M3 = Intrinsic_Matrix * Mat(P3);
	Mat M4 = Intrinsic_Matrix * Mat(P4);
	Mat M5 = Intrinsic_Matrix * Mat(P5);
	Mat M6 = Intrinsic_Matrix * Mat(P6);

	vector<Mat> M;
	M.push_back(M1);
	M.push_back(M2);
	M.push_back(M3);
	M.push_back(M4);
	M.push_back(M5);
	M.push_back(M6);

	vector<Mat> M_r;
	M_r.push_back(M1_r);
	M_r.push_back(M2_r);
	M_r.push_back(M3_r);
	M_r.push_back(M4_r);
	M_r.push_back(M5_r);
	M_r.push_back(M6_r);

	vector<Mat> M_t;
	M_t.push_back(M1_t);
	M_t.push_back(M2_t);
	M_t.push_back(M3_t);
	M_t.push_back(M4_t);
	M_t.push_back(M5_t);
	M_t.push_back(M6_t);

	//Reproject error
	vector<Point2f> imgpts2;
	vector<float> ReprojectError(4,0);
	projectPoints(Points3D, M3_r, M3_t, Intrinsic_Matrix, Distortion_Matrix, imgpts2);
	int myradius = 2;
	for (int i = 0; i < imgpts2.size(); i++)
		circle(ColorImages[2], cvPoint(imgpts2[i].x, imgpts2[i].y), myradius, CV_RGB(100, 0, 0), -1, 8, 0);
	for (int i = 0; i < TrackPoint2f[2].size(); i++)
		circle(ColorImages[2], cvPoint(TrackPoint2f[2][i].x, TrackPoint2f[2][i].y), myradius, CV_RGB(0, 255, 0), -1, 8, 0);
	imshow("reprojected points3", ColorImages[2]);
	for (int i = 0; i < imgpts2.size(); i++){
		ReprojectError[0] += sqrt((imgpts2[i].x - TrackPoint2f[2][i].x)*(imgpts2[i].x - TrackPoint2f[2][i].x) +
			(imgpts2[i].y - TrackPoint2f[2][i].y)*(imgpts2[i].y - TrackPoint2f[2][i].y));
	}
	

	imgpts2.clear();
	projectPoints(Points3D, M4_r, M4_t, Intrinsic_Matrix, Distortion_Matrix, imgpts2);
	for (int i = 0; i < imgpts2.size(); i++)
		circle(ColorImages[3], cvPoint(imgpts2[i].x, imgpts2[i].y), myradius, CV_RGB(100, 0, 0), -1, 8, 0);
	for (int i = 0; i < TrackPoint2f[3].size(); i++)
		circle(ColorImages[3], cvPoint(TrackPoint2f[3][i].x, TrackPoint2f[3][i].y), myradius, CV_RGB(0, 255, 0), -1, 8, 0);
	imshow("reprojected points4", ColorImages[3]);
	for (int i = 0; i < imgpts2.size(); i++){
		ReprojectError[1] += sqrt((imgpts2[i].x - TrackPoint2f[3][i].x)*(imgpts2[i].x - TrackPoint2f[3][i].x) +
			(imgpts2[i].y - TrackPoint2f[3][i].y)*(imgpts2[i].y - TrackPoint2f[3][i].y));
	}

	imgpts2.clear();
	projectPoints(Points3D, M5_r, M5_t, Intrinsic_Matrix, Distortion_Matrix, imgpts2);
	for (int i = 0; i < imgpts2.size(); i++)
		circle(ColorImages[4], cvPoint(imgpts2[i].x, imgpts2[i].y), myradius, CV_RGB(100, 0, 0), -1, 8, 0);
	for (int i = 0; i < TrackPoint2f[4].size(); i++)
		circle(ColorImages[4], cvPoint(TrackPoint2f[4][i].x, TrackPoint2f[4][i].y), myradius, CV_RGB(0, 255, 0), -1, 8, 0);
	imshow("reprojected points5", ColorImages[4]);
	for (int i = 0; i < imgpts2.size(); i++){
		ReprojectError[2] += sqrt((imgpts2[i].x - TrackPoint2f[4][i].x)*(imgpts2[i].x - TrackPoint2f[4][i].x) +
			(imgpts2[i].y - TrackPoint2f[4][i].y)*(imgpts2[i].y - TrackPoint2f[4][i].y));
	}

	imgpts2.clear();
	projectPoints(Points3D, M6_r, M6_t, Intrinsic_Matrix, Distortion_Matrix, imgpts2);
	for (int i = 0; i < imgpts2.size(); i++)
		circle(ColorImages[5], cvPoint(imgpts2[i].x, imgpts2[i].y), myradius, CV_RGB(100, 0, 0), -1, 8, 0);
	for (int i = 0; i < TrackPoint2f[5].size(); i++)
		circle(ColorImages[5], cvPoint(TrackPoint2f[5][i].x, TrackPoint2f[5][i].y), myradius, CV_RGB(0, 255, 0), -1, 8, 0);
	imshow("reprojected points6", ColorImages[5]);
	for (int i = 0; i < imgpts2.size(); i++){
		ReprojectError[3] += sqrt((imgpts2[i].x - TrackPoint2f[5][i].x)*(imgpts2[i].x - TrackPoint2f[5][i].x) +
			(imgpts2[i].y - TrackPoint2f[5][i].y)*(imgpts2[i].y - TrackPoint2f[5][i].y));
	}

	for (int i = 3; i < 7; i++){
		cout << "The ReprojectError for image " << i << " is:" << ReprojectError[i - 3] << endl;
	}

	FileStorage file1("points4D.xml", FileStorage::WRITE);
	file1 << "points4D" << Points4D.t();
	file1.release();

	FileStorage file2("Tranlation.xml", FileStorage::WRITE);
	for (int i = 0; i < M_t.size(); i++){
		file2 << "t" << M_t[i];
	}
	file2.release();

	FileStorage file3("Rotation.xml", FileStorage::WRITE);
	for (int i = 0; i < M_r.size(); i++){
		file3 << "r" << M_r[i];
	}
	file3.release();

	vector <Mat> HPoints(6);
	for (int i = 0; i < TrackPoint2f.size(); i++){	
		convertPointsToHomogeneous(TrackPoint2f[i], HPoints[i]);
	}
	FileStorage file4("ImagePoints.txt", FileStorage::WRITE);
	for (int i = 0; i < HPoints.size(); i++){
		file4 << "ImagePoints" << HPoints[i];
	}
	file4.release();

	FileStorage file5("CameraMatrixBefore.txt", FileStorage::WRITE);
	file5 << "CameraMatrix_1" << M1;
	file5 << "CameraMatrix_1_R" << M1_r;
	file5 << "CameraMatrix_1_t" << M1_t;
	file5 << "CameraMatrix_2" << M2;
	file5 << "CameraMatrix_2_R" << M2_r;
	file5 << "CameraMatrix_2_t" << M2_t;
	file5 << "CameraMatrix_3" << M3;
	file5 << "CameraMatrix_3_R" << M3_r;
	file5 << "CameraMatrix_3_t" << M3_t;
	file5 << "CameraMatrix_4" << M4;
	file5 << "CameraMatrix_4_R" << M4_r;
	file5 << "CameraMatrix_4_t" << M4_t;
	file5 << "CameraMatrix_5" << M5;
	file5 << "CameraMatrix_5_R" << M5_r;
	file5 << "CameraMatrix_5_t" << M5_t;
	file5 << "CameraMatrix_6" << M6;
	file5 << "CameraMatrix_6_R" << M6_r;
	file5 << "CameraMatrix_6_t" << M6_t;
	file5.release();


	waitKey(0);
}

//check if it is rotation matrix	
bool CheckCoherentRotation(Mat& R) {
	if (fabsf(determinant(R)) - 1.0 > 1e-07) {
		cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
		return false;
	}
	return true;
}
//find matched index between 2 image
void FindMatchesIndex(Mat descriptor, Mat descriptor1, vector<KeyPoint> kp, vector<KeyPoint> kp1, 
	vector<DMatch>& matches, vector<DMatch>::const_iterator& itM, vector<DMatch>& outMatches,
	vector<Point2f>& points1, vector<Point2f>& points2){
	BruteForceMatcher<L2<float>> matcher;
	matcher.match(descriptor, descriptor1, matches);

	//Transnform "KeyPoint" to "Point2f" in the next step we will use "point2f" compute F
	Point2f pt;
	points1.clear();
	points2.clear();
	for (int i = 0; i<kp.size(); i++)
	{
		pt = kp[matches[i].queryIdx].pt;
		points1.push_back(pt);

		pt = kp1[matches[i].trainIdx].pt;
		points2.push_back(pt);
	}

	// Compute F matrix using RANSAC
	vector<uchar> inliers(points1.size(), 0);//inlier or outliers
	Mat fundamental_matrix = findFundamentalMat(
		points1, points2, // matching points
		inliers,      // match status (inlier ou outlier)  
		FM_RANSAC, // RANSAC method
		1,     // distance to epipolar line
		0.99);  // confidence probability

	// extract the surviving (inliers) matches
	// this code I reference opencv cookbook chap09 matcher.hpp
	vector<uchar>::const_iterator itIn = inliers.begin();
	itM = matches.begin();
	//good correspondense points matches
	// for all matches
	for (; itIn != inliers.end(); ++itIn, ++itM)
	{
		if (*itIn)
		{
			// it is a valid match
			outMatches.push_back(*itM);
		}
	}
	//because there still lots of dismatch Picture point so we compute F again
	points1.clear();
	points2.clear();
	for (int i = 0; i<outMatches.size(); i++)
	{
		pt = kp[outMatches[i].queryIdx].pt;
		points1.push_back(pt);

		pt = kp1[outMatches[i].trainIdx].pt;
		points2.push_back(pt);
	}
}
//get the record index between 3d points and 2d points
void Find3Dto2DMatch(vector<int>& ThreeDRecord, vector<int>& TwoDRecord, 
	vector<DMatch> outMatches1,vector<DMatch> outMatches2){
	for (int i = 0; i < outMatches1.size(); i++){
		for (int j = 0; j < outMatches2.size(); j++){
			if (outMatches1[i].trainIdx == outMatches2[j].queryIdx){
				ThreeDRecord.push_back(i);
				TwoDRecord.push_back(j);
			}
		}
	}
}
//keypoints to points2f
void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) {
	ps.clear();
	for (unsigned int i = 0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}
//points2f to keypoints
void PointsToKeyPoints(const vector<Point2f>& ps, vector<KeyPoint>& kps) {
	kps.clear();
	for (unsigned int i = 0; i<ps.size(); i++) kps.push_back(KeyPoint(ps[i], 1.0f));
}
//Track Point
void TrackPoints(vector<DMatch> outMatches12, vector<DMatch> outMatches23, vector<DMatch> outMatches34, vector<DMatch> outMatches45, vector<DMatch> outMatches56, vector< vector <int> >&  Record){
	int count;
	int flag;
	int flag2;
	vector<int> Record1, Record2, Record3, Record4, Record5;
	for (int i = 0; i < outMatches12.size(); i++){
		for (int j = 0; j < outMatches23.size(); j++){
			if (outMatches12[i].trainIdx == outMatches23[j].queryIdx){
				Record1.push_back(i);
				Record2.push_back(j);
			}
		}
	}
	count = Record2.size();
	flag = 0;
	for (int i = 0; i < count; i++){
		flag2 = 0;
		for (int j = 0; j < outMatches34.size(); j++){
			if (outMatches23[Record2[i - flag]].trainIdx == outMatches34[j].queryIdx){
				Record3.push_back(j);
				flag2 = 1;
				break;
			}
		}
		if(flag2 == 0){
			Record2.erase(Record2.begin() + (i - flag));
			Record1.erase(Record1.begin() + (i - flag));
			flag++;
		}
	}
	count = Record3.size();
	flag = 0;
	for (int i = 0; i < count; i++){
		flag2 = 0;
		for (int j = 0; j < outMatches45.size(); j++){
			if (outMatches34[Record3[i - flag]].trainIdx == outMatches45[j].queryIdx){
				Record4.push_back(j);
				flag2 = 1;
				break;
			}
		}
		if (flag2 == 0){
			Record3.erase(Record3.begin() + (i - flag));
			Record2.erase(Record2.begin() + (i - flag));
			Record1.erase(Record1.begin() + (i - flag));
			flag++;
		}
	}
	count = Record4.size();
	flag = 0;
	for (int i = 0; i < count; i++){
		flag2 = 0;
		for (int j = 0; j < outMatches56.size(); j++){
			if (outMatches45[Record4[i - flag]].trainIdx == outMatches56[j].queryIdx){
				Record5.push_back(j);
				flag2 = 1;
				break;
			}
		}
		if (flag2 == 0){
			Record4.erase(Record4.begin() + (i - flag));
			Record3.erase(Record3.begin() + (i - flag));
			Record2.erase(Record2.begin() + (i - flag));
			Record1.erase(Record1.begin() + (i - flag));
			flag++;
		}
	}
	Record.push_back(Record1);
	Record.push_back(Record2);
	Record.push_back(Record3);
	Record.push_back(Record4);
	Record.push_back(Record5);
}