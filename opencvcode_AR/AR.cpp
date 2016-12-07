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
using namespace std;
using namespace cv;

int main()
{
	/****Caculate the the world coordinates according to two image**************************/
	//read image
	Mat Image1, Image2, ColorImage1, ColorImage2;
	Image1 = imread("Im1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Image2 = imread("Im2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	ColorImage1 = imread("Im1.jpg");
	ColorImage2 = imread("Im2.jpg");
	Mat Intrinsic_Matrix = (Mat_<double>(3, 3) << 5.9275660408893941e+002, 0, 326, 0, 5.9275660408893941e+002,
		2.4450000000000000e+002, 0, 0, 1);
	Mat Distortion_Matrix = (Mat_<double>(5, 1) << -2.1982242153765506e-002, 1.4992239174059787e+000, 0, 0,
		-4.8912910339827480e+000);
	if (Image1.empty()){
		cout << "can not load image" << endl;
		return 0;
	}

	//Sift detect
	SiftFeatureDetector  siftdtc;
	vector<KeyPoint> kp1, kp2;
	siftdtc.detect(Image1, kp1);
	siftdtc.detect(Image2, kp2);
	SiftDescriptorExtractor extractor;
	Mat descriptor1,descriptor2;
	//matches
	BruteForceMatcher<L2<float>> matcher;
	extractor.compute(Image1, kp1, descriptor1);
	extractor.compute(Image2, kp2, descriptor2);
	vector<DMatch> matches;
	matcher.match(descriptor1, descriptor2, matches);
	//keypoints to point2f
	vector<Point2f> points1, points2;
	Point2f pt;
	for (int i = 0; i<kp1.size(); i++)
	{
		pt = kp1[matches[i].queryIdx].pt;
		points1.push_back(pt);

		pt = kp2[matches[i].trainIdx].pt;
		points2.push_back(pt);
	}

	// Compute F matrix using RANSAC
	vector<uchar> inliers(points1.size(), 0);//inlier or outliers
	Mat fundamental_matrix = findFundamentalMat(
		points1, points2, // matching points
		inliers,      // match status (inlier ou outlier)  
		FM_RANSAC, // RANSAC method
		0.5,     // distance to epipolar line
		0.99);  // confidence probability

	// extract the surviving (inliers) matches
	// this code I reference opencv cookbook chap09 matcher.hpp
	vector<uchar>::const_iterator itIn = inliers.begin();
	vector<DMatch>::const_iterator itM = matches.begin();
	vector<DMatch> outMatches;//good correspondense points matches
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
		pt = kp1[outMatches[i].queryIdx].pt;
		points1.push_back(pt);

		pt = kp2[outMatches[i].trainIdx].pt;
		points2.push_back(pt);
	}

	//fundamental_matrix = findFundamentalMat(points12a, points12b, inliers, CV_FM_8POINT);
	cout << "fundamental_matrix :\n" << fundamental_matrix << endl;

	//output matchimage
	Mat img_matches_RANSAC;
	drawMatches(Image1, kp1, Image2, kp2, outMatches, img_matches_RANSAC);
	imshow("12matches points RANSAC", img_matches_RANSAC);

	//Using partial camera calibration to find camera matrix
	Mat Essential_matrix = Intrinsic_Matrix.t() * fundamental_matrix * Intrinsic_Matrix;
	SVD svd(Essential_matrix);
	Matx33d W(0, -1, 0,//HZ 9.13
		1, 0, 0,
		0, 0, 1);
	Mat_<double> M2_R = svd.u * Mat(W) * svd.vt;
	Mat_<double> u3 = svd.u.col(2); //u3
	Matx34d P1(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
	Matx34d P2(M2_R(0, 0), M2_R(0, 1), M2_R(0, 2), u3(0),
		M2_R(1, 0), M2_R(1, 1), M2_R(1, 2), u3(1),
		M2_R(2, 0), M2_R(2, 1), M2_R(2, 2), u3(2));
	Mat M1 = Intrinsic_Matrix * Mat(P1);
	Mat M2 = Intrinsic_Matrix * Mat(P2);

	//get the ckeck board points
	int b_width = 6;
	int b_height = 8;
	Size b_size(b_width, b_height);
	vector<Point2f> corners1, corners2;

	int found1 = findChessboardCorners(Image1, b_size, corners1,
		CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
	int found2 = findChessboardCorners(Image2, b_size, corners2,
		CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

	//triangulate the 2images get the homography points
	Mat Points4D;
	points1.insert(points1.begin(), corners1.begin(), corners1.end());
	points2.insert(points2.begin(), corners2.begin(), corners2.end());
	
	triangulatePoints(M1, M2, points1, points2, Points4D);
	FileStorage file1("Corners4D.xml", FileStorage::WRITE);
	file1 << "Corners4D" << Points4D.t();
	file1.release();

	//convert corners to 3D
	vector<Point3f> Corners3D;
	for (int i = 0; i < 48; i++){
		Point3f pt;
		pt.x = Points4D.at<float>(0, i) / Points4D.at<float>(3, i);
		pt.y = Points4D.at<float>(1, i) / Points4D.at<float>(3, i);
		pt.z = Points4D.at<float>(2, i) / Points4D.at<float>(3, i);
		Corners3D.push_back(pt);
	}

	//use four 4 chessboard corners as cubiod vertices
	vector<Point3f> Vertices;
	Vertices.push_back(Corners3D[0]);
	Vertices.push_back(Corners3D[b_width - 1]);
	Vertices.push_back(Corners3D[b_width*(b_height - 1)]);
	Vertices.push_back(Corners3D[b_width*b_height - 1]);

	//corss product get vector z;
	Point3f vecZ = (Vertices[3] - Vertices[2]).cross(Vertices[0] - Vertices[2]);
	vecZ.x /= 10;
	vecZ.y /= 10;
	vecZ.z /= 10;
	Point3f pt3f = Vertices[0] + vecZ;
	Vertices.push_back(pt3f);
	pt3f = Vertices[1] + vecZ;
	Vertices.push_back(pt3f);
	pt3f = Vertices[2] + vecZ;
	Vertices.push_back(pt3f);
	pt3f = Vertices[3] + vecZ;
	Vertices.push_back(pt3f);
	vector<Point2f> pt2d1,pt2d2;
	Mat M1_R = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	Mat M1_t = (Mat_<double>(3, 1) << 0, 0, 0);
 	projectPoints(Vertices, M1_R, M1_t, Intrinsic_Matrix ,Distortion_Matrix, pt2d1);
	projectPoints(Vertices, M1_R, M1_t, Intrinsic_Matrix, Distortion_Matrix, pt2d2);

	//line the vertices
	int myradius = 2;
	for (int i = 0; i < pt2d1.size(); i++)
		circle(ColorImage1, cvPoint(pt2d1[i].x, pt2d1[i].y), myradius, CV_RGB(0, 255, 0));
	line(ColorImage1, pt2d1[0], pt2d1[1], Scalar(0, 255, 0), 2);
	line(ColorImage1, pt2d1[1], pt2d1[3], Scalar(0, 255, 0), 2);
	line(ColorImage1, pt2d1[0], pt2d1[2], Scalar(0, 255, 0), 2);
	line(ColorImage1, pt2d1[2], pt2d1[3], Scalar(0, 255, 0), 2);
	line(ColorImage1, pt2d1[0], pt2d1[4], Scalar(0, 255, 0), 2);
	line(ColorImage1, pt2d1[1], pt2d1[5], Scalar(0, 255, 0), 2);
	line(ColorImage1, pt2d1[2], pt2d1[6], Scalar(0, 255, 0), 2);
	line(ColorImage1, pt2d1[3], pt2d1[7], Scalar(0, 255, 0), 2);
	line(ColorImage1, pt2d1[4], pt2d1[5], Scalar(0, 255, 0), 2);
	line(ColorImage1, pt2d1[5], pt2d1[7], Scalar(0, 255, 0), 2);
	line(ColorImage1, pt2d1[7], pt2d1[6], Scalar(0, 255, 0), 2);
	line(ColorImage1, pt2d1[4], pt2d1[6], Scalar(0, 255, 0), 2);
	imshow("Insert 3D object Im1",ColorImage1);

	for (int i = 0; i < pt2d2.size(); i++)
		circle(ColorImage2, cvPoint(pt2d1[i].x, pt2d1[i].y), myradius, CV_RGB(0, 255, 0));
	line(ColorImage2, pt2d2[0], pt2d2[1], Scalar(0, 255, 0), 2);
	line(ColorImage2, pt2d2[1], pt2d2[3], Scalar(0, 255, 0), 2);
	line(ColorImage2, pt2d2[0], pt2d2[2], Scalar(0, 255, 0), 2);
	line(ColorImage2, pt2d2[2], pt2d2[3], Scalar(0, 255, 0), 2);
	line(ColorImage2, pt2d2[0], pt2d2[4], Scalar(0, 255, 0), 2);
	line(ColorImage2, pt2d2[1], pt2d2[5], Scalar(0, 255, 0), 2);
	line(ColorImage2, pt2d2[2], pt2d2[6], Scalar(0, 255, 0), 2);
	line(ColorImage2, pt2d2[3], pt2d2[7], Scalar(0, 255, 0), 2);
	line(ColorImage2, pt2d2[4], pt2d2[5], Scalar(0, 255, 0), 2);
	line(ColorImage2, pt2d2[5], pt2d2[7], Scalar(0, 255, 0), 2);
	line(ColorImage2, pt2d2[7], pt2d2[6], Scalar(0, 255, 0), 2);
	line(ColorImage2, pt2d2[4], pt2d2[6], Scalar(0, 255, 0), 2);
	imshow("Insert 3D object Im2", ColorImage2);



	Point2f p1[4], p2[4];
	p1[0] = pt2d1[4];
	p1[1] = pt2d1[5];
	p1[2] = pt2d1[7];
	p1[3] = pt2d1[6];

	p2[0] = pt2d2[4];
	p2[1] = pt2d2[5];
	p2[2] = pt2d2[7];
	p2[3] = pt2d2[6];

	Mat Picture = imread("dota.png");
	Point2f q[4];
	q[0].x = Picture.cols * 0;
	q[0].y = Picture.rows * 0;
	q[1].x = Picture.cols;
	q[1].y = Picture.rows * 0;
	q[2].x = Picture.cols;
	q[2].y = Picture.rows;
	q[3].x = Picture.cols * 0;
	q[3].y = Picture.rows;

	Mat warp_matrix;
	warp_matrix = getPerspectiveTransform(q, p1);
	Mat blank(Picture.size(), CV_8UC3, Scalar(255, 255, 255));

	Mat neg_img, cpy_img;
	//warp the image1
	warpPerspective(Picture, neg_img, warp_matrix, ColorImage1.size(), INTER_LINEAR, BORDER_CONSTANT);
	warpPerspective(blank, cpy_img, warp_matrix, ColorImage1.size(), INTER_LINEAR, BORDER_CONSTANT);
	bitwise_not(cpy_img, cpy_img);
	bitwise_and(cpy_img, ColorImage1, cpy_img);
	bitwise_or(cpy_img, neg_img, ColorImage1);
	imshow("Map Picture 1", ColorImage1);

	warp_matrix = getPerspectiveTransform(q, p2);
	//warp the image2
	warpPerspective(Picture, neg_img, warp_matrix, ColorImage2.size(), INTER_LINEAR, BORDER_CONSTANT);
	warpPerspective(blank, cpy_img, warp_matrix, ColorImage2.size(), INTER_LINEAR, BORDER_CONSTANT);
	bitwise_not(cpy_img, cpy_img);
	bitwise_and(cpy_img, ColorImage2, cpy_img);
	bitwise_or(cpy_img, neg_img, ColorImage2);
	imshow("Map Picture 2", ColorImage2);

	waitKey(0);
	return 0;
}


//#include <vector>
//#include <iostream>
//using namespace std; 
//int main(void)
//{
//	vector<int> a;
//	a.push_back(1);
//	a.push_back(2);
//	a.push_back(8);
//
//	vector<int> b;
//	b.push_back(4);
//	b.push_back(5);
//	b.push_back(6);
//
//	b.insert(b.begin(), a.begin(), a.end());
//
//	for (int i = 0; i<b.size(); i++)
//		cout << b[i];
//	return 0;
//}