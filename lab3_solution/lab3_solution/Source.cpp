/*	CS585_Lab3
*	CS585 Image and Video Computing
*	Lab 3
*	--------------
*	This program introduces the following concepts:
*		a) Reading a stream of images from a webcamera, and displaying the video
*		b) Skin color detection
*		c) Background differencing
*		d) Visualizing motion history
*	--------------
*/

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//C++ standard libraries
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <list>
#include <map>
#include <stack>
#include <time.h>

using namespace cv;
using namespace std;

//function declarations

/**
Function that returns the maximum of 3 integers
@param a first integer
@param b second integer
@param c third integer
*/
int myMax(int a, int b, int c);

/**
Function that returns the minimum of 3 integers
@param a first integer
@param b second integer
@param c third integer
*/
int myMin(int a, int b, int c);

/**
Function that detects whether a pixel belongs to the skin based on RGB values
@param src The source color image
@param dst The destination grayscale image where skin pixels are colored white and the rest are colored black
*/
void mySkinDetect(Mat& src, Mat& dst);

/**
Function that does frame differencing between the current frame and the previous frame
@param src The current color image
@param prev The previous color image
@param dst The destination grayscale image where pixels are colored white if the corresponding pixel intensities in the current
and previous image are not the same
*/
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst);

/**
Function that accumulates the frame differences for a certain number of pairs of frames
@param mh Vector of frame difference images
@param dst The destination grayscale image to store the accumulation of the frame difference images
*/
void myMotionEnergy(vector<Mat> mh, Mat& dst);
/**

*/
Mat getTemplate(Mat origin);
void LabelColor(const cv::Mat& labelImg, cv::Mat& colorLabelImg);
cv::Mat_<uchar> ocmu_maxconnecteddomain(cv::Mat_<uchar> binImg);
double getThre(String file_path);
bool getGesture(Mat origin, Mat templa);
int getTime();
int result[2] = { 0, 0 }; 
double threshold_gesture = 0;

int main()
{

	//----------------
	//a) Reading a stream of images from a webcamera, and displaying the video
	//----------------
	// For more information on reading and writing video: http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html
	// open the video camera no. 0
	VideoCapture cap(0);

	// if not successful, exit program
	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	//create a window called "MyVideoFrame0"
	namedWindow("MyVideo0", WINDOW_AUTOSIZE);
	Mat frame0;

	// read a new frame from video
	bool bSuccess0 = cap.read(frame0);

	//if not successful, break loop
	if (!bSuccess0)
	{
		cout << "Cannot read a frame from video stream" << endl;
	}

	//show the frame in "MyVideo" window
	imshow("MyVideo0", frame0);

	//create a window called "MyVideo"
	namedWindow("MyVideo", WINDOW_AUTOSIZE);
	namedWindow("MyVideoFI", WINDOW_AUTOSIZE);
	namedWindow("Skin", WINDOW_AUTOSIZE);

	vector<Mat> myMotionHistory;
	Mat fMH1, fMH2, fMH3;
	fMH1 = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);
	fMH2 = fMH1.clone();
	fMH3 = fMH1.clone();
	myMotionHistory.push_back(fMH1);
	myMotionHistory.push_back(fMH2);
	myMotionHistory.push_back(fMH3);

	////template detecting
	String file_path = "PA2_rockinroll_2.jpg";
	Mat origin = imread(file_path, IMREAD_COLOR);
	Mat templa = getTemplate(origin);
	resize(templa, templa, Size(), 0.4, 0.4);
	imshow("resize", templa);
	int lastTime = 0;
	bool detec = false;
	threshold_gesture = getThre(file_path);
	//waitKey(0);
	while (1)
	{
		// read a new frame from video
		Mat frame;
		bool bSuccess = cap.read(frame);
		medianBlur(frame, frame, 7);
		Mat frameDest;
		/*imshow("MyVideoFI", frameDest);*/
		//if not successful, break loop
		if (!bSuccess)
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		//show the frame in "MyVideo" window
		//imshow("MyVideo0", frame0);

		// destination frame
		frameDest = Mat::zeros(frame.rows, frame.cols, CV_8UC1); //Returns a zero array of same size as src mat, and of type CV_8UC1
		//----------------
		//	b) Skin color detection
		//----------------
		mySkinDetect(frame, frameDest);
		Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
		morphologyEx(frameDest, frameDest, MORPH_OPEN, element);
		int now = getTime();
		if (now - lastTime > 2)
		{
			detec = getGesture(frameDest, templa);
			lastTime = now;
		}
		if (detec) {
			Rect rect(result[1], result[0], templa.cols, templa.rows);
			cv::rectangle(frame0, rect, Scalar(255, 0, 0), 1, LINE_8, 0);
		}
		imshow("MyVideo0", frame0);
		imshow("Skin", frameDest);

		//----------------
		//	c) Background differencing
		//----------------


		//call myFrameDifferencing function
		/*myFrameDifferencing(frame0, frame, frameDest);
		imshow("MyVideo", frameDest);


		myMotionHistory.erase(myMotionHistory.begin());
		myMotionHistory.push_back(frameDest);
		Mat myMH = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);*/

		//----------------
		//  d) Visualizing motion history
		//----------------

		//call myMotionEnergy function
		//myMotionEnergy(myMotionHistory, myMH);


		//imshow("MyVideoMH", myMH); //show the frame in "MyVideo" window
		frame0 = frame;
		//wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		if (waitKey(30) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}

	 }
	cap.release();
	return 0;
}

//Function that returns the maximum of 3 integers
int myMax(int a, int b, int c) {
	int m = a;
	(void)((m < b) && (m = b)); //short-circuit evaluation
	(void)((m < c) && (m = c));
	return m;
}

//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
	int m = a;
	(void)((m > b) && (m = b));
	(void)((m > c) && (m = c));
	return m;
}

//Function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst) {
	//Surveys of skin color modeling and detection techniques:
	//Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
	//Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			//For each pixel, compute the average intensity of the 3 color channels
			Vec3b intensity = src.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			if ((R > 95 && G > 40 && B > 20) && (myMax(R, G, B) - myMin(R, G, B) > 15) && (abs(R - G) > 15) && (R > G) && (R > B)) {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
}

//Function that does frame differencing between the current frame and the previous frame
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst) {
	//For more information on operation with arrays: http://docs.opencv.org/modules/core/doc/operations_on_arrays.html
	//For more information on how to use background subtraction methods: http://docs.opencv.org/trunk/doc/tutorials/video/background_subtraction/background_subtraction.html
	absdiff(prev, curr, dst);
	Mat gs = dst.clone();
	cvtColor(dst, gs, CV_BGR2GRAY);
	dst = gs > 50;
}

//Function that accumulates the frame differences for a certain number of pairs of frames
void myMotionEnergy(vector<Mat> mh, Mat& dst) {
	Mat mh0 = mh[0];
	Mat mh1 = mh[1];
	Mat mh2 = mh[2];

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			if (mh0.at<uchar>(i, j) == 255 || mh1.at<uchar>(i, j) == 255 || mh2.at<uchar>(i, j) == 255) {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
}

//Function that draw the template of certain gesture
Mat getTemplate(Mat origin) {
	// destination frame
	Mat skin = Mat::zeros(origin.rows, origin.cols, CV_8UC1);
	for (int i = 0; i < origin.rows; i++) {
		for (int j = 0; j < origin.cols; j++) {
			//For each pixel, compute the average intensity of the 3 color channels
			Vec3b intensity = origin.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			skin.at<uchar>(i, j) = R * 0.299 + G * 0.587 + B * 0.114;
		}
	}
	/*namedWindow("window1", 0);
	imshow("window1", skin);*/
	Mat bina;
	threshold(skin, bina, 110, 255, CV_THRESH_BINARY);
	namedWindow("window2", 0);
	imshow("window2", bina);
	// find the biggest connected field
	Mat label = Mat::zeros(bina.rows, bina.cols, CV_8UC1);
	label = ocmu_maxconnecteddomain(bina);
	/*cv::Mat colorLabelImg;
	LabelColor(label, colorLabelImg);*/
	namedWindow("window3", 0);
	imshow("window3", label);
	return label;
}


cv::Scalar GetRandomColor()
{
	uchar r = 255 * (rand() / (1.0 + RAND_MAX));
	uchar g = 255 * (rand() / (1.0 + RAND_MAX));
	uchar b = 255 * (rand() / (1.0 + RAND_MAX));
	return cv::Scalar(b, g, r);
}

double getThre(String file_path) {
	if (file_path == "PA2_thunmbsup_2.jpg") {
		return 0.2;
	}
	if (file_path == "PA2_yeah_2.jpg") {
		return 0.2;
	}
	return 0.2;
}

void LabelColor(const cv::Mat& labelImg, cv::Mat& colorLabelImg)
{
	if (labelImg.empty() ||
		labelImg.type() != CV_32SC1)
	{
		return;
	}

	std::map<int, cv::Scalar> colors;

	int rows = labelImg.rows;
	int cols = labelImg.cols;

	colorLabelImg.release();
	colorLabelImg.create(rows, cols, CV_8UC3);
	colorLabelImg = cv::Scalar::all(0);

	for (int i = 0; i < rows; i++)
	{
		const int* data_src = (int*)labelImg.ptr<int>(i);
		uchar* data_dst = colorLabelImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int pixelValue = data_src[j];
			if (pixelValue > 1)
			{
				if (colors.count(pixelValue) <= 0)
				{
					colors[pixelValue] = GetRandomColor();
				}

				cv::Scalar color = colors[pixelValue];
				*data_dst++ = color[0];
				*data_dst++ = color[1];
				*data_dst++ = color[2];
			}
			else
			{
				data_dst++;
				data_dst++;
				data_dst++;
			}
		}
	}
}

cv::Mat_<uchar> ocmu_maxconnecteddomain(cv::Mat_<uchar> binImg)
{
	cv::Mat_<uchar> maxRegion;

	cv::Mat_<uchar> contourImg;
	binImg.copyTo(contourImg);
	std::vector<std::vector<cv::Point>> contourVecs;
	cv::findContours(contourImg, contourVecs, CV_RETR_EXTERNAL, \
		CV_CHAIN_APPROX_NONE);

	if (contourVecs.size() > 0) { 
		double maxArea = 0;
		std::vector<cv::Point> maxContour;
		for (size_t i = 0; i < contourVecs.size(); i++) {
			double area = cv::contourArea(contourVecs[i]);
			if (area > maxArea) {
				maxArea = area;
				maxContour = contourVecs[i];
			}
		}

		cv::Rect maxRect = cv::boundingRect(maxContour);
		int xBegPos = maxRect.y;
		int yBegPos = maxRect.x;
		int xEndPos = xBegPos + maxRect.height;
		int yEndPos = yBegPos + maxRect.width;

		maxRegion = binImg(cv::Range(xBegPos, xEndPos), \
			cv::Range(yBegPos, yEndPos));
	}

	return maxRegion;
}

bool getGesture(Mat frameDest, Mat templa) {
	double sum;
	double total = templa.rows * templa.cols;
	double confidence = 1;
	double max = 1;
	Mat roi;
	Mat rec;
	for (int i = 0; i < frameDest.rows - templa.rows; i++) {
		for (int j = 0; j < frameDest.cols - templa.cols; j++) {
			sum = 0;
			roi = frameDest(Range(i, i + templa.rows), Range(j, j + templa.cols));
			absdiff(templa, roi, rec);
			sum = countNonZero(rec);
			confidence = sum / total;
			if (confidence < threshold_gesture && confidence < max) {
				cout << "find" << confidence << endl;
				max = confidence;
				result[0] = i;
				result[1] = j;
			}
		}
	}
	if (max != 1) {
		return true;
	}
	return false;
}

int getTime()
{
	return clock() / CLOCKS_PER_SEC;
}