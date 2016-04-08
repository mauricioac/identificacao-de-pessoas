#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stack>
#include <vector>

// g++ `pkg-config --cflags opencv` -o hello hello.cpp `pkg-config --libs opencv`
using namespace cv;
using namespace std;
//crescimento de região

	

// void Traverse(int xs, int ys, cv::Mat &ids,cv::Mat &image, int blobID, cv::Point &leftTop, cv::Point &rightBottom) {
//     std::stack<cv::Point> S;
//     S.push(cv::Point(xs,ys));

//     while (!S.empty()) {
//         cv::Point u = S.top();
//         S.pop();

//         int x = u.x;
//         int y = u.y;

//         if (image.at<unsigned char>(y,x) == 0 || ids.at<unsigned char>(y,x) > 0)
//             continue;

//         ids.at<unsigned char>(y,x) = blobID;
//         if (x < leftTop.x)
//             leftTop.x = x;
//         if (x > rightBottom.x)
//             rightBottom.x = x;
//         if (y < leftTop.y)
//             leftTop.y = y;
//         if (y > rightBottom.y)
//             rightBottom.y = y;

//         if (x > 0)
//             S.push(cv::Point(x-1,y));
//         if (x < ids.cols-1)
//             S.push(cv::Point(x+1,y));
//         if (y > 0)
//             S.push(cv::Point(x,y-1));
//         if (y < ids.rows-1)
//             S.push(cv::Point(x,y+1));
//     }


// }

// int FindBlobs(cv::Mat &image, std::vector<cv::Rect> &out, float minArea) {
//     cv::Mat ids = cv::Mat::zeros(image.size(),CV_32FC1);
//     cv::Mat thresholded;
//     cv::cvtColor(image, thresholded, CV_RGB2GRAY);
//     const int thresholdLevel = 50;
//     cv::threshold(thresholded, thresholded, thresholdLevel, 255, CV_THRESH_BINARY);
//     int blobId = 1;
//     for (int x = 0;x<ids.cols;x++)
//         for (int y=0;y<ids.rows;y++){
//             if (thresholded.at<unsigned char>(y,x) > 0 && ids.at<unsigned char>(y,x) == 0) {
//                 cv::Point leftTop(ids.cols-1, ids.rows-1), rightBottom(0,0);
//                 Traverse(x,y,ids, thresholded,blobId++, leftTop, rightBottom);
//                 cv::Rect r(leftTop, rightBottom);
//                 if (r.area() > minArea)
//                     out.push_back(r);
//             }
//         }
//     return blobId;
// }

int  main()
{
	int c;
	VideoCapture cap("OneStopMoveEnter2cor.mpg");//lê video

	if (!cap.isOpened())
	{
		return 0;
	}

	namedWindow("Video",0);
    // namedWindow("THRESH",1);
    // namedWindow("DIFF",1);
	// namedWindow("ACC",1);

	/**
	 * Corta a parte superior do frame, para
	 * eliminar a parte da informação de tempo
	 * da câmera
	 */
	Mat acc, frame, temp;
	Rect ROI(0,10, 384, 278);
	frame = imread("fundo.png",CV_LOAD_IMAGE_COLOR);
	frame = frame(ROI);

	acc = Mat::zeros(frame.size(), CV_32FC3);  // note: 3channel now
	
    accumulateWeighted(frame, acc, 1);

	while(1)
	{
		cap >> temp;
		if ( temp.empty() ) break;

		frame = temp(ROI);

		// cap.retrieve(frame); // video probably has 1 stream only
        Mat floatimg;
        frame.convertTo(floatimg, CV_32FC3);
        accumulateWeighted(frame, acc, 0.001);
        Mat res;
        convertScaleAbs(acc, res, 1, 0);

        Mat diff;
        subtract(res, frame, diff);

        vector<vector<Point> > contours;
        cv::Mat thresholded;
        cv::cvtColor(diff, thresholded, CV_RGB2GRAY);
        cv::threshold(thresholded, thresholded, 50, 255, CV_THRESH_BINARY);
        findContours(thresholded, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < (int) contours.size(); i++) {
            Rect rect  = boundingRect(contours[i]);
            if (rect.area() > 45)
        	   rectangle(frame, rect, Scalar(255, 0, 0));
        }

        imshow("Video",frame);
        // imshow("THRESH", thresholded);
        // imshow("DIFF", diff);
        // imshow("ACC", res);

        if (waitKey(30) == 27) {
        	break;
        }
	}

	return 0;
}