#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stack>
#include <vector>

// g++ `pkg-config --cflags opencv` -o hello hello.cpp `pkg-config --libs opencv`
using namespace cv;
using namespace std;

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

    Mat element = getStructuringElement(MORPH_RECT, Size(3, 7), Point(1,3) ); 

    while(1)
    {
        cap >> temp;
        if ( temp.empty() ) break;

        frame = temp(ROI);

        Mat floatimg;
        frame.convertTo(floatimg, CV_32FC3);
        accumulateWeighted(frame, acc, 0.001);
    }

    cap.release();

    VideoCapture cap2("OneStopMoveEnter2cor.mpg");//lê video

	while(1)
	{
		cap2 >> temp;
		if ( temp.empty() ) break;

		frame = temp(ROI);

        Mat floatimg;
        frame.convertTo(floatimg, CV_32FC3);
        accumulateWeighted(frame, acc, 0.0001);
        Mat res;
        convertScaleAbs(acc, res, 1, 0);

        Mat diff;
        subtract(res, frame, diff);

        vector<vector<Point> > contours;
        cv::Mat thresholded;
        cv::cvtColor(diff, thresholded, CV_RGB2GRAY);
        morphologyEx(thresholded, thresholded, CV_MOP_CLOSE, element);
        morphologyEx(thresholded, thresholded, CV_MOP_CLOSE, element);
        cv::threshold(thresholded, thresholded, 50, 255, CV_THRESH_BINARY);
        findContours(thresholded, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < (int) contours.size(); i++) {
            Rect rect  = boundingRect(contours[i]);
            if (rect.area() > 30)
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