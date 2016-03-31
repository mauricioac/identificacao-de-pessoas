/**
 * 
 */
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// g++ `pkg-config --cflags opencv` -o hello hello.cpp `pkg-config --libs opencv`
using namespace cv;
//crescimento de região
int  main()
{
	int c;
	VideoCapture cap("OneStopMoveEnter2cor.mpg");//lê video

	if (!cap.isOpened())
	{
		return 0;
	}

	namedWindow("Video",1);
	namedWindow("ACC",1);

	/**
	 * Corta a parte superior do frame, para
	 * eliminar a parte da informação de tempo
	 * da câmera
	 */
	Mat acc, frame, temp;
	Rect ROI(0,10, 384, 278);
	cap >> frame;
	temp = frame(ROI);
	cvtColor(temp, frame, COLOR_BGR2GRAY);

	acc = Mat::zeros(frame.size(), CV_32FC1);  // note: 3channel now
	
	while(1)
	{
		cap >> temp;
		if ( temp.empty() ) break;

		temp = temp(ROI);
		cvtColor(temp, frame, COLOR_BGR2GRAY);

		imshow("Video",frame);

		// cap.retrieve(frame); // video probably has 1 stream only
        Mat floatimg;
        frame.convertTo(floatimg, CV_32FC1);
        accumulateWeighted(frame, acc, 0.01);
        Mat res;
        convertScaleAbs(acc, res, 1, 0);

        Mat diff;
        subtract(res, frame, diff);

        imshow("ACC", diff);

        if (waitKey(30) == 27) {
        	break;
        }
	}

	return 0;
}