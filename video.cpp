/**
 * Trabalho 1: Detecção de pessoas em movimento
 * ----
 * Alunos: Silvana Trindade e Maurício André Cinelli
 */
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
        cout << "Erro ao ler o vídeo." << endl;
		return 0;
	}

	/**
	 * Corta a parte superior do frame, para
	 * eliminar a parte da informação de tempo
	 * da câmera
	 */
	Mat acc, frame, temp;
	Rect ROI(0,10, 384, 278);
	frame = imread("fundo.png",CV_LOAD_IMAGE_COLOR);//lê plano de fundo inicial
	frame = frame(ROI);

	acc = Mat::zeros(frame.size(), CV_32FC3);  // preenche com 0 (zero) a imagem do acumulador
	
    accumulateWeighted(frame, acc, 1);//pega imagem de fundo e aplica como acumulado

    //elemento estruturante com a matriz de 3X7  
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 7)); 

    /**
     * Percorre todo o video, para 
     * encontrar um background melhor
     */
    cout << "Processando vídeo." << endl;
    while(1)
    {
        cap >> temp;
        if ( temp.empty() ) break;

        frame = temp(ROI);

        Mat floatimg;
        frame.convertTo(floatimg, CV_32FC3);

        accumulateWeighted(frame, acc, 0.001);//aplica média corrida
    }
    cout << "Iniciando reconhecimento de pessoas." << endl;
   
    namedWindow("Video",0);//abre uma janela com o nome "Video"
    namedWindow("Diferenca",0);

    cap.release();//fecha video

    VideoCapture cap2("OneStopMoveEnter2cor.mpg");//lê video novamente

	while(1)
	{
		cap2 >> temp;//pega o frame
		if ( temp.empty() ) break;

		frame = temp(ROI);//corta a informação contendo data e hora

        accumulateWeighted(frame, acc, 0.0001);
        Mat res;//plano de fundo
        convertScaleAbs(acc, res);

        Mat diff;
        subtract(res, frame, diff);//subtrai o plano de fundo do frame atual

        vector<vector<Point> > contours;
        cv::Mat thresholded;
        cv::cvtColor(diff, thresholded, CV_RGB2GRAY);//converte frame para escala de cinza
        
        //Realiza duas operações de fechamento (dilatação+erosão)
        morphologyEx(thresholded, thresholded, CV_MOP_CLOSE, element);
        morphologyEx(thresholded, thresholded, CV_MOP_CLOSE, element);
        
        //Aplica um limiar, se a intensidade do pixel passar de 50 é branco
        cv::threshold(thresholded, thresholded, 50, 255, CV_THRESH_BINARY);
        Mat t = thresholded.clone();//faz uma cópia

        //Encontra contornos na imagem
        //Internamente utiliza crescimento de regiões
        //Retorna somente os retângulos mais externos
        findContours(thresholded, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < (int) contours.size(); i++) 
        {
            //pega o retângulo que engloba o contorno
            Rect rect  = boundingRect(contours[i]);

            if (rect.area() > 30)
            {
        	   rectangle(frame, rect, Scalar(255, 0, 0));
            }
        }

        imshow("Video",frame);//exibe o video 
        imshow("Diferenca",t);//exibe diferença
        
        //espera comando de saída "ESC"
        if (waitKey(30) == 27) 
        {
        	break;
        }
	}

	return 0;
}