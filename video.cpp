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
	 * Corta a parte superior do quadro, para
	 * eliminar a parte da informação de tempo
	 * da câmera
	 */
	Mat media_corrida, quadro, temp;
	Rect ROI(0,10, 384, 278);
	quadro = imread("fundo.png",CV_LOAD_IMAGE_COLOR);//lê plano de fundo inicial
	quadro = quadro(ROI);

	media_corrida = Mat::zeros(quadro.size(), CV_32FC3);  // preenche com 0 (zero) a imagem do acumulador

    accumulateWeighted(quadro, media_corrida, 1);//pega imagem de fundo e aplica como acumulado

    //elemento estruturante com a matriz de 3X7
    Mat elemento = getStructuringElement(MORPH_RECT, Size(3, 7));

    /**
     * Percorre todo o video, para
     * encontrar um background melhor
     */
    cout << "Processando vídeo." << endl;
    while(1)
    {
        cap >> temp;
        if ( temp.empty() ) break;

        quadro = temp(ROI);

        accumulateWeighted(quadro, media_corrida, 0.001);//aplica média corrida
    }
    cout << "Iniciando reconhecimento de pessoas." << endl;

    namedWindow("Video",0);//abre uma janela com o nome "Video"
    namedWindow("Diferenca",0);

    cap.release();//fecha video

    VideoCapture cap2("OneStopMoveEnter2cor.mpg");//lê video novamente

	while(1)
	{
		cap2 >> temp;//pega o quadro
		if ( temp.empty() ) break;

		quadro = temp(ROI);//corta a informação contendo data e hora

        accumulateWeighted(quadro, media_corrida, 0.0001);
        Mat fundo;//plano de fundo
        convertScaleAbs(media_corrida, fundo);

        Mat diff;
        subtract(fundo, quadro, diff);//subtrai o plano de fundo do quadro atual

        vector<vector<Point> > contornos;
        cv::Mat imagem_binaria;
        cv::cvtColor(diff, imagem_binaria, CV_RGB2GRAY);//converte quadro para escala de cinza

        //Realiza duas operações de fechamento (dilatação+erosão)
        morphologyEx(imagem_binaria, imagem_binaria, CV_MOP_CLOSE, elemento);
        morphologyEx(imagem_binaria, imagem_binaria, CV_MOP_CLOSE, elemento);

        //Aplica um limiar, se a intensidade do pixel passar de 50 é branco
        cv::threshold(imagem_binaria, imagem_binaria, 50, 255, CV_THRESH_BINARY);
        Mat t = imagem_binaria.clone();//faz uma cópia para mostrar depois

        //Encontra contornos na imagem
        //Internamente utiliza crescimento de regiões
        //Retorna somente os retângulos mais externos
        findContours(imagem_binaria, contornos, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < (int) contornos.size(); i++)
        {
            //pega o retângulo que engloba o contorno
            Rect rect  = boundingRect(contornos[i]);

            if (rect.area() > 30)
            {
        	   rectangle(quadro, rect, Scalar(255, 0, 0));
            }
        }

        imshow("Video",quadro);//exibe o video
        imshow("Diferenca",t);//exibe diferença

        //espera comando de saída "ESC"
        if (waitKey(30) == 27)
        {
        	break;
        }
	}

	return 0;
}
