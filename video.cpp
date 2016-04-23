/**
 * Trabalho 1: Detecção de pessoas em movimento
 * ----
 * Alunos: Silvana Trindade e Maurício André Cinelli
 */
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
 #include <random>
#include <stack>
#include <vector>

using namespace cv;
using namespace std;

class Objeto {
public:
  Point centro;
  vector<Point> contorno;
  Scalar cor;
};

int randomiza(int min,int max) { random_device rd; mt19937_64 gen(rd()); uniform_int_distribution<> dis(min, max); return dis(gen); }

int  main()
{
    int cont_cor = 0;
	int c;
	VideoCapture cap("TownCentreXVID720x480.mpg");//lê video

	if (!cap.isOpened())
	{
        cout << "Erro ao ler o vídeo." << endl;
		return 0;
	}

    // MultiTracker tracker("TLD");

	/**
	 * Corta a parte superior do quadro, para
	 * eliminar a parte da informação de tempo
	 * da câmera
	 */
	Mat media_corrida, quadro, temp;
	Rect ROI(0,10, 384, 278);
	cap >> quadro;

	media_corrida = Mat::zeros(quadro.size(), CV_32FC3);  // preenche com 0 (zero) a imagem do acumulador

    // accumulateWeighted(quadro, media_corrida, 1);//pega imagem de fundo e aplica como acumulado

    //elemento estruturante com a matriz de 3X7
    Mat elemento = getStructuringElement(MORPH_RECT, Size(3, 7));

    /**
     * Percorre todo o video, para
     * encontrar um background melhor
     */
    cout << "Processando vídeo." << endl;
    Ptr< BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2(500,60,true);
    Mat fgMaskMOG2;

    pMOG2->setBackgroundRatio(0.001);
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 7), Point(1,3) );

    while(1)
    {
      cap >> quadro;
      if ( quadro.empty() ) break;

      pMOG2->apply(quadro, fgMaskMOG2);//,-0.5);
    }
    cout << "Iniciando reconhecimento de pessoas." << endl;

    namedWindow("Video",0);//abre uma janela com o nome "Video"
    namedWindow("Bin",0);//abre uma janela com o nome "Video"
    namedWindow("BinOrig",0);//abre uma janela com o nome "Video"
    namedWindow("BG",0);//abre uma janela com o nome "Video"
    // namedWindow("Diferenca",0);

    cap.release();//fecha video

    VideoCapture cap2("TownCentreXVID720x480.mpg");//lê video novamente

  vector<Objeto> objetos;

	while(1)
	{
		cap2 >> quadro;//pega o quadro
		if ( quadro.empty() ) break;

    Mat originalFrame = quadro.clone();
    // blur(quadro, quadro, Size(10,10) );
    pMOG2->apply(quadro, fgMaskMOG2);//,-0.5);
    Mat binaryImg;
    morphologyEx(fgMaskMOG2, binaryImg, CV_MOP_CLOSE, element);
    threshold(binaryImg, binaryImg, 128, 255, CV_THRESH_BINARY);
    Mat binOrig = binaryImg.clone();

    for (int i = 0; i < 5; i++)
      morphologyEx(binaryImg, binaryImg, CV_MOP_DILATE, element);

    for (int i = 0; i < 2; i++)
      morphologyEx(binaryImg, binaryImg, CV_MOP_ERODE, element);

    for (int i = 0; i < 2; i++)
      morphologyEx(binaryImg, binaryImg, CV_MOP_OPEN, element);

    Mat ContourImg = binaryImg.clone();

    vector< vector<Point> > contornos;
    findContours(ContourImg, contornos, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    vector<Objeto> obj_detectados;

    for (int i = 0; i < (int) contornos.size(); i++)
    {
        Rect rect1  = boundingRect(contornos[i]);

        if ( (rect1.height == rect1.width)  ||  rect1.width > 150 || rect1.height < 70)
        {
          continue;
        }

        bool achou = false;
        Scalar cor;
        Point centro(rect1.x + (rect1.width / 2), rect1.y + (rect1.height / 2));

        int menor_indice = -1;
        double menor_distancia = 9999;

        for (int j = 0; j < (int) objetos.size(); j++) {
          double res = cv::norm(centro - objetos[j].centro);
          // cout << res << endl;

          if (res < 30 && res < menor_distancia) {
            menor_distancia = res;
            menor_indice = j;
            achou = true;
          }
        }

        if (achou) {
          objetos[menor_indice].contorno = contornos[i];
          objetos[menor_indice].centro = centro;
          cor = objetos[menor_indice].cor;

          obj_detectados.push_back(objetos[menor_indice]);
        } else {
          Objeto novo;
          novo.centro = centro;
          novo.contorno = contornos[i];
          novo.cor = Scalar(randomiza(120, 200), randomiza(120,200), randomiza(120,200));

          cor = novo.cor;

          obj_detectados.push_back(novo);
        }

        // cout << objetos.size() << endl;

        vector<Point> tmp = contornos[i];
        const Point* pts[1] = { &tmp[0] };
        int s = (int) contornos[i].size();

        fillPoly(originalFrame, pts, &s, 1, cor);
    }

    for (Objeto o : obj_detectados) {
      Rect r = boundingRect(o.contorno);
      rectangle(originalFrame, r, Scalar(0,255,0));
    }

    objetos = obj_detectados;

    imshow("Video",originalFrame);//exibe o video
    imshow("Bin",binaryImg);//exibe o video
    imshow("BinOrig",binOrig);//exibe o video
    imshow("BG",fgMaskMOG2);//exibe o video
    // imshow("Diferenca",t);//exibe diferença

    //espera comando de saída "ESC"
    if (waitKey(30) == 27)
    {
    	break;
    }
	}

	return 0;
}
