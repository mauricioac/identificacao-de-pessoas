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
#include <algorithm>

using namespace cv;
using namespace std;

class Objeto {
public:
  Point canto;
  vector<Point> contorno;
  Rect bb;
  Scalar cor;
};

float euclideanDist(Point& p, Point& q) {
    Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

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
    // namedWindow("Bin",0);//abre uma janela com o nome "Video"
    // namedWindow("BinOrig",0);//abre uma janela com o nome "Video"
    // namedWindow("BG",0);//abre uma janela com o nome "Video"
    // namedWindow("Diferenca",0);

    cap.release();//fecha video

    VideoCapture cap2("TownCentreXVID720x480.mpg");//lê video novamente

  vector<Objeto> objetos;

  bool pausado = false;

	while(1)
	{
		cap2 >> quadro;//pega o quadro
		if ( quadro.empty() ) break;

    Mat originalFrame = quadro.clone();

    pMOG2->apply(quadro, fgMaskMOG2);//,-0.5);
    Mat binaryImg;
    morphologyEx(fgMaskMOG2, binaryImg, CV_MOP_CLOSE, element);
    threshold(binaryImg, binaryImg, 128, 255, CV_THRESH_BINARY);
    Mat binOrig = binaryImg.clone();

    for (int i = 0; i < 2; i++)
      morphologyEx(binaryImg, binaryImg, CV_MOP_OPEN, element);

    for (int i = 0; i < 5; i++)
      morphologyEx(binaryImg, binaryImg, CV_MOP_DILATE, element);

    for (int i = 0; i < 3; i++)
      morphologyEx(binaryImg, binaryImg, CV_MOP_ERODE, element);

    for (int i = 0; i < 20; i++)
      morphologyEx(binaryImg, binaryImg, CV_MOP_OPEN, element);

    blur(binaryImg, binaryImg, Size(10,10) );
    Mat ContourImg = binaryImg.clone();


    vector< vector<Point> > _contornos;
    vector< vector<Point> > contornos;
    findContours(ContourImg, _contornos, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    for (int i = 0; i < (int) _contornos.size(); i++) {
      Rect bb = boundingRect(_contornos[i]);
      Point centro(bb.x + (bb.width / 2), bb.y + (bb.height / 2));

      int tamanho = 40;

      if (centro.y > (binaryImg.size().height / 2)) {
        tamanho = 60;
      }

      if (bb.width > tamanho) {
        Mat img2 = binaryImg.clone();
        img2 = Scalar(0,0,0);
        drawContours(img2, vector<vector<Point> >(1,_contornos[i]), -1, Scalar(255,255,255), -1, 8);
        Point p1(bb.x + (bb.width / 2), bb.y);
        Point p2(bb.x + (bb.width / 2), bb.y + bb.height);
        line(img2, p1, p2, Scalar(0,0,0), 2);

        vector< vector<Point> > divididos;
        findContours(img2, divididos, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        for (int k = 0; k < (int) divididos.size(); k++) {
          contornos.push_back(divididos[k]);
        }
      } else {
        contornos.push_back(_contornos[i]);
      }
    }


    vector<int> lixo;
    vector<bool> contorno_utilizado = vector<bool> ((int) contornos.size(), false);
    vector <Objeto> novos_objetos;

    for (int w = 0; w < (int)objetos.size(); w++) {
      int menor_indice = -1;
      double menor_distancia = std::numeric_limits<double>::max();
      bool achou = false;

      for (int i = 0; i < (int)contornos.size(); i++) {

        Rect bb = boundingRect(contornos[i]);
        Point canto(bb.x, bb.y);

        float res = euclideanDist(canto, objetos[w].canto);

        // caso trivial
        if (res < 15 && res < menor_distancia) {
          menor_distancia = res;
          menor_indice = i;
          achou = true;
        }
      }

      if (achou) {
        Rect _b = boundingRect(contornos[menor_indice]);

        bool alturaRadical = abs(objetos[w].bb.height - _b.height) > 30;
        bool larguraRadical = abs(objetos[w].bb.width - _b.width) > 30;

        int caso = 1;

        // if (larguraRadical)
        // {
        //   caso = 2;
        // }

        switch (caso) {
          case 1:
            objetos[w].contorno = contornos[menor_indice];
            objetos[w].bb = boundingRect(contornos[menor_indice]);
            objetos[w].canto = Point(objetos[w].bb.x, objetos[w].bb.y);
            contorno_utilizado[menor_indice] = true;
            novos_objetos.push_back(objetos[w]);

            break;
          case 2:
            // split motherfucker

            contorno_utilizado[menor_indice] = true;
            cout << "PQP!" << endl;
            Rect bb = boundingRect(contornos[menor_indice]);
            Mat img2 = binaryImg.clone();
            img2 = Scalar(0,0,0);
            drawContours(img2, vector<vector<Point> >(1,contornos[menor_indice]), -1, Scalar(255,255,255), -1, 8);
            Point p1(bb.x + (bb.width / 2), bb.y);
            Point p2(bb.x + (bb.width / 2), bb.y + bb.height);
            line(img2, p1, p2, Scalar(0,0,0), 2);

            // imshow("Video", img2);
            // waitKey(4000);

            vector< vector<Point> > divididos;
            findContours(img2, divididos, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

            double min = std::numeric_limits<double>::max();
            int min_idx = -1;

            for (int k = 0; k < (int) divididos.size(); k++) {
              Rect bb2 = boundingRect(divididos[k]);
              Point canto(bb.x + bb2.x, bb.y + bb2.y);

              float res = euclideanDist(canto, objetos[w].canto);

              if (res < min) {
                min = res;
                min_idx = k;
              }
            }

            if (min_idx > -1) {
              objetos[w].contorno = divididos[min_idx];
              objetos[w].bb = boundingRect(divididos[min_idx]);
              objetos[w].canto = Point(objetos[w].bb.x, objetos[w].bb.y);
              novos_objetos.push_back(objetos[w]);

              cout << "TOMA NO CU" << endl;
              pausado = true;

              for (int k = 0; k < (int) divididos.size(); k++) {
                if (k != min_idx) {

                  double obj_dist = std::numeric_limits<double>::max();
                  int bejetinho_idx = -1;

                  for (int u = 0; u < (int) objetos.size(); u++) {
                    Rect bt = boundingRect(divididos[k]);
                    Point canto = Point(bt.x, bt.y);

                    float res = euclideanDist(canto, objetos[u].canto);

                    if (res < 15 && res < obj_dist) {
                      obj_dist = res;
                      bejetinho_idx = u;
                    }
                  }

                  if (bejetinho_idx > -1) {
                    objetos[bejetinho_idx].contorno = divididos[k];
                    objetos[bejetinho_idx].bb = boundingRect(divididos[k]);
                    objetos[bejetinho_idx].canto = Point(objetos[bejetinho_idx].bb.x, objetos[bejetinho_idx].bb.y);
                    novos_objetos.push_back(objetos[bejetinho_idx]);
                  } else {
                    contornos.push_back(divididos[k]);
                  }
                }
              }
            }

            break;
        }
      }
      else if(menor_indice > -1)
        contorno_utilizado[menor_indice] = false;
    }

    for (int i = 0; i < (int) contornos.size(); i++) {

      if (contorno_utilizado[i]) {
        continue;
      }

      Rect rect1  = boundingRect(contornos[i]);


      if ( rect1.height < 20  ||  rect1.width > 150   )
      {
        continue;
      }
      else if ( ( (float) rect1.height / (float) rect1.width ) < 1.5f)
      {
        continue;
      }
      else if ( rect1.area() < 100.0f)
      {
        continue;
      }

      Point canto(rect1.x, rect1.y);

      Objeto novo;
      novo.canto = canto;
      novo.contorno = contornos[i];
      novo.cor = Scalar(randomiza(120, 200), randomiza(120,200), randomiza(120,200));


      novos_objetos.push_back(novo);
    }

    objetos = novos_objetos;

    for (Objeto o : objetos) {
      vector<Point> tmp = o.contorno;
      const Point* pts[1] = { &tmp[0] };
      int s = (int) o.contorno.size();

      fillPoly(originalFrame, pts, &s, 1, o.cor);
      rectangle(originalFrame, o.bb, Scalar(255,0,0));
    }


    imshow("Video",originalFrame);//exibe o video
    // imshow("Bin",binaryImg);//exibe o video
    // imshow("BinOrig",binOrig);//exibe o video
    // imshow("Diferenca",t);//exibe diferença

    //espera comando de saída "ESC"
    if (waitKey(30) == 27)
    {
    	break;
    }

    // if (pausado) {
    //   waitKey(7000);
    //   pausado = false;
    // }
	}

	return 0;
}
