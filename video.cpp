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

// Classe usada tracking dos objetos
class Objeto {
public:
  Point centro;
  vector<Point> contorno;
  Rect bb; // bounding box, ou retangulo ao redor do contorno
  Scalar cor;
};

float distanciaEuclidiana(Point& p, Point& q) {
    Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

int randomiza(int min,int max) { random_device rd; mt19937_64 gen(rd()); uniform_int_distribution<> dis(min, max); return dis(gen); }

int  main()
{
	VideoCapture cap("TownCentreXVID720x480.mpg");//lê video

	if (!cap.isOpened())
	{
        cout << "Erro ao ler o vídeo." << endl;
		return 0;
	}

	Mat quadro;

  // elemento estruturante com a matriz de 3X7
  // usado para dilate, erode e open
  Mat elemento = getStructuringElement(MORPH_RECT, Size(3, 7), Point(1,3) );

  /**
   * Percorre todo o video, para
   * encontrar um background melhor
   */
  cout << "Processando vídeo." << endl;

  // Usa subtração de background por Misturas de Gauss
  Ptr< BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2(500,60,true);
  Mat mascara_background;

  mog2->setBackgroundRatio(0.001);

  while(1)
  {
    cap >> quadro;
    if ( quadro.empty() ) break;

    mog2->apply(quadro, mascara_background);//,-0.5);
  }

  cout << "Iniciando reconhecimento de pessoas." << endl;

  namedWindow("Video",0);//abre uma janela com o nome "Video"
  // namedWindow("Bin",0);//abre uma janela com o nome "Video"
  // namedWindow("BinOrig",0);//abre uma janela com o nome "Video"
  // namedWindow("BG",0);//abre uma janela com o nome "Video"
  // namedWindow("Diferenca",0);

  cap.release();//fecha video

  // diminui o aprendizado de background
  mog2->setBackgroundRatio(0.0001);

  VideoCapture cap2("TownCentreXVID720x480.mpg");//lê video novamente

  // inicia lista de objetos do nosso tracker de objetos
  vector<Objeto> objetos;

	while(1)
	{
		cap2 >> quadro;//pega o quadro
		if ( quadro.empty() ) break;

    Mat originalFrame = quadro.clone();

    Mat binaryImg;
    // passa mascara do MOG2 para imagem binaria
    // já realizando um close para reduzir ruidos
    morphologyEx(mascara_background, binaryImg, CV_MOP_CLOSE, elemento);

    // remove sombras
    threshold(binaryImg, binaryImg, 128, 255, CV_THRESH_BINARY);
    Mat binOrig = binaryImg.clone();

    // Aplica quadro atual ao subtrator de background
    mog2->apply(quadro, mascara_background);

    // Separa um pouco os objetos
    for (int i = 0; i < 2; i++)
      morphologyEx(binaryImg, binaryImg, CV_MOP_OPEN, elemento);

    // Faz pessoas/objetos virarem bolhas
    for (int i = 0; i < 5; i++)
      morphologyEx(binaryImg, binaryImg, CV_MOP_DILATE, elemento);

    // Diminui um pouco
    for (int i = 0; i < 3; i++)
      morphologyEx(binaryImg, binaryImg, CV_MOP_ERODE, elemento);

    // Se pessoas se juntarem por poucos pixels
    // aqui separamos elas
    for (int i = 0; i < 20; i++)
      morphologyEx(binaryImg, binaryImg, CV_MOP_OPEN, elemento);

    // Embaça imagem binária
    blur(binaryImg, binaryImg, Size(10,10) );

    // clona imagem binaria porque
    // a findContours "estraga" a imagem
    Mat ContourImg = binaryImg.clone();

    vector< vector<Point> > _contornos;
    vector< vector<Point> > contornos;
    findContours(ContourImg, _contornos, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // Armazena os terços da largura e altura do frame atual
    double hframe = binaryImg.size().height / 3.0f;
    double wframe = binaryImg.size().width / 3.0f;

    // Realiza cortes de contornos por largura
    // dividindo a imagem em 9 quadrantes
    //
    // "corrige perspectiva"
    for (int i = 0; i < (int) _contornos.size(); i++) {
      Rect bb = boundingRect(_contornos[i]);
      Point centro(bb.x + (bb.width / 2), bb.y + (bb.height / 2));

      int largura = 40;

      if (centro.y < hframe) {
        if (centro.x < wframe) {
          largura = 35;
        } else if (centro.x < wframe * 2) {
          largura = 25;
        } else {
          largura = 35;
        }
      } else if (centro.y < hframe * 2) {
        if (bb.width < 42 && bb.height < 70) {
          continue;
        }

        if (centro.x < wframe) {
          largura = 50;
        } else if (centro.x < wframe * 2) {
          largura = 52;
        } else {
          largura = 50;
        }
      } else {
        if (bb.width < 42 && bb.height < 70) {
          continue;
        }

        if (centro.x < wframe) {
          largura = 42;
        } else if (centro.x < wframe * 2) {
          largura = 45;
        } else {
          largura = 55;
        }
      }

      // Se contorno for maior que largura de corte E
      // essa largura for o dobro pelo menos, então corta
      //
      // corte é feito desenhando uma linha vertical preta e chamando
      // a findContours novamente
      if (bb.width > largura && (bb.width - largura) > (largura / 2.0f)) {
        Mat img2 = binaryImg.clone();
        img2 = Scalar(0,0,0);
        drawContours(img2, vector<vector<Point> >(1,_contornos[i]), -1, Scalar(255,255,255), -1, 8);
        Point p1(bb.x + largura, bb.y);
        Point p2(bb.x + largura, bb.y + bb.height);
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

    // transforma contornos numa variavel temporaria
    // de novo, pois vamos cortar por altura
    _contornos = contornos;
    contornos.clear();

    // corte de contornos por altura
    for (int i = 0; i < (int) _contornos.size(); i++) {
      Rect bb = boundingRect(_contornos[i]);
      Point centro(bb.x + (bb.width / 2), bb.y + (bb.height / 2));

      int altura = 130;

      if (bb.height > altura && (bb.height - altura) > (altura / 3.0f)) {
        Mat img2 = binaryImg.clone();
        img2 = Scalar(0,0,0);
        drawContours(img2, vector<vector<Point> >(1,_contornos[i]), -1, Scalar(255,255,255), -1, 8);
        Point p1(bb.x, bb.y + bb.height - altura);
        Point p2(bb.x + bb.width, bb.y + bb.height - altura);
        line(img2, p1, p2, Scalar(0,0,0), 2);

        vector< vector<Point> > divididos;
        findContours(img2, divididos, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        for (int k = 0; k < (int) divididos.size(); k++) {
          contornos.push_back(divididos[k]);
        }
      }
      else {
        contornos.push_back(_contornos[i]);
      }
    }

    // para vermos quais são novos
    vector<bool> contorno_utilizado = vector<bool> ((int) contornos.size(), false);

    // objetos que detectarem o seu respectivo contorno no frame atual
    // visto que se não encontrar, não entra nesse frame, assim
    // limpando objetos ao sair do video
    vector <Objeto> novos_objetos;

    // percorre cada objeto, tentando encontrar
    // um contorno que estiver mais perto do
    // contorno no frame anterior
    for (int w = 0; w < (int)objetos.size(); w++) {
      int menor_indice = -1;
      double menor_distancia = std::numeric_limits<double>::max();
      bool achou = false;

      for (int i = 0; i < (int)contornos.size(); i++) {

        Rect bb = boundingRect(contornos[i]);
        Point centro(bb.x + (bb.width / 2), bb.y + (bb.height / 2));

        float res = distanciaEuclidiana(centro, objetos[w].centro);

        if (res < 20 && res < menor_distancia) {
          menor_distancia = res;
          menor_indice = i;
          achou = true;
        }
      }

      if (achou) {
        Rect _b = boundingRect(contornos[menor_indice]);

        objetos[w].contorno = contornos[menor_indice];
        objetos[w].bb = boundingRect(contornos[menor_indice]);
        objetos[w].centro = Point(objetos[w].bb.x + (objetos[w].bb.width / 2), objetos[w].bb.y + (objetos[w].bb.height / 2));
        contorno_utilizado[menor_indice] = true;
        novos_objetos.push_back(objetos[w]);
      }
      else if(menor_indice > -1)
        contorno_utilizado[menor_indice] = false;
    }

    for (int i = 0; i < (int) contornos.size(); i++) {
      // se um objeto detectou este contorno, ignora
      if (contorno_utilizado[i]) {
        continue;
      }

      Rect rect1  = boundingRect(contornos[i]);

      if ( rect1.width < 8 || rect1.height < 20  ||  rect1.width > 150   )
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

      Point centro(rect1.x + (rect1.width / 2), rect1.y + (rect1.height / 2));

      Objeto novo;
      novo.centro = centro;
      novo.contorno = contornos[i];
      novo.cor = Scalar(randomiza(120, 200), randomiza(120,200), randomiza(120,200));


      novos_objetos.push_back(novo);
    }

    objetos = novos_objetos;

    // desenha objetos na imagem final
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
	}

	return 0;
}
