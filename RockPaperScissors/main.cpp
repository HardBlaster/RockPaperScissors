#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <conio.h>

#define KEY_UP 72
#define KEY_DOWN 80
#define KEY_LEFT 75
#define KEY_RIGHT 77

using namespace std;
using namespace cv;

float convexity(const vector<Point>& cont) {
	vector<Point> hull;
	convexHull(cont, hull, false, true);
	return static_cast<float>(arcLength(hull, true) / arcLength(cont, true));
}

float circularity(const vector<Point>& cont) {
	double k = arcLength(cont, true);
	return static_cast<float>(4 * CV_PI * contourArea(cont) / (k * k));
}

//A befoglal� t�glalap k�nnyen v�ltozhat, ha az objektumokt elforgatjuk
float cextent(const vector<Point>& cont) {
	double area = contourArea(cont);
	Rect r = boundingRect(cont);
	return static_cast<float>(area / (r.width * r.height));
}

//Ha ellipszist illeszt�nk az objektumra, akkor jobbn j�runk.
float cextent2(const vector<Point>& cont) {
	double area = contourArea(cont);
	RotatedRect r = fitEllipse(cont);
	return static_cast<float>(r.size.aspectRatio());
}


float diameter(const vector<Point>& cont) {
	double maxd = 0, d;
	for (auto p : cont) {
		for (auto q : cont) {
			d = cv::norm(p - q);
			if (d > maxd)
				maxd = d;
		}
	}
	return static_cast<float>(maxd);
}

void createHisto(const cv::Mat img, cv::Mat& histo) {

	vector<Mat> kepek;
	kepek.push_back(img); // egy k�pet haszn�lunk

	vector<int> csatornak;
	csatornak.push_back(0); // annak az egy k�pnek a 0. csatorn�j�t haszn�ljuk

	vector<int> hiszto_meretek;
	hiszto_meretek.push_back(256);  //minden vil�goss�gk�dot k�l�n sz�molunk

	vector<float> hiszto_tartomanyok;
	hiszto_tartomanyok.push_back(0.0f); //hol kezd�dik a tartom�ny
	hiszto_tartomanyok.push_back(255);  //meddig tart	

	//accumlate: marad false (null�zza a hisztogrammot)
	calcHist(kepek, csatornak, noArray(), histo, hiszto_meretek, hiszto_tartomanyok, false);
}

vector<cv::Point> preProcessing(Mat src, Mat& dest) {
	Mat mask;
	
	resize(src, dest, Size(), 0.1, 0.1);

	GaussianBlur(dest, dest, Size(5, 5), 0);
	threshold(dest, mask, 45, 255, THRESH_BINARY);
	Mat se = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));

	dilate(mask, mask, se);
	//imshow("dilate",di);
	cvtColor(mask, mask, COLOR_BGR2GRAY);
	//imshow("mask", mask);

	vector<vector<cv::Point>> conts;
	findContours(mask.clone(), conts, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	assert(conts.size() == 1);

	return conts[0];
}

Mat judge(const int thRound) {
	vector<vector<cv::Point>> conts;

	Mat rock = imread("game/starter/rock.jpg");
	conts.push_back(preProcessing(rock, rock));

	Mat paper = imread("game/starter/paper.jpg");
	conts.push_back(preProcessing(paper, paper));

	Mat scissor = imread("game/starter/scissor.jpg");
	conts.push_back(preProcessing(scissor, scissor));

	Mat player1 = imread("game/player1/" + to_string(thRound) + ".jpg");
	conts.push_back(preProcessing(player1, player1));

	Mat player2 = imread("game/player2/" + to_string(thRound) + ".jpg");
	conts.push_back(preProcessing(player2, player2));

	Mat data(5, 3, CV_32F);
	for (int i = 0; i < 5; ++i) {

		data.at<float>(i, 0) = circularity(conts[i]);
		data.at<float>(i, 1) = convexity(conts[i]);
		data.at<float>(i, 2) = cextent2(conts[i]);

	}

	TermCriteria crit(TermCriteria::Type::EPS | TermCriteria::Type::MAX_ITER, 100, 0.001);

	Mat labels;
	kmeans(data, 3, labels, crit, 3, KMEANS_RANDOM_CENTERS);

	return labels;
}

int main() {

	int korszam = 1;
	bool kilep = true;

	while (kilep)
	{
		Mat labels = judge(korszam);

		cout << "Number of turns: " << korszam << endl;

		int player1 = labels.at<int>(3); //3 player 1, 4 player 2
		int player2 = labels.at<int>(4); //0 kó 1 papír és 2 olló

		//cout << "Player 1 = " << player1 << endl;
		//cout << "Player 2 = " << player2 << endl;

		if (player1 == player2)
			cout << "Draw." << endl;
			
		bool p1 = false;
		bool p2 = false;

		switch (player1)
		{
			case 0:
			{
				if (player2 == 1)
					p2 = true;
				if (player2 == 2)
					p1 = true;
			}
			case 1:
			{
				if (player2 == 0)
					p1 = true;
				if (player2 == 2)
					p2 = true;
			}
			case 2:
			{
				if (player2 == 0)
					p2 = true;
				if (player2 == 1)
					p1 = true;
			}
		}
		
		Mat Mplayer1 = imread("game/player1/" + to_string(korszam) + ".jpg");
		Mat Mplayer2 = imread("game/player2/" + to_string(korszam) + ".jpg");

		Mat kep1_2;
		Mat kep2_2;
		resize(Mplayer1, kep1_2, Size(), 0.1, 0.1);
		resize(Mplayer2, kep2_2, Size(), 0.1, 0.1);

		if (p1 == true && !p2)
		{
			imshow("Player 1 - winner", kep1_2);
			imshow("Player 2 - looser", kep2_2);
			moveWindow("Player 1 - winner", 0, 0);
			moveWindow("Player 2 - looser", 300, 0);
		}
		else if(p2 == true && !p1)
		{
			imshow("Player 1 - looser", kep1_2);
			imshow("Player 2 - winner", kep2_2);
			moveWindow("Player 1 - looser", 0, 0);
			moveWindow("Player 2 - winner", 300, 0);
		}
		else if (!p2 && !p1)
		{
			imshow("Player 1 - Draw", kep1_2);
			imshow("Player 2 - Draw", kep2_2);
			moveWindow("Player 1 - Draw", 0, 0);
			moveWindow("Player 2 - Draw", 300, 0);
		}

		waitKey(1);

		string leptet = "";

		cin >> leptet;

		if (leptet == "j")
		{
			korszam += 1;
		}
		else if (leptet == "b")
		{
			if (korszam == 1)
				cout << "Lejjebb nem lehet léptetni." << endl;
			else
				korszam = korszam - 1;
		}
		else if (leptet == "e")
			kilep = false;
		
		if (p1 == true && !p2)
		{
			destroyWindow("Player 1 - winner");
			destroyWindow("Player 2 - looser");
		}
		else if (p2 == true && !p1)
		{
			destroyWindow("Player 1 - looser");
			destroyWindow("Player 2 - winner");
		}
		else if (!p2 && !p1)
		{
			destroyWindow("Player 1 - Draw");
			destroyWindow("Player 2 - Draw");
		}
		

	}

	return 0;
}