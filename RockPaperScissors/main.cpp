#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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


int main() {
	/*
	Mat img,mask;
	vector<vector<Point> > contours;
	resize(imread("black2.jpg"), img, Size(), 0.1, 0.1);

	GaussianBlur(img, img, Size(5, 5), 0);
	threshold(img, mask, 45, 255, THRESH_BINARY);
	Mat se = getStructuringElement(MORPH_ELLIPSE,Size(7,7));

	dilate(mask, mask, se);
	//imshow("dilate",di);
	cvtColor(mask,mask,COLOR_BGR2GRAY);
	imshow("mask",mask);

	findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(0, 255, 0);
		drawContours(img, contours, i, color, 2);
	}
	imshow("img", img);

	waitKey(0);
	return 0;

	*/
	Mat mask;
	Mat img;

	Mat data(6, 3, CV_32F);  //18 minta; 3 jellemz�

	for (int i = 1; i <= 6; ++i) {

		resize(imread("game/" + to_string(i) + ".jpg"), img, Size(), 0.1, 0.1);

		GaussianBlur(img, img, Size(5, 5), 0);
		threshold(img, mask, 45, 255, THRESH_BINARY);
		Mat se = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));

		dilate(mask, mask, se);
		//imshow("dilate",di);
		cvtColor(mask, mask, COLOR_BGR2GRAY);
		imshow("mask", mask);

		vector<vector<cv::Point>> conts;
		findContours(mask.clone(), conts, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		assert(conts.size() == 1);

		//ha t�bb jellemz�t akarsz, a data r�szt is �rd �t.
		data.at<float>(i - 1, 0) = circularity(conts[0]);  //-1 a k�pek sorsz�moz�sa miatt
		data.at<float>(i - 1, 1) = convexity(conts[0]);
		data.at<float>(i - 1, 2) = cextent2(conts[0]);
		//data.at<float>(i - 1, 0) = diameter(conts[0]);
	}

	cout << data << endl;

	TermCriteria crit(TermCriteria::Type::EPS | TermCriteria::Type::MAX_ITER, 100, 0.001);
	Mat labels;
	kmeans(data, 3, labels, crit, 3, KMEANS_RANDOM_CENTERS);


	//Egyszer� megjelen�t�s
	//for (int i = 1; i <= 18; ++i) {
	//	img = imread("../../kekszek_vegyes/" + to_string(i) + ".png", IMREAD_COLOR);
	//	int lbl = labels.at<int>(i - 1);  //-1 csak a k�pek sorsz�moz�sa miatt
	//	imshow(to_string(lbl), img);
	//	waitKey(0);
	//}


	// Csoportba rendezett megjelen�t�s (most soronk�nt, hogy t�bb f�rjen)
	int counters[] = { 0, 0, 0, };  //sz�moljuk, hogy mely csoportba h�ny mint�t pakol a g�p
	for (int i = 1; i <= 7; ++i) {
		img = imread("game/" + to_string(i) + ".jpg", IMREAD_COLOR);

		int lbl = labels.at<int>(i - 1);  //-1 csak a k�pek sorsz�moz�sa miatt

		string winname = to_string(lbl) + to_string(counters[lbl]);
		namedWindow(winname, WINDOW_NORMAL);  //resize miatt
		imshow(winname, img);
		resizeWindow(winname, Size(150, 150)); //csak hogy jobban kiferjen
		moveWindow(winname, counters[lbl] * 150, lbl * 200);
		counters[lbl]++;
	}
	waitKey(0);
	return 0;
}