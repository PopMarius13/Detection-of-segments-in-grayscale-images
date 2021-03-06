// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <cmath>
#include <queue>

struct houghValue {
	float val;
	int p;
	int q;
};

bool isInside(Mat img, int i, int j) {
	if (i >= 0 && i < img.rows) {
		if (j >= 0 && j < img.cols)
		{
			return true;
		}
	}
	return false;

}

Mat nucleuGaussian(int w) {
	int k = w / 2;
	float pi = (float)w / 6.0f;

	Mat nucleu(w, w, CV_32FC1);
	/*for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			float exponent = (pow((j - k), 2) + pow((i - k), 2)) / (2 * pow(pi, 2));
			float val = 1.0f / (PI * 2 * pow(pi, 2));
			nucleu.at<float>(i, j) = (float)exp(-exponent) * val;
			std::cout << nucleu.at<float>(i, j) << " ";
		}
		std::cout << "\n";
	}*/
	nucleu.at<float>(0, 0) = 0.0005;
	nucleu.at<float>(0, 1) = 0.0050;
	nucleu.at<float>(0, 2) = 0.0109;
	nucleu.at<float>(0, 3) = 0.0050;
	nucleu.at<float>(0, 4) = 0.0005;
	nucleu.at<float>(1, 0) = 0.0050;
	nucleu.at<float>(1, 1) = 0.0521;
	nucleu.at<float>(1, 2) = 0.1139;
	nucleu.at<float>(1, 3) = 0.0521;
	nucleu.at<float>(1, 4) = 0.0050;
	nucleu.at<float>(2, 0) = 0.0109;
	nucleu.at<float>(2, 1) = 0.1139;
	nucleu.at<float>(2, 2) = 0.2487;
	nucleu.at<float>(2, 3) = 0.1139;
	nucleu.at<float>(2, 4) = 0.0109;
	nucleu.at<float>(3, 0) = 0.0050;
	nucleu.at<float>(3, 1) = 0.0521;
	nucleu.at<float>(3, 2) = 0.1139;
	nucleu.at<float>(3, 3) = 0.0521;
	nucleu.at<float>(3, 4) = 0.0050;
	nucleu.at<float>(4, 0) = 0.0005;
	nucleu.at<float>(4, 1) = 0.0050;
	nucleu.at<float>(4, 2) = 0.0109;
	nucleu.at<float>(4, 3) = 0.0050;
	nucleu.at<float>(4, 4) = 0.0005;

	return nucleu;
}

float convolutieTreceJos(Mat src, int x, int y, Mat ng, int w, float c) {
	float sum = 0;
	int k = w / 2;
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			sum = sum + ng.at<float>(i, j) * src.at<uchar>(x + i - k, y + j - k);
		}
	}
	return sum / c;
}

Mat filterGaussian(Mat img, int w) {
	Mat fg = img.clone();
	Mat ng = nucleuGaussian(w);

	float c = 0;
	int k = w / 2;
	for (int i = 0; i < w; i++)
		for (int j = 0; j < w; j++)
			c += ng.at<float>(i, j);

	for (int i = k; i < fg.rows - k; i++) {
		for (int j = k; j < fg.cols - k; j++) {
			fg.at<uchar>(i, j) = (uchar)convolutieTreceJos(img, i, j, ng, w, c);
		}
	}

	return fg;
}

void filtruGaussianTest() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		Mat fg = filterGaussian(src, 5);

		imshow("input image", src);
		imshow("gauss image", fg);
		waitKey();
	}
}

float valConvTS(Mat H, int x, int y, int k, Mat img) {
	float val = 0;
	for (int u = 0; u < H.rows; u++) {
		for (int v = 0; v < H.cols; v++) {
			val += (float)H.at<float>(u, v)* img.at<uchar>(x + u - k, y + v - k);
		}
	}
	return (float)val;
}

Mat convolutieTreceSus(Mat img, Mat H) {
	Mat dst(img.rows, img.cols, CV_32FC1);
	dst.setTo(0);

	int k = H.rows / 2;

	for (int i = k; i < img.rows - k; i++) {
		for (int j = k; j < img.cols - k; j++) {
			dst.at<float>(i, j) = (float)valConvTS(H, i, j, k, img);
		}
	}
	return dst;
}


void convTreceSusTest() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<float> H(3, 3);
		H(0, 0) = 0;
		H(0, 1) = -1;
		H(0, 2) = 0;
		H(1, 0) = -1;
		H(1, 1) = 4;
		H(1, 2) = -1;
		H(2, 0) = 0;
		H(2, 1) = -1;
		H(2, 2) = 0;

		Mat fg = convolutieTreceSus(src, H);

		imshow("input image", src);
		imshow("conv trece sus image", fg/255);
		waitKey();
	}
}

int* histogram(Mat img) {

	int* hist = (int*)malloc(256 * sizeof(int));

	for (int i = 0; i < 256; i++) {
		hist[i] = 0;
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int poz = img.at<uchar>(i, j);
			hist[poz] += 1;
		}
	}
	return hist;
}

Mat sobel(Mat img) {
	Mat ffx(img.rows, img.cols, CV_32FC1);
	Mat ffy(img.rows, img.cols, CV_32FC1);
	Mat panta = Mat::zeros(img.rows, img.cols, CV_32FC1);
	Mat directie = Mat::zeros(img.rows, img.cols, CV_32FC1);

	Mat_<float> fx(3, 3);
	fx(0, 0) = -1;
	fx(0, 1) = 0;
	fx(0, 2) = 1;
	fx(1, 0) = -2;
	fx(1, 1) = 0;
	fx(1, 2) = 2;
	fx(2, 0) = -1;
	fx(2, 1) = 0;
	fx(2, 2) = 1;

	Mat_<float> fy(3, 3);
	fy(0, 0) = 1;
	fy(0, 1) = 2;
	fy(0, 2) = 1;
	fy(1, 0) = 0;
	fy(1, 1) = 0;
	fy(1, 2) = 0;
	fy(2, 0) = -1;
	fy(2, 1) = -2;
	fy(2, 2) = -1;


	ffx = convolutieTreceSus(img, fx);
	ffy = convolutieTreceSus(img, fy);
	Mat panta_print = Mat::zeros(img.rows, img.cols, CV_8UC1);

	for (int i = 2; i < panta.rows - 2; i++) {
		for (int j = 2; j < panta.cols - 2; j++) {
			panta.at<float>(i, j) = (float)sqrt((float)ffx.at<float>(i, j) * ffx.at<float>(i, j) + (float)ffy.at<float>(i, j) * ffy.at<float>(i, j));
			panta_print.at<uchar>(i, j) = panta.at<float>(i, j) / (4 * sqrt(2));

			directie.at<float>(i, j) = (float)atan2(ffy.at<float>(i, j), ffx.at<float>(i, j));
			if (directie.at<float>(i, j) < 0) {
				directie.at<float>(i, j) += 2 * PI;
			}
			//directie.at<float>(i, j) *= 180 / PI;
		}
	}

	imshow("Magnitudine", panta_print);
	//Mat clone_panta = panta_print.clone();
	
	/*for (int i = 1; i < panta_print.rows - 1; i++) {
		for (int j = 1; j < panta_print.cols - 1; j++) {
			if ((directie.at<float>(i, j) > (3 * PI / 8) && directie.at<float>(i, j) < (5 * PI / 8)) ||
				(directie.at<float>(i, j) > (11 * PI / 8) && directie.at<float>(i, j) < (13 * PI / 8) ))
				if (clone_panta.at<uchar>(i, j) <= clone_panta.at<uchar>(i + 1, j) || clone_panta.at<uchar>(i, j) <= clone_panta.at<uchar>(i - 1, j)) {
					panta_print.at<uchar>(i, j) = 0;
				}
			if ((directie.at<float>(i, j) > PI/8 && directie.at<float>(i, j) <= (3*PI / 8)) ||
				(directie.at<float>(i, j) > (9 * PI / 8) && directie.at<float>(i, j) <= (11*PI / 8)))
				if (clone_panta.at<uchar>(i, j) <= clone_panta.at<uchar>(i - 1, j + 1) || clone_panta.at<uchar>(i, j) <= clone_panta.at<uchar>(i + 1, j - 1)) {
					panta_print.at<uchar>(i, j) = 0;
				}
			if ((directie.at<float>(i, j) > (7 * PI / 8) && directie.at<float>(i, j) <= (9 * PI / 8)) ||
				(directie.at<float>(i, j) > (15 * PI / 8) || directie.at<float>(i, j) <= (PI / 8)))
				if (clone_panta.at<uchar>(i, j) <= clone_panta.at<uchar>(i, j - 1) || clone_panta.at<uchar>(i, j) <= clone_panta.at<uchar>(i, j + 1)) {
					panta_print.at<uchar>(i, j) = 0;
				}
			if ((directie.at<float>(i, j) >= (5 * PI / 8) && directie.at<float>(i, j) <= (7*PI / 8)) ||
				(directie.at<float>(i, j) >= (13* PI / 8) && directie.at<float>(i, j) <= (15 * PI / 8)))
				if (clone_panta.at<uchar>(i, j) <= clone_panta.at<uchar>(i - 1, j - 1) || clone_panta.at<uchar>(i, j) <= clone_panta.at<uchar>(i + 1, j + 1)) {
					panta_print.at<uchar>(i, j) = 0;
				}
		}
	}*/
	Mat panta_clone = panta.clone();

	for (int i = 2; i < panta.rows - 2; i++) {
		for (int j = 2; j < panta.cols - 2; j++) {
			if ((directie.at<float>(i, j) > (3 * PI / 8) && directie.at<float>(i, j) < (5 * PI / 8)) ||
				(directie.at<float>(i, j) > (11 * PI / 8) && directie.at<float>(i, j) < (13 * PI / 8)))
				if (panta.at<float>(i, j) <= panta.at<float>(i + 1, j) || panta.at<float>(i, j) <= panta.at<float>(i - 1, j)) {
					panta_clone.at<float>(i, j) = 0;
				}
			if ((directie.at<float>(i, j) > PI / 8 && directie.at<float>(i, j) <= (3 * PI / 8)) ||
				(directie.at<float>(i, j) > (9 * PI / 8) && directie.at<float>(i, j) <= (11 * PI / 8)))
				if (panta.at<float>(i, j) <= panta.at<float>(i - 1, j + 1) || panta.at<float>(i, j) <= panta.at<float>(i + 1, j - 1)) {
					panta_clone.at<float>(i, j) = 0;
				}
			if ((directie.at<float>(i, j) > (7 * PI / 8) && directie.at<float>(i, j) <= (9 * PI / 8)) ||
				(directie.at<float>(i, j) > (15 * PI / 8) || directie.at<float>(i, j) <= (PI / 8)))
				if (panta.at<float>(i, j) <= panta.at<float>(i, j - 1) || panta.at<float>(i, j) <= panta.at<float>(i, j + 1)) {
					panta_clone.at<float>(i, j) = 0;
				}
			if ((directie.at<float>(i, j) >= (5 * PI / 8) && directie.at<float>(i, j) <= (7 * PI / 8)) ||
				(directie.at<float>(i, j) >= (13 * PI / 8) && directie.at<float>(i, j) <= (15 * PI / 8)))
				if (panta.at<float>(i, j) <= panta.at<float>(i - 1, j - 1) || panta.at<float>(i, j) <= panta.at<float>(i + 1, j + 1)) {
					panta_clone.at<float>(i, j) = 0;
				}
			panta_print.at<uchar>(i, j) = panta_clone.at<float>(i, j) / (4 * sqrt(2));
		}
	}
	
	imshow("Fil Gaus", img);
	imshow("ElimNonMaxime", panta_print);

	int *hist = histogram(panta_print);
	float nrNotM = (float)0.92 * (panta_print.rows * panta_print.cols - hist[0]);
	float th = 0;
	int i = 1;

	while (th < nrNotM && i < 256) {
		th += hist[i++];
	}
	th = i - 1;
	std::cout << th << "  " << 0.4 * th << " " << nrNotM << "\n";
	th = 15;

	for (int i = 0; i < panta_print.rows; i++) {
		for (int j = 0; j < panta_print.cols; j++) {
			if ((int)panta_print.at<uchar>(i, j) < 0.4 * th) {
				panta_print.at<uchar>(i, j) = 0;
			}else if ((int)panta_print.at<uchar>(i, j) >= 0.4 * th && panta_print.at<uchar>(i, j) < th) {
				panta_print.at<uchar>(i, j) = 128;
			}else {
				panta_print.at<uchar>(i, j) = 255;
			}
		}
	}

	imshow("Binarizare", panta_print);

	int di[] = { 1, 1, 1, -1, -1, -1, 0, 0 };
	int dj[] = { -1, 0, 1, -1, 0, 1, 1, -1 };
	std::queue<Point2i> Q;

	for (int i = 0; i < panta_print.rows; i++) {
		for (int j = 0; j < panta_print.cols; j++) {
			if (panta_print.at<uchar>(i, j) == 255) {
				Q.push(Point2i(i, j));
				while (!Q.empty()) {
					Point2i q = Q.front();
					Q.pop();
					for (int i = 0; i < 8; i++) {
						if (isInside(panta_print, q.x + di[i], q.y + dj[i]) && panta_print.at<uchar>(q.x + di[i], q.y + dj[i]) == 128) {
							panta_print.at<uchar>(q.x + di[i], q.y + dj[i]) = 255;
							Q.push(Point2i(q.x + di[i], q.y + dj[i]));
						}
					}
				}
			}
		}
	}
	
	for (int i = 0; i < panta.rows; i++) {
		for (int j = 0; j < panta.cols; j++) {
			if (panta_print.at<uchar>(i, j) == 128) {
				panta_print.at<uchar>(i, j) = 0;
			}
		}
	}

	imshow(" Histereza", panta_print);

	return panta_print;
}


void sobelTest() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		sobel(src);
		waitKey(0);
	}
}


void cannyTest() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat fg = filterGaussian(src, 5);
		imshow("Img Originala", src);
		sobel(fg);
		waitKey(0);
	}
}

boolean checkMaximLocal(int v, Mat src, int x, int y) {

	if (v < 10) {
		return false;
	}
	for (int i = x - 3; i < x + 3; i++) {
		for (int j = y - 3; j < y + 3; j++) {
			if(i >= 0 && j >= 0 && i < src.rows && j < src.cols)
				if (v < src.at<uchar>(i, j)) {
					return false;
				}
		}
	}
	return true;
}

void results(Mat TH, Mat src, Mat res_canny, houghValue a[], int n) {
	Mat result(src.rows, src.cols, CV_8UC3);
	cvtColor(src, result, COLOR_GRAY2BGR);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			for (int k = 0; k < 360; k++) {
				float grade = k;
				if (grade != 0) {
					grade = (float)k * PI / 180;
				}
				int pNew = j * cos(grade) + i * sin(grade);
				for (int l = 0; l < n; l++) {
					if (a[l].p == pNew && k == a[l].q) {
						result.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
						if (res_canny.at<uchar>(i, j) == 255) {
							result.at<Vec3b>(i, j) = Vec3b(255, 0, 255);
						}
					}
				}
			}
		}
	}


	imshow("Result", result);
	waitKey();

}

void transformataHough(Mat src, int n, Mat img) {

	int p = sqrt(src.rows * src.rows + src.cols * src.cols);
	Mat H = Mat::zeros(p, 360, CV_32FC1);
	houghValue* a = new houghValue[p * 360];
	int ia = 0;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 255) {
				for (int k = 0; k < 360; k++) {
					float grade = k;
					if (grade != 0) {
						grade = (float)k * PI / 180;
					}
					int pNew = j * cos(grade) + i * sin(grade);
					if (pNew <= p && pNew >= 0) {
						H.at<float>(pNew, k)++;
					}
				}
			}
		}
	}

	Mat HN;
	normalize(H, HN, 0, 255, NORM_MINMAX, CV_8UC1);
	for (int i = 0; i < HN.rows; i++) {
		for (int j = 0; j < HN.cols; j++) {
			if(checkMaximLocal(HN.at<uchar>(i, j),HN, i, j)) {
				houghValue v = { HN.at<uchar>(i, j), i, j };
				a[ia++] = v;
			}
		}
	}
	for (int i = 0; i < ia; i++) {
		for (int j = i + 1; j < ia; j++) {
			if (a[i].val < a[j].val) {
				houghValue v = a[i];
				a[i] = a[j];
				a[j] = v;
			}
		}
	}


	Mat HNC(HN.rows, HN.cols, CV_8UC3);
	cvtColor(HN, HNC, COLOR_GRAY2BGR);
	for (int i = 0; i < n; i++) {
		std::cout << a[i].val << " " << a[i].p << " " << a[i].q << "\n";
		line(HNC, Point( a[i].q, a[i].p - 2), Point( a[i].q, a[i].p + 2), CV_RGB(255, 0, 0));
		line(HNC, Point( a[i].q - 2, a[i].p), Point(a[i].q + 2, a[i].p), CV_RGB(255, 0, 0));
	}
	imshow("TH", HNC);


	results(HNC, img, src, a, n);

}

void proiect(int n) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat fg = filterGaussian(src, 5);
		Mat res_canny = sobel(fg);
		transformataHough(res_canny, n, src);

	}
}


int main()
{
	int op = 0;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - filtruGaussianTest\n");
		printf(" 2 - convTreceSusTest\n");
		printf(" 3 - sobelTest\n");
		printf(" 4 - CannyTest\n");
		printf(" 5 - TH\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);

		switch (op)
		{
			case 1:
				filtruGaussianTest();
				break;
			case 2:
				convTreceSusTest();
				break;
			case 3:
				sobelTest();
				break;
			case 4:
				cannyTest();
				break;
			case 5:
				std::cout << "N = ";
				int n;
				std::cin >> n;
				proiect(n);
				break;
		}
	}
	while (op!=0);
	return 0;
}