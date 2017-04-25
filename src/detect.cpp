#include "detect.hpp"
#include <ctime>
#include <algorithm>

#define restore(x) (x + 1) * 2

using namespace std;
using namespace cv;

Detect::Detect()
{}

Detect::~Detect()
{}

void Detect::buildgrid(Mat &img)
{
	im = img;
	
	int init_w = round(im.cols * 0.5);
	int init_h = round(im.rows * 0.5);

	time_t t1, t2;

	double mean, stddev;
	t1 = clock();
	
	getgradient(im, G, mean, stddev);
	_threhold = mean + 2 * stddev;

	threshold(G, G_th, _threhold, 255, THRESH_TOZERO); // 二值去噪

	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
	morphologyEx(G_th, G_tophat, MORPH_TOPHAT, element);// 顶帽运算(是否需要这一步可以试验一下)

	setFeaturePoints(G_tophat, 0);// 对顶帽运算的图像进行采样
	getGrids(im);// 对获取到的轮廓构建bbox
	
	t2 = clock();
	double time = ((double)(t2 - t1) / CLOCKS_PER_SEC);
	cout << "BuildGrid Time: " << time << endl;
	cout << "Create " << grid.size() << " boxes." << endl;
}

void Detect::setFeaturePoints(Mat &im, bool allboxes)
{
	int width = im.cols - 2;
	int height = im.rows - 2;
	step_c = 2, step_r = 2; // only for 426 x 240
	int count_c = 0, count_r = 0;
	Point2i point_tmp;

	Mat contours(Size(round((width - 2) * 0.5), round((height - 2) * 0.5)), CV_8UC1, Scalar(0));

	// printf("In setFeaturePOints!\n");

	for (int i = 2; i < height; i = i + step_r)
	{
		uchar* outData = contours.ptr<uchar>(count_r);
		for (int j = 2; j < width; j = j + step_c)
		{
			//get mask
			for (int k = -1; k < 2; ++k)
			{
				bool stop = false;
				const uchar* inData = im.ptr<uchar>(i + k);
				for (int l = -1; l < 2; ++l)
				{
					
					// printf("i,j,k,l: %d, %d, %d, %d\n", i, j, k, l);
					if (inData[j + l] > 0)
					{
						point_tmp.x = j;
						point_tmp.y = i;
						points.push_back(point_tmp);
						outData[count_c] = 255;
						stop = true;
						break;
					}
				}
				if (stop)
					break;
			}
			count_c++;
			
		}
		count_r++;
		count_c = 0;
		
	}
	// 寻找连通体
	int choice;
	if (allboxes)
		choice = CV_RETR_LIST;
	else
		choice = CV_RETR_EXTERNAL;

	findContours(contours, regions, choice, CV_CHAIN_APPROX_SIMPLE);
}

void Detect::getGrids(Mat &im)
{
	if (regions.size() == 0)
	{
		cout << "No regions!" << endl;
		return;
	}

	Rect box;
// #pragma omp parallel for num_threads(4)
	for (int i = 0; i < regions.size(); ++i)
	{
		int min_x = (regions[i][0].x + 1) * 2;
		int min_y = (regions[i][0].y + 1) * 2;
		int max_x = (regions[i][0].x + 1) * 2;
		int max_y = (regions[i][0].y + 1) * 2;
		for (int j = 1; j < regions[i].size(); ++j)
		{
			min_x = min(min_x, (regions[i][j].x + 1) * 2);
			min_y = min(min_y, (regions[i][j].y + 1) * 2);
			max_x = max(max_x, (regions[i][j].x + 1) * 2);
			max_y = max(max_y, (regions[i][j].y + 1) * 2);
		}
		box.x = min_x; box.y = min_y;
		box.width = max_x - min_x + 1;
		box.height = max_y - min_y + 1;
		// 限制box的大小
		if (box.width >= 10 && box.height >= 10 &&
			box.width <= 0.3 * im.cols && box.height <= 0.3 * im.rows)
			grid.push_back(box);
	}
}

void Detect::getgradient(Mat& im, Mat& G, double& mean, double& stddev)
{
	Mat gray;
	if (im.channels() == 3)
		cvtColor(im, gray, CV_RGB2GRAY);
	else
		gray = im;

	// 求得x和y方向的一阶微分
	Mat sobelx;
	Mat sobely;
	Sobel(gray, sobelx, CV_32F, 1, 0, 3);
	Sobel(gray, sobely, CV_32F, 0, 1, 3);

	// 求得梯度和方向
	Mat norm;
	Mat dir;
	cartToPolar(sobelx, sobely, norm, dir);

	// 转换为8位单通道图像进行显示
	double normMax;
	minMaxLoc(norm, NULL, &normMax);
	norm.convertTo(G, CV_8UC1, 255.0 / normMax, 0);

	//计算梯度图的均值和方差
	Mat G_tmp = G.clone();
	Mat m, s;
	meanStdDev(G_tmp, m, s);
	mean = m.at<double>(0, 0);
	stddev = s.at<double>(0, 0);

	gray.release();
	sobelx.release();
	sobely.release();
	norm.release();
	dir.release();
	m.release();
	s.release();
}