#include <iostream>
// #include <omp.h>
#include <opencv2/opencv.hpp>

class Detect
{
public:
	Detect();
	~Detect();

	void buildgrid(cv::Mat &im);

public:
	std::vector<cv::Rect> grid;

	cv::Mat im, G, G_th, G_tophat;

private:
	void setFeaturePoints(cv::Mat &im, bool allboxes);
	void getGrids(cv::Mat &im);
	void getgradient(cv::Mat& im, cv::Mat& G, double& mean, double& stddev);

private:
	int step_c, step_r; // for setting feature points
	std::vector<std::vector<cv::Point2i>> regions;
	std::vector<cv::Point2i> points;
	float _threhold;
};