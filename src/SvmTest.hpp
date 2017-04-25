#include "opencv2/opencv.hpp"
#include <iostream>
#include <cstdio>
using namespace cv;
using namespace cv::ml;
using namespace std;

class SvmTest
{
public:
	Ptr<SVM> model;
	int dim;

	SvmTest() {}

	SvmTest(int dim, const char* modelPath) {
		model = Algorithm::load<SVM>(modelPath);
		this->dim = dim;
	}

	float test(float* sample) {
		Mat sampleMat(1, dim, CV_32FC1, sample);
		float response = model->predict(sampleMat);
		return response;
	}

	~ SvmTest() {}
	
};