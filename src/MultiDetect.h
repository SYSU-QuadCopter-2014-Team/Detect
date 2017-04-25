#include <iostream>

#include <opencv2/core/core.hpp>
#include "detect.hpp"
#include "SvmTest.hpp"
using namespace cv;
using namespace std;

class MultiDetect{


public:

	MultiDetect(int framesToVote, int ROIPadding, bool verbose);
	
	void verboseSwitch(bool verbose);
	
	void buildGrid(Mat& frame);
	
	Rect getInitBBox();

private:

	bool verbose;

	int framesToVote;

	int votedFrames;

	int padding;

	Detect d;

	Mat unresized;

	Mat frame;

	Mat localMaxi;

	vector<Rect> grids;
	
	vector<vector<Point2i>> regions;
	
	vector<Rect> stackedGrids;
	
	vector<Mat> ROIs;
	
	vector<vector<float>> ROIHogs;
	
	vector<int> detectRes;
	
	SvmTest svm;
	
private:
	
	void voteForStackedRegion();
	
	void extractROIHogs();

	void getGrids();


};