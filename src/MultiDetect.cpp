#include <iostream>

#include <opencv2/core/core.hpp>
#include "MultiDetect.h"
using namespace cv;
using namespace std;

MultiDetect::MultiDetect(int ftv, int ROIPadding, bool v) {
	
	verbose = v;
	
	d = Detect();
	
	localMaxi = Mat(Size(426, 240), CV_8UC1, cvScalar(0.));
	
	padding = ROIPadding;
	
	framesToVote = v;
	
	votedFrames = 0;
	
}

void MultiDetect::verboseSwitch(bool v) {
	verbose = v;
}

void MultiDetect::buildGrid(Mat& f) {
	
	if (votedFrames == framesToVote) {
		unresized = f.clone();
	}
	
	resize(f, f, Size(426, 240));
	
	if (votedFrames == framesToVote) {
		frame = f.clone();
	}
	
	d.buildgrid(f);
	
	votedFrames++;
	
}

Rect MultiDetect::getInitBBox() {
	
	voteForStackedRegion();
	
	extractROIHogs();
	
	//Default Bounding box, center of image, width and height of 60 pixel.
	Rect initBBox = Rect(
		Point(unresized.cols/2 - 30, unresized.rows/2 - 30), 
		Point(unresized.cols/2 - 30, unresized.rows/2 - 30)
			);
	
	int truth = -1;
	if (!ROIHogs.empty()){
		svm = SvmTest(ROIHogs[0].size(), "./SVMModel.xml");
		
		for (int i = 0; i < ROIHogs.size(); i++) {
			
			int response = svm.test(ROIHogs[i].data());
			detectRes.push_back(response);
			if (response == 1)
				truth = i;
			
		}
	}
	
	if (truth == -1){
		cout << "Undetected!" << endl;
		return initBBox;
	}

	cout << "Object Detected!" << endl;
	if (verbose) {
		imshow("Detected object", ROIs[truth]);
		waitKey(0);
	}

	initBBox = stackedGrids[truth];
	
	float fc = unresized.cols / 426.0;
	float fr = unresized.rows / 240.0;

	cout << "fc: " << fc << " fr: " << fr << endl;

	initBBox.x *= fc;
	initBBox.width *= fc;
	initBBox.y *= fr;
	initBBox.height *= fr;
	
	return initBBox;
	
}

void MultiDetect::voteForStackedRegion() {
	
	grids = d.grid;

	vector<Rect> grids = d.grid;

	for (int k = 0; k < grids.size(); k++) {
		for (int i = grids[k].x; i < grids[k].x + grids[k].width; i++) {
			for (int j = grids[k].y; j < grids[k].y + grids[k].height; j++ ) {

					localMaxi.at<unsigned char>(j, i)++;

			}
		}
	}
	
	normalize(localMaxi, localMaxi, 0, 255, NORM_MINMAX);
	
	threshold( localMaxi, localMaxi, 255 * 0.5, 255, 0 );
	
	if(verbose){
		
		imshow("After 3 seconds", localMaxi);
		waitKey(0);
	}

	
	findContours(localMaxi, regions, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	
	if(verbose) {
		Mat drawing = Mat::zeros(frame.size(), CV_8UC1);
	   for ( int i = 0; i< regions.size(); i++ ) {
	     Scalar color = Scalar( 128 );
	     drawContours( drawing, regions, i, color, 2, 8 );
	   }

	   imshow("Contours", drawing);
	   waitKey(0);
	}
	
	getGrids();
}

void MultiDetect::extractROIHogs() {
	
	
	for (int i = 0; i < stackedGrids.size(); i++) {

		stackedGrids[i].x -= padding;
		stackedGrids[i].y -= padding;
		stackedGrids[i].width += (padding * 2);
		stackedGrids[i].height += (padding * 2);
		
		if (stackedGrids[i].x < 0)
			stackedGrids[i].x = 0;
		if (stackedGrids[i].y < 0)
			stackedGrids[i].y = 0;
		if (stackedGrids[i].x + stackedGrids[i].width > frame.cols)
			stackedGrids[i].width = frame.cols - stackedGrids[i].x - 1;
		if (stackedGrids[i].y + stackedGrids[i].height > frame.rows)
			stackedGrids[i].height = frame.rows - stackedGrids[i].y - 1;
		
		if (verbose)
			cout << "x:" << stackedGrids[i].x <<
				" y:" << stackedGrids[i].y <<
					" width:" << stackedGrids[i].width <<
						" height:" << stackedGrids[i].height << endl;
		rectangle(frame,stackedGrids[i], Scalar(255,0,255), 1, 8, 0 );
		
		ROIs.push_back(frame(stackedGrids[i]));
	}
	
	HOGDescriptor hog(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);    
	
	for (int i = 0; i < ROIs.size(); i++) {
		resize(ROIs[i], ROIs[i], Size(64, 64));
		
		vector<float> hogDesc;
		
		hog.compute(ROIs[i], hogDesc, Size(1,1), Size(0,0));
		
		ROIHogs.push_back(hogDesc);
		
		cout << "Desc dim" << hogDesc.size() << endl;
		
		if (verbose) {
			imshow("ROI", ROIs[i]);
			waitKey(0);
		}

	}
	
}

void MultiDetect::getGrids()
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
		int min_x = (regions[i][0].x + 1);
		int min_y = (regions[i][0].y + 1);
		int max_x = (regions[i][0].x + 1);
		int max_y = (regions[i][0].y + 1);
		for (int j = 1; j < regions[i].size(); ++j)
		{
			min_x = min(min_x, (regions[i][j].x + 1));
			min_y = min(min_y, (regions[i][j].y + 1));
			max_x = max(max_x, (regions[i][j].x + 1));
			max_y = max(max_y, (regions[i][j].y + 1));
		}
		box.x = min_x; box.y = min_y;
		box.width = max_x - min_x + 1;
		box.height = max_y - min_y + 1;
		// 限制box的大小
		if (box.width >= 30 && box.height >= 30 &&
			box.width <= 0.3 * frame.cols && box.height <= 0.3 * frame.rows)
			stackedGrids.push_back(box);
	}
	
}


