#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"
#include "MultiDetect.h"

#include <dirent.h>

using namespace std;
using namespace cv;
//
// void getGrids(Mat &im, vector<vector<Point2i>> &regions, vector<Rect> &grid)
// {
// 	if (regions.size() == 0)
// 	{
// 		cout << "No regions!" << endl;
// 		return;
// 	}
//
// 	Rect box;
// // #pragma omp parallel for num_threads(4)
// 	for (int i = 0; i < regions.size(); ++i)
// 	{
// 		int min_x = (regions[i][0].x + 1);
// 		int min_y = (regions[i][0].y + 1);
// 		int max_x = (regions[i][0].x + 1);
// 		int max_y = (regions[i][0].y + 1);
// 		for (int j = 1; j < regions[i].size(); ++j)
// 		{
// 			min_x = min(min_x, (regions[i][j].x + 1));
// 			min_y = min(min_y, (regions[i][j].y + 1));
// 			max_x = max(max_x, (regions[i][j].x + 1));
// 			max_y = max(max_y, (regions[i][j].y + 1));
// 		}
// 		box.x = min_x; box.y = min_y;
// 		box.width = max_x - min_x + 1;
// 		box.height = max_y - min_y + 1;
// 		// 限制box的大小
// 		if (box.width >= 30 && box.height >= 30 &&
// 			box.width <= 0.3 * im.cols && box.height <= 0.3 * im.rows)
// 			grid.push_back(box);
// 	}
// }

int main(int argc, char* argv[]){

	if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	for(int i = 0; i < argc; i++){
		if ( strcmp (argv[i], "hog") == 0 )
			HOG = true;
		if ( strcmp (argv[i], "fixed_window") == 0 )
			FIXEDWINDOW = true;
		if ( strcmp (argv[i], "singlescale") == 0 )
			MULTISCALE = false;
		if ( strcmp (argv[i], "show") == 0 )
			SILENT = false;
		if ( strcmp (argv[i], "lab") == 0 ){
			LAB = true;
			HOG = true;
		}
		if ( strcmp (argv[i], "gray") == 0 )
			HOG = false;
	}
	
	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	Mat frame;

	// Tracker results
	Rect result;

	// Path to list.txt
	ifstream listFile;
	string fileName = "images.txt";
  	listFile.open(fileName);

  	// Read groundtruth for the 1st frame
  	ifstream groundtruthFile;
	string groundtruth = "region.txt";
  	groundtruthFile.open(groundtruth);
  	string firstLine;
  	getline(groundtruthFile, firstLine);
	groundtruthFile.close();
  	
  	istringstream ss(firstLine);
	
	// Read Images
	ifstream listFramesFile;
	string listFrames = "images.txt";
	listFramesFile.open(listFrames);
	string frameName;


	// Write Results
	ofstream resultsFile;
	string resultsPath = "output.txt";
	resultsFile.open(resultsPath);

	// Frame counter
	int nFrames = 0;
	
	int framesToDetect = 72;
	int ROIPadding = 20;
	
	MultiDetect md(framesToDetect, ROIPadding, true);
	// Perform Object Detection
	while ( getline(listFramesFile, frameName) && nFrames < framesToDetect ){
		
		frameName = frameName;
		
		// cout << frameName << endl;
		
		frame = imread(frameName, CV_LOAD_IMAGE_COLOR);
		
		if( !frame.data )
		 {
		   printf( " No image data \n " );
		   return -1;
		 }
		
		// resize(frame, frame, Size(426, 240));
		//
		// if (nFrames == 0)
		//    firstFrame = frame;
		//
		//
		// // Detect d;
		// d.buildgrid(frame);
		
		 md.buildGrid(frame);
		
		nFrames++;
			
	}
	
// 	// cout << "Reading" << endl;
//
// 	vector<Rect> grids = d.grid;
//
// 	for (int k = 0; k < grids.size(); k++) {
// 		for (int i = grids[k].x; i < grids[k].x + grids[k].width; i++) {
// 			for (int j = grids[k].y; j < grids[k].y + grids[k].height; j++ ) {
// 				// if (j < localMaxi.cols && i < localMaxi.rows && i >= 0 && j >= 0)
// 					localMaxi.at<unsigned char>(j, i)++;
// 				// if (i >= 0 && j >= 0 && i < 512 && j < 512)
// // 					localMaxi[i][j]++;
// 			}
// 		}
// 	}
//
// 	// Mat lmi = Mat(512, 512, CV_32SC1, localMaxi);
// 	//
// 	// normalize(lmi, lmi, 0, 255, NORM_MINMAX, CV_8UC1);
//
// 	normalize(localMaxi, localMaxi, 0, 255, NORM_MINMAX);
//
// 	// for (int i = 0 ; i < 512; i++){
// 	// 	for (int j = 0 ; j< 512; j++)
// 	// 		cout << localMaxi.at<unsigned char>(j,i) << ' ';
// 	// 	cout << endl;
// 	// }
//
// 	threshold( localMaxi, localMaxi, 255 * 0.5, 255, 0 );
//
//
// 	// imshow("After 3 seconds", localMaxi);
// 	// waitKey(0);
//
// 	vector<vector<Point2i>> regions;
// 	vector<Rect> stackedGrids;
//
// 	findContours(localMaxi, regions, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
//
// 	// Mat drawing = Mat::zeros(frame.size(), CV_8UC1);
// 	//     for( int i = 0; i< regions.size(); i++ )
// 	//    {
// 	//      Scalar color = Scalar( 128 );
// 	//      drawContours( drawing, regions, i, color, 2, 8 );
// 	//    }
// 	//
// 	//    imshow("Contours", drawing);
// 	//    waitKey(0);
//
// 	getGrids(frame, regions, stackedGrids);
//
// 	cout << stackedGrids.size() << endl;
//
// 	vector<Mat> ROIs;
//
// 	for (int i = 0; i < stackedGrids.size(); i++) {
// 		// stackedGrids[i] += Size(20, 20);
// 		stackedGrids[i].x -= 20;
// 		stackedGrids[i].y -= 20;
// 		stackedGrids[i].width += 40;
// 		stackedGrids[i].height += 40;
//
// 		if (stackedGrids[i].x < 0)
// 			stackedGrids[i].x = 0;
// 		if (stackedGrids[i].y < 0)
// 			stackedGrids[i].y = 0;
// 		if (stackedGrids[i].x + stackedGrids[i].width > firstFrame.cols)
// 			stackedGrids[i].width = firstFrame.cols - stackedGrids[i].x - 1;
// 		if (stackedGrids[i].y + stackedGrids[i].height > firstFrame.rows)
// 			stackedGrids[i].height = firstFrame.rows - stackedGrids[i].y - 1;
//
// 		cout << "x:" << stackedGrids[i].x <<
// 			" y:" << stackedGrids[i].y <<
// 				" width:" << stackedGrids[i].width <<
// 					" height:" << stackedGrids[i].height << endl;
// 		rectangle(firstFrame,stackedGrids[i], Scalar(255,0,255), 1, 8, 0 );
//
// 		ROIs.push_back(firstFrame(stackedGrids[i]));
// 	}
//
// 	imshow("Stacked Grids", firstFrame);
// 	waitKey(0);
//
// 	vector<vector<float>> ROIHogs;
// 	HOGDescriptor hog(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);
//
// 	for (int i = 0; i < ROIs.size(); i++) {
// 		resize(ROIs[i], ROIs[i], Size(64, 64));
//
// 		vector<float> hogDesc;
//
// 		hog.compute(ROIs[i], hogDesc, Size(1,1), Size(0,0));
//
// 		ROIHogs.push_back(hogDesc);
//
// 		cout << "Desc dim" << hogDesc.size() << endl;
//
// 		imshow("ROI", ROIs[i]);
// 		waitKey(0);
// 	}
//
// 	vector<int> detectRes;
// 	int truth = -1;
// 	if (!ROIHogs.empty()){
// 		SvmTest svm = SvmTest(ROIHogs[0].size());
//
// 		for (int i = 0; i < ROIHogs.size(); i++) {
//
// 			int response = svm.test(ROIHogs[i].data());
// 			detectRes.push_back(response);
// 			if (response == 1)
// 				truth = i;
//
// 		}
// 	}
//
// 	if (truth == -1){
// 		cout << "Undetected!" << endl;
// 		return -1;
// 	}
//
// 	imshow("Detected object", ROIs[truth]);
// 	waitKey(0);
//
// 	Rect initBBox = stackedGrids[truth];

	Rect initBBox = md.getInitBBox();

	// resultsFile.close();

	// listFile.close();
	// return 1;
	
	//Reset to first frame
	nFrames = 0;
	listFramesFile.clear();
	listFramesFile.seekg(0);

	while ( getline(listFramesFile, frameName) ){
		frameName = frameName;

		// Read each frame from the list
		frame = imread(frameName, CV_LOAD_IMAGE_COLOR);

		// First frame, give the groundtruth to the tracker
		if (nFrames == 0) {

			// float fc = frame.cols / 426.0;
			// float fr = frame.rows / 240.0;
			//
			// cout << "fc: " << fc << " fr: " << fr << endl;
			//
			// initBBox.x *= fc;
			// initBBox.width *= fc;
			// initBBox.y *= fr;
			// initBBox.height *= fr;

			tracker.init( initBBox, frame );
			rectangle( frame, initBBox, Scalar( 0, 255, 255 ), 1, 8 );
			resultsFile << initBBox.x << "," << initBBox.y << "," << initBBox.width << "," << initBBox.height << endl;
		}
		// Update
		else{
			result = tracker.update(frame);
			rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 1, 8 );
			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
		}

		nFrames++;

		if (!SILENT){
			imshow("Image", frame);
			waitKey(1);
		}
	}
	resultsFile.close();

	listFile.close();

}
