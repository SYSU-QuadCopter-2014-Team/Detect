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
	
	//See outputs
	bool verbose;
	
	MultiDetect md(framesToDetect, ROIPadding, verbose);
	// Perform Object Detection
	while ( getline(listFramesFile, frameName) && nFrames < framesToDetect ){
		
		frameName = frameName;

		frame = imread(frameName, CV_LOAD_IMAGE_COLOR);
		
		if( !frame.data )
		 {
		   printf( " No image data \n " );
		   return -1;
		 }

		 md.buildGrid(frame);
		
		nFrames++;
			
	}
	
	Rect initBBox = md.getInitBBox();
	
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
