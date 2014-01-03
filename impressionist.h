#ifndef IMPRESSIONIST
#define IMPRESSIONIST

#include <iostream>
#include "stdlib.h"
#include "cv.h"
#include "highgui.h"
#include <fstream>
using namespace std;
using namespace cv;
void readParameters(char*);
Mat smoothImage(const Mat& src, Size kernelSize, double sigmaX, double sigmaY);
Mat calIntensityImage(const Mat& src);
double randomBrushLength(int startValue, int endValue);
double randomBrushRadius(double startValue, double endValue);
void sobelFilter(const Mat& src, Mat& gradientX, Mat& gradientY, Mat& gradient);

#endif
