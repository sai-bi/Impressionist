#ifndef IMPRESSIONIST
#define IMPRESSIONIST

#include <iostream>
#include "stdlib.h"
#include "cv.h"
#include "highgui.h"
#include <fstream>

#define BRUSH_LENGTH_1 4 
#define BRUSH_LENGTH_2 10
#define BRUSH_RADIUS_1 1.5
#define BRUSH_RADIUS_2 2.0
#define FALLOFF 6
#define GRADIENT_RADIUS 10
#define PI 3.1415926
using namespace std;
using namespace cv;
void readParameters(char*);
Mat smoothImage(const Mat& src, Size kernelSize, double sigmaX, double sigmaY);
Mat calIntensityImage(const Mat& src);
double randomBrushLength(int startValue, int endValue);
double randomBrushRadius(double startValue, double endValue);
void sobelFilter(const Mat& src, Mat& gradientX, Mat& gradientY, Mat& gradient);
void clipStroke(const Mat& sobelFilteredImage, int centerX, int centerY, 
                Point2d direction, Point2d& startPoint,  Point2d& endPoint,
                int strokeLength);
#endif
