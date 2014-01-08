#ifndef IMPRESSIONIST
#define IMPRESSIONIST

#include <iostream>
#include "stdlib.h"
#include "cv.h"
#include "highgui.h"
#include <fstream>

#define BRUSH_LENGTH_1 4 
#define BRUSH_LENGTH_2 10
#define BRUSH_RADIUS_1 1
#define BRUSH_RADIUS_2 1.5
#define FALLOFF 1
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
double bilinearIntensitySample(const Mat& sobelFilteredImage, double x, double y);
void renderStroke(const Mat& originImg, Mat& targetImg,
                const Mat& sobelFilteredImage,
                int centerX, int centerY,Point2d direction);
bool pointInRectangle(const Point2d& point, const Point2d& rectangle1, 
                    const Point2d& rectangle2,
                    const Point2d& rectangle3, const Point2d& rectangle4);
Point2d calDirection(const Mat& gradientX, const Mat& gradientY, double centerX, double centerY);

double operator*(const Point2d& p1, const Point2d& p2);
Point2d operator/(const Point2d& p1, double t);

void renderRectangle(Point2d startPoint, Point2d endPoint, double brushRadius,
                    const Mat& originImg, Mat& targetImg, Vec3b currColor);

void renderCircle(Point2d center, double brushRadius, const Mat& originImg, Mat& targetImg, Vec3b currColor);
#endif
