#ifndef IMPRESSIONIST
#define IMPRESSIONIST

#include <iostream>
#include "stdlib.h"
#include "cv.h"
#include "highgui.h"
#include <fstream>
#include <utility>
#include <algorithm>
#define BRUSH_LENGTH_1 30 
#define BRUSH_LENGTH_2 50
#define BRUSH_RADIUS_1 1.5
#define BRUSH_RADIUS_2 2.5
#define FALLOFF 5
#define GRADIENT_RADIUS 10
#define PI 3.1415926
#define MAXBRUSHNUM 10
#define MINBRUSHNUM 5
#define FIXEDPIVOTNUM 2000
#define MINPIVOTDIST 10

using namespace std;
using namespace cv;

struct myPair{
    double value;
    int index;
};
void readRegionInfo(vector<vector<Point2d> >& region, Mat& regionLabel);

void readParameters(char*);

Mat smoothImage(const Mat& src, Size kernelSize, double sigmaX, double sigmaY);


double randomBrushRadius();


int randomBrushLength();


Mat calIntensityImage(const Mat& src);



void sobelFilter(const Mat& src, Mat& gradientX, Mat& gradientY, Mat& gradient);


bool pointInRectangle(const Point2d& point, const Point2d& rectangle1, 
        const Point2d& rectangle2,const Point2d& rectangle3, const Point2d& rectangle4);

double operator*(const Point2d& p1, const Point2d& p2);


Point2d operator/(const Point2d& p1, double t);


void calDirectionWithTPS(const Mat& gradientX, const Mat& gradientY, 
        const vector<vector<Point2d> >& region, Mat& interPolationGX, 
        Mat& interPolationGY, vector<vector<Point2d> >& selectedRegionPivot);


void renderImage();


void getBrushPoints(const Mat& originImg, const Mat& gradientX, const Mat& regionLabel, const Mat& graidentY,
        Point2d pivot, int brushNum, vector<Point3d>& brushPoint);


void renderCircle(Point2d center, vector<Point3d>& brush, double radius);


void renderRectangle(Point2d startPoint, Point2d endPoint, vector<Point3d>& brush, 
        double radius);


bool checkPointValid(const Point2d& p, const Mat& img);

bool pairCompare( pair<double,int>& p1,  pair<double, int>& p2);

// bool pairCompare( myPair& p1,  myPair& p2);

void mySort(vector<pair<double,Point2d> >& mypair);


void hsv2rgb(double H, double S, double V, double& R, double &G, double& B);
void rgb2hsv(double R, double G, double B, double &H, double &S, double &V);

#endif
