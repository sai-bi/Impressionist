/**
 * @author Bi Sai
 * @version 2014/01/02
 */



#include "impressionist.h"


/**
 * @param src source images
 * @param kernelSize Gaussian kernel size
 * @param sigmaX Gaussian kernel standard deviation in X direction
 * @param sigmaY Gaussian kernel standard deviation in Y direction
 * @return Mat the images after Gaussian smooth
 * Smoothing images with Gaussian.
 */
Mat smoothImage(const Mat& src, Size kernelSize, double sigmaX, double sigmaY){
    Mat result = src.clone();
    GaussianBlur(src,result,kernelSize,sigmaX,sigmaY);
    return result;
}

/**
 * Generate random number in range [start, end]
 * @param startValue
 * @param endValue
 * @return a random double number
 */
double randomBrushRadius(){
    double startValue = BRUSH_RADIUS_1;
    double endValue = BRUSH_RADIUS_2; 
    startValue = int(startValue * 10);
    endValue = int(endValue * 10);
    int range = endValue - startValue;
    srand(time(NULL));
    int randNum = rand() % range;

    return (double(randNum) / 10+startValue);
}

/**
 * Generate random number in range[start, end]
 * @param startValue
 * @param endValue
 * @return a random integer
 */
int randomBrushLength(){
    int startValue = BRUSH_LENGTH_1;
    int endValue = BRUSH_LENGTH_2; 
    int range = endValue - startValue;
    srand(time(NULL));

    return(rand()%range + startValue);
}
/**
 * Calculate the intensity image of the given RGB image, the intensity of a pixel
 * is calculated as (30*r + 59*g + 11*b)/100;
 * @param src Input image in RGB color space.
 * @return the intensity image
 */
Mat calIntensityImage(const Mat& src){
    Mat intensity = Mat::zeros(src.rows,src.cols,CV_8U);
    for(int i = 0;i < src.rows;i++){
        for(int j = 0;j < src.cols;j++){
            Vec3b pixel = src.at<Vec3b>(i,j);
            double b = pixel.val[0];
            double g = pixel.val[1];
            double r = pixel.val[2];
            intensity.at<uchar>(i,j) = (uchar)((30*r + 59*g + 11*b)/100.0);
        }
    }
    return intensity;
}

/**
 * Calculate the Gradient of each pixel using sobel filter.
 * @param src input image
 * @param gradientX graident in X direction
 * @param gradientY graident in Y direction
 * @param gradient total gradient
 */
void sobelFilter(const Mat& src, Mat& gradientX, Mat& gradientY, Mat& gradient){
    int scale = 1;
    int ddepth = CV_16S;
    int delta = 0;
    // Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    Sobel(src,gradientX,ddepth,1,0,3,scale,delta,BORDER_DEFAULT);
    Sobel(src,gradientY,ddepth,0,1,3,scale,delta,BORDER_DEFAULT);

    // calculate the power of each matrix
    Mat temp1;
    Mat temp2;
    cv::pow(gradientX,2.0,temp1);
    cv::pow(gradientY,2.0,temp2);

    addWeighted(temp1,1,temp2,1,0,gradient);
}


/**
 * Check if a point lies in a given rectangle, the algorithm is: a point M lies
 * in a rectangle ABCD iff (0<AM*AB<AB*AB) and (0<AM*AD<AD*AD);
 */
bool pointInRectangle(const Point2d& point, const Point2d& rectangle1, const Point2d& rectangle2,
                    const Point2d& rectangle3, const Point2d& rectangle4){
    double temp1 = (point - rectangle1) * (rectangle2 - rectangle1);
    double temp2 = (rectangle2 - rectangle1) * (rectangle2 - rectangle1);
    if(temp1 > temp2 || temp1 < 0)
        return false;
    double temp3 = (point - rectangle1) * (rectangle4 - rectangle1);
    double temp4 = (rectangle4 - rectangle1) * (rectangle4 - rectangle1);
    if(temp3 < 0 || temp3 > temp4)
        return false;

    return true;
}

/**
 * Operator overload for multiply two points.
 */
double operator*(const Point2d& p1, const Point2d& p2){
    return (p1.x * p2.x +  p1.y * p2.y);
}


Point2d operator/(const Point2d& p1, double t){
    return Point2d(p1.x/t,p1.y/t);
}
/**
 * Find points with maximum gradient, use them as pivot points, and 
 * do interpolation for other points to get their graident so as to 
 * get a smooth result.
 */
void calDirectionWithTPS(const Mat& gradientX, const Mat& gradientY, 
        const vector<vector<Point2d> >& region, Mat& interPolationGX, 
        Mat& interPolationGY, vector<vector<Point2d> >& selectedRegionPivot){
    for(int i = 0;i < region.size();i++){
        vector<Point2d> currRegion = region[i];
        // a pair, first is gradient value, second is the position of the point
        vector<pair<double, int> > gradientPointPair;
        // vector<myPair> gradientPointPair;
        for(int j = 0;j < currRegion.size();j++){
            pair<double, int> temp;
            // myPair temp;
            double temp1 = gradientX.at<short int>(currRegion[j].y,currRegion[j].x);
            double temp2 = gradientY.at<short int>(currRegion[j].y,currRegion[j].x);
            temp.first = 0 - (temp1 * temp1 + temp2 * temp2);
            temp.second = j;
            // temp.value = temp1 * temp1 + temp2 * temp2;
            // temp.index = j;
            gradientPointPair.push_back(temp);
        }
        //sort the pair vector
        cout<<gradientPointPair.size()<<endl;
        std::sort(gradientPointPair.begin(),gradientPointPair.end());
        cout<<"sort finished"<<endl;
        // mySort(gradientPointPair);
        // select pivot points to calculate the gradient of other points
        // the pivot points are ones with maximum gradient  
        vector<Point2d> selectedPivot;
        int selectedCount = 0;
        for(int j = 0;j < gradientPointPair.size();j++){
            if(selectedCount > FIXEDPIVOTNUM)
                break;
            // Point2d currPoint = gradientPointPair[j].second;
            Point2d currPoint = currRegion[gradientPointPair[j].second];
            bool flag = true;
            //every pivot points should not be closer than a threhold MINPIVOTDIST
            for(int k = 0;k < selectedPivot.size();k++){
                if(norm(currPoint - selectedPivot[k]) < MINPIVOTDIST){
                    flag = false;
                    break;  
                }
            } 
            if(flag){
                selectedPivot.push_back(currPoint);
                selectedCount++;
            }
        }

        selectedRegionPivot.push_back(selectedPivot); 
        // now do interpolation
        for(int j = 0;j < currRegion.size();j++){
            Point2d currPoint = currRegion[j];
            vector<double> dist;
            double distSum = 0;
            bool flag = false; 
            for(int k = 0;k < selectedPivot.size();k++){
                double temp = norm(currPoint - selectedPivot[k]);
                // for pivot points, set gradient directly
                if(temp < 1e-6){
                    interPolationGX.at<double>(currPoint.y,currPoint.x) = gradientX.at<short int>(currPoint.y,currPoint.x);
                    interPolationGY.at<double>(currPoint.y,currPoint.x) = gradientY.at<short int>(currPoint.y,currPoint.x);
                    flag = true;
                    break;
                }
                distSum += 1.0/temp;
                dist.push_back(1.0/temp);
            }
            if(flag)
                continue;
            double gx = 0;
            double gy = 0;
            for(int k = 0;k < dist.size();k++){
                double temp1 = gradientX.at<short int>(selectedPivot[k].y,selectedPivot[k].x);
                double temp2 = gradientY.at<short int>(selectedPivot[k].y,selectedPivot[k].x);
                gx = gx + temp1 * dist[k]/distSum;
                gy = gy + temp2 * dist[k]/distSum;    
            } 
            interPolationGX.at<double>(currPoint.y,currPoint.x) = gx;
            interPolationGY.at<double>(currPoint.y,currPoint.x) = gy; 
        }
    }
}

void mySort(vector<pair<double,Point2d> >& mypair){
    for(int i = 0;i < mypair.size();i++){
        double minValue = mypair[i].first;
        int minIndex = i;
        for(int j = i+1;j < mypair.size();j++){
            if(mypair[j].first > minValue){
                minValue = mypair[j].first;
                minIndex = j;
            }
        }
        pair<double,Point2d> temp;
        temp = mypair[i];
        mypair[i] = mypair[minIndex];
        mypair[minIndex] = temp;
    }
}




void readRegionInfo(vector<vector<Point2d> >& region, Mat& regionLabel){
    ifstream fin;
    fin.open("./385028.txt");
    int temp;
    vector<Point2d> a[50];  
    int count = -1;
    for(int i = 0;i < regionLabel.rows;i++){
        for(int j = 0;j < regionLabel.cols;j++){
            fin>>temp;
            regionLabel.at<int>(i,j) = temp;
            a[temp-1].push_back(Point2d(j,i));
            if(temp > count) 
               count = temp; 
        }
    }  

    for(int i = 0;i < count;i++){
       region.push_back(a[i]); 
    }
     
}

// void renderImage(){
Mat renderImage(const Mat& img, const Mat& regionLabel, const vector<vector<Point2d> >& region){
      
    // Mat img = imread("./385028.jpg");
    // Gaussian Blur
    cout<<"Image blur..."<<endl;
    Mat blurImage = smoothImage(img,cv::Size(21,21),0,0);
    imshow("Blur Image",blurImage); 

    //read region information 
    // cout<<"Read region info..."<<endl;
    // vector<vector<Point2d> > region;
    // Mat regionLabel = Mat::zeros(img.rows,img.cols,CV_32SC1);
    // readRegionInfo(region,regionLabel);

    
    

    Mat gradientX;
    Mat gradientY;
    Mat targetImg = Mat::zeros(img.rows,img.cols,CV_8UC3);
    Mat sobelFilteredImage; 
    //calculate intensity image
    cout<<"Calculate intensity image..."<<endl; 
    Mat intensity = calIntensityImage(blurImage);
    //smooth intensity image
    intensity = smoothImage(img,cv::Size(21,21),0,0);

    sobelFilter(intensity,gradientX,gradientY,sobelFilteredImage);
    
    Mat interPolationGX = Mat::zeros(gradientX.rows,gradientX.cols,CV_64FC1);
    Mat interPolationGY = Mat::zeros(gradientX.rows,gradientY.cols,CV_64FC1);
    
    vector<vector<Point2d> > selectedRegionPivot;
    //calculate the gradient of each point
    cout<<"Calculate gradient..."<<endl;
    calDirectionWithTPS(gradientX,gradientY,region,interPolationGX,interPolationGY,
            selectedRegionPivot);

    // render the image region by region
    cout<<"Render image region by region..."<<endl;
    
    // record whether a pivot has been covered or not, if not, draw brush there
    // otherwise, do nothing
    Mat hasDrawn = Mat::zeros(img.rows,img.cols,CV_32SC1);
    for(int i = 0;i < region.size();i++){
        vector<Point2d> currRegion = region[i];
        vector<Point2d> currPivot = selectedRegionPivot[i];
        cout<<"Pivot size: "<<currRegion.size()<<" "<<currPivot.size()<<endl;
        for(int j = 0;j < currPivot.size();j++){
            // randomy number of brushes 
            // if(hasDrawn.at<int>(currPivot[j].y,currPivot[j].x) != 0){
                // continue;
            // }

            int brushNum = rand() % (MAXBRUSHNUM - MINBRUSHNUM) + MINBRUSHNUM;
            vector<Point3d> brushPoint;
            cout<<"Get brush points..."<<endl;
            getBrushPoints(img,regionLabel,interPolationGX,interPolationGY,currPivot[j],brushNum,brushPoint);
            cout<<"Get brush points finished..."<<endl; 
            // paint the brushes
            int r = 0;
            int g = 0;
            int b = 0;
            int count = 0; 
            
            // pivot label
            int pivotLabel = regionLabel.at<int>(currPivot[j].y,currPivot[j].x);
            // get average color
            if(brushPoint.size() == 0)
                continue;
            for(int k = 0;k < brushPoint.size();k++){
                Point2d currPoint(brushPoint[k].x,brushPoint[k].y); 
                int label = regionLabel.at<int>(currPoint.y,currPoint.x);
                if(label != pivotLabel && label != pivotLabel + 1000000)
                    continue;
                if(checkPointValid(currPoint,img)){
                    Vec3b color = img.at<Vec3b>(currPoint.y,currPoint.x);
                    r = r + (int)color.val[2];
                    g = g + (int)color.val[1];
                    b = b + (int)color.val[0];
                    count++;
                }
            }
            if(count == 0)
                continue;
            r = r/count;
            g = g/count;
            b = b/count;
        
            for(int k = 0;k < brushPoint.size();k++){

                int x = brushPoint[k].x;
                int y = brushPoint[k].y;
                int label = regionLabel.at<int>(y,x);
                if(label != pivotLabel)
                    continue;
                if(!checkPointValid(Point2d(x,y),img)){
                    continue;
                }
                double alpha = brushPoint[k].z;
                int backgroundR = targetImg.at<Vec3b>(y,x).val[2];
                int backgroundG = targetImg.at<Vec3b>(y,x).val[1];
                int backgroundB = targetImg.at<Vec3b>(y,x).val[0];
                targetImg.at<Vec3b>(y,x).val[0] = (uchar)(int)( (1-alpha)*b + alpha * backgroundB); 
                targetImg.at<Vec3b>(y,x).val[1] = (uchar)(int)((1-alpha)*g + alpha * backgroundG);
                targetImg.at<Vec3b>(y,x).val[2] = (uchar)(int)((1-alpha)*r + alpha * backgroundR);

                // hasDrawn.at<int>(y,x) = 1;
            }  
            imshow("Target Image", targetImg);
            waitKey(5);
        }
    }

    waitKey(0);

    return targetImg; 
      
}

void getBrushPoints(const Mat& originImg, const Mat& regionLabel,const Mat& gradientX, const Mat& gradientY,
        Point2d pivot, int brushNum, vector<Point3d>& brushPoint){
    Point2d currPoint = pivot;
    Point2d currDirection;
    int num = 0;
    int label = regionLabel.at<int>((int)(currPoint.y),(int)(currPoint.x));
    while(num < brushNum){
        currDirection.x = gradientY.at<double>(currPoint.y,currPoint.x);
        currDirection.y = 0 - gradientX.at<double>(currPoint.y,currPoint.x);
        if(norm(currDirection) == 0)
            return;
        currDirection = currDirection / norm(currDirection);
        // cout<<currDirection.x<<" "<<currDirection.y<<endl;
        Point2d endPoint = currPoint;
        int maxLength = randomBrushLength();
        while(true){
            endPoint = endPoint + currDirection;
            // check label, if different, it means that the brush exceeds the region boundary
            if(!checkPointValid(endPoint,originImg)){
                // cout<<"invalid point exit..."<<endl;
                endPoint = endPoint - currDirection;
                break;
            }
            int currLabel = regionLabel.at<int>((int)(endPoint.y),(int)(endPoint.x));
            if(currLabel != label){
                break;
            }
            if(cv::norm(currPoint-endPoint) > maxLength){
                break;
            }
        }
        // if(currPoint.x == endPoint.x && currPoint.y == endPoint.y)
            // return;  
        // Point3d: (x,y,alpha)
        double radius = randomBrushRadius();
        // cout<<currPoint.x<<" "<<currPoint.y << " "<<endPoint.x<<" "<<endPoint.y<<endl;
        cout<<"Start rendering..."<<endl;
        renderRectangle(currPoint,endPoint,brushPoint,radius); 
        cout<<"End rendering..."<<endl;
        renderCircle(currPoint,brushPoint,radius);
        renderCircle(endPoint,brushPoint,radius);
         
        // render next brush
        currPoint = endPoint;
        num++;
    }
    
}

void renderCircle(Point2d center, vector<Point3d>& brush, double radius){
    int minimumX = center.x - radius - FALLOFF;
    int maximumX = center.x + radius + FALLOFF;
    int minimumY = center.y - radius - FALLOFF;
    int maximumY = center.y + radius + FALLOFF;

    for(int i = minimumX - 1;i <= maximumX + 1;i++){
        for(int j = minimumY -1;j<= maximumY+1;j++){
            double dist = norm(Point2d(i-center.x,j-center.y));
            if(dist <= radius){
                brush.push_back(Point3d(i,j,0));
            }  
            else if(dist <= radius + FALLOFF){
                double alpha = (dist - radius) / FALLOFF;
                brush.push_back(Point3d(i,j,alpha));
            }
        }
    }
}


/*
 * Render a rectangle.
 */
void renderRectangle(Point2d startPoint, Point2d endPoint, vector<Point3d>& brush, 
        double radius){
    Point2d direction = endPoint - startPoint;
    if(norm(direction) < 1e-6)
        return;    
    direction = direction / norm(direction);
    Point2d perpendicular(direction.y, -1 * direction.x);
    perpendicular = perpendicular / cv::norm(perpendicular);
    // find points in triangle
    Point2d rectangleVertex1 = startPoint - perpendicular * radius;
    Point2d rectangleVertex2 = startPoint + perpendicular * radius;
    Point2d rectangleVertex3 = endPoint + perpendicular * radius;
    Point2d rectangleVertex4 = endPoint - perpendicular * radius;
    
    Point2d outerRectangleVertex1 = startPoint - perpendicular * (radius + FALLOFF);
    Point2d outerRectangleVertex2 = startPoint + perpendicular * (radius + FALLOFF);
    Point2d outerRectangleVertex3 = endPoint + perpendicular * (radius + FALLOFF);
    Point2d outerRectangleVertex4 = endPoint - perpendicular * (radius + FALLOFF);
    
    int minimumX = min(outerRectangleVertex1.x,min(outerRectangleVertex2.x,min(outerRectangleVertex3.x,outerRectangleVertex4.x)));
    int minimumY = min(outerRectangleVertex1.y,min(outerRectangleVertex2.y,min(outerRectangleVertex3.y,outerRectangleVertex4.y)));
    int maximumX = max(outerRectangleVertex1.x,max(outerRectangleVertex2.x,max(outerRectangleVertex3.x,outerRectangleVertex4.x)));
    int maximumY = max(outerRectangleVertex1.y,max(outerRectangleVertex2.y,max(outerRectangleVertex3.y,outerRectangleVertex4.y)));
   
       
    for(int i = minimumX - 1;i < maximumX + 1;i++){
        for(int j = minimumY - 1;j < maximumY + 1;j++){
            Point2d currPoint(i,j);
            // inner rectangle, set alpha to 1
            if(pointInRectangle(currPoint,rectangleVertex1,rectangleVertex2,rectangleVertex3,rectangleVertex4)){
                // cout<<"hello world"<<endl;
                Point3d temp(i,j,0);     
                brush.push_back(temp);
            }
            else if(pointInRectangle(currPoint,outerRectangleVertex1,outerRectangleVertex2,outerRectangleVertex3,outerRectangleVertex4)){
                // cout<<"hello world"<<endl;
                // outer rectangle, calculate alpha value
                Point2d temp(startPoint.x-i,startPoint.y-j);
                double dotProduct = temp.x * perpendicular.x + temp.y * perpendicular.y;
                double dist = abs(dotProduct / norm(perpendicular));
                double alpha = (dist - radius) / FALLOFF;
                brush.push_back(Point3d(i,j,alpha));      
            }
        }
    }
}




bool checkPointValid(const Point2d& p, const Mat& img){
    int x = p.x;
    int y = p.y;
    if(x < 0 || y < 0 || x >= img.cols || y >= img.rows)
        return false;
    return true;
}


bool pairCompare(pair<double,int>& p1, pair<double, int>& p2){    
    if(p1.first < p2.first)
        return false;
    else
        return true;
} 

// bool pairCompare( myPair& p1,  myPair& p2){
    // if(p1.value < p2.value)
        // return false;
    
    // return true;
// }




