/**
 * @author Bi Sai
 * @version 2014/01/02
 */



#include "impressionist.h"


/**
 * @param fileName name of the text file.
 * Read parameters from a text file, and the text file is in the format of "
 * parameterName value" for each line.
 */
void readParameters(char* fileName){
    ifstream fin;
    fin.open(fileName);
    if(fin.is_open()){
        string parameterName;
        double value;
        while(fin>>parameterName>>value){
            // assign value to corresponding parameters
        }
    }
}

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
double randomBrushRadius(double startValue, double endValue){
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
double randomBrushLength(int startValue, int endValue){
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
 * Clip a stroke so that it doesn't exceed the stroke length, and at the same
 * time it doesn't overlap the edges in images.
 * @param sobelFilteredImage the image after soble filter
 * @param centerX x coordinate of center of stroke
 * @param centerY y coordinate of center of stroke
 * @param direction direction of the stroke
 * @param startPoint the output start point of the stroke
 * @param endPoint the output end point of the stroke
 * @param strokeLength the length of the stroke
 */
void clipStroke(const Mat& sobelFilteredImage, int centerX, int centerY,
                Point2d direction, Point2d& startPoint,  Point2d& endPoint,
                int strokeLength){
    double startX;
    double startY;
    double endX;
    double endY;
    double tempX = centerX;
    double tempY = centerY;

    direction = direction / cv::norm(direction);
    // bilinear sample intensity at (tempX, tempY)
    // cout<<tempX << " "<<tempY
        // <<" "<<sobelFilteredImage.rows
        // <<" "<<sobelFilteredImage.cols<<endl;
    double lastIntensity = bilinearIntensitySample(sobelFilteredImage, tempX, tempY);
    cout<<"hello"<<endl;
    while(true){
        tempX = tempX + direction.x;
        tempY = tempY + direction.y;
        if(tempX < 0 || tempY < 0 || tempX >= sobelFilteredImage.cols
            || tempY >= sobelFilteredImage.rows)
            break;
        if(norm(Point2d(tempX - centerX, tempY - centerY)) > strokeLength/2)
            break;
        double newIntensity = bilinearIntensitySample(sobelFilteredImage,tempX,tempY);
        if(newIntensity < lastIntensity)
            break;
        lastIntensity = newIntensity;
    }
    startX = tempX;
    startY = tempY;

    tempX = centerX;
    tempY = centerY;
    direction = (-1) * direction;
    lastIntensity = bilinearIntensitySample(sobelFilteredImage,tempX,tempY);

    cout<<"hello1"<<endl;
    while(true){
        tempX = tempX + direction.x;
        tempY = tempY + direction.y;
        if(tempX < 0 || tempY < 0 || tempX >= sobelFilteredImage.cols
            || tempY >= sobelFilteredImage.rows)
            break;
        if(norm(Point2d(tempX - centerX, tempY - centerY)) > strokeLength)
            break;
        double newIntensity = bilinearIntensitySample(sobelFilteredImage,tempX,tempY);
        if(newIntensity < lastIntensity)
            break;
        lastIntensity = newIntensity;
    }
    endX = tempX;
    endY = tempY;

    cout<<"hello2"<<endl;
    if(startX < endX){
        startPoint.x = startX;
        startPoint.y = startY;
        endPoint.x = endX;
        endPoint.y = endY;
    }
    else{
        startPoint.x = endX;
        startPoint.y = endY;
        endPoint.x = startX;
        endPoint.y = startY;
    }
}
/**
 * Given a point(x,y), calculate the intensity at that point using bilinear interpolation.
 * @param sobelFilteredImage image after sobel filter
 * @param x
 * @param y
 */
double bilinearIntensitySample(const Mat& sobelFilteredImage, double x, double y){
    cout<<"hello3"<<endl;
    int leftX = int(x);
    int leftY = int(y);
    if(leftX >= sobelFilteredImage.cols-1 || leftY >= sobelFilteredImage.rows-1
            ||leftX < 0 || leftY< 0){
        cout<<"hello4"<<endl;
        return -1000000;
    }
    cout<<leftX << " "<<leftY
        <<" "<<sobelFilteredImage.rows
        <<" "<<sobelFilteredImage.cols<<endl;

    double leftupIntensity = sobelFilteredImage.at<short int>(leftY,leftX);
    double leftdownIntensity = sobelFilteredImage.at<short int>(leftY+1,leftX);
    double rightUpIntensity = sobelFilteredImage.at<short int>(leftY,leftX+1);
    double rightDownIntensity = sobelFilteredImage.at<short int>(leftY+1,leftX+1);

    double temp1 = leftupIntensity * (x-leftX) + rightUpIntensity * (leftX+1-x);
    double temp2 = leftdownIntensity * (x-leftX) + rightDownIntensity *(leftX+1-x);

    double result = temp1 * (y-leftY) + temp2*(leftY+1-y);

    cout<<"hello4"<<endl;
    return result;
}

/**
 * Given the center and direction of a stroke, we render it.
 * @param originImg
 * @param targetImg image after rendering
 * @param sobelFilteredImage
 * @param centerX x coordinate of the stroke center
 * @param centerY y coordinate of the stroke center
 * @param direction direction of the stroke
 */

void renderStroke(const Mat& originImg, Mat& targetImg,
                const Mat& sobelFilteredImage,
                int centerX, int centerY,Point2d direction){
    Point2d startPoint;
    Point2d endPoint;
    int strokeLength = randomBrushLength(BRUSH_LENGTH_1,BRUSH_LENGTH_2);
    clipStroke(sobelFilteredImage,centerX,centerY,direction,startPoint,endPoint,
            strokeLength);

    double brushRadius = randomBrushRadius(BRUSH_RADIUS_1, BRUSH_RADIUS_2)/2;

    //color to be filled
    Vec3b currColor = originImg.at<Vec3b>(centerY,centerX);

    // if(direction.x > 0){
        // startPoint = Point2d(centerX - direction.x * strokeLength, centerY - direction.y * strokeLength);
        // endPoint = Point2d(centerX + direction.x * strokeLength, centerY + direction.y * strokeLength);
    // }
    // else{
        // startPoint = Point2d(centerX + direction.x * strokeLength, centerY + direction.y * strokeLength);
        // endPoint = Point2d(centerX - direction.x * strokeLength, centerY - direction.y * strokeLength);
    // }

    renderRectangle(startPoint,endPoint,brushRadius,originImg,targetImg,currColor);
    // line(targetImg,startPoint,endPoint,Scalar(currColor.val[0],currColor.val[1],currColor.val[2]),
            // 3);
            // imshow("Target - brushRadius)image",targetImg);
            // waitKey(0);

    // line(targetImg,startPoint,endPoint,Scalar(currColor.val[0],currColor.val[1],currColor.val[2]),
            // 3);

    // imshow("Target image",targetImg);
            // waitKey(0);
    renderCircle(startPoint,brushRadius,originImg,targetImg,currColor);

            // imshow("Target image",targetImg);
            // waitKey(0);
    renderCircle(endPoint,brushRadius,originImg,targetImg,currColor);
            // imshow("Target image",targetImg);
            // waitKey(0);

}

/**
 * Render a rectangle.
 */
void renderRectangle(Point2d startPoint, Point2d endPoint, double brushRadius,
                    const Mat& originImg, Mat& targetImg, Vec3b currColor){


    Point2d lineDirection = endPoint - startPoint;
    Point2d perpendicular(lineDirection.y, -1 * lineDirection.x);
    perpendicular = perpendicular/cv::norm(perpendicular);

    Point2d innerRect1 = startPoint - brushRadius * perpendicular;
    Point2d innerRect2 = startPoint + brushRadius * perpendicular;
    Point2d innerRect3 = endPoint + brushRadius * perpendicular;
    Point2d innerRect4 = endPoint - brushRadius * perpendicular;

    Point2d outerRect1 = startPoint - (brushRadius+FALLOFF) * perpendicular;
    Point2d outerRect2 = startPoint + (brushRadius+FALLOFF) * perpendicular;
    Point2d outerRect3 = endPoint + (brushRadius+FALLOFF) * perpendicular;
    Point2d outerRect4 = endPoint - (brushRadius+FALLOFF) * perpendicular;
    /*
    Point pt[1][4];
    pt[0][0] = innerRect1;
    pt[0][1] = innerRect2;
    pt[0][2] = innerRect3;
    pt[0][3] = innerRect4;

    const Point* ppt[1] = {pt[0]};
    int npt[] = {4};
    fillPoly(targetImg,ppt,npt,1,Scalar(currColor.val[0],currColor.val[1],currColor.val[2]),8);
    */
    int minimumX = min(outerRect1.x,min(outerRect2.x,min(outerRect3.x,outerRect4.x)));
    int maximumX = max(outerRect1.x,max(outerRect2.x,max(outerRect3.x,outerRect4.x)));
    int minimumY = min(outerRect1.y,min(outerRect2.y,min(outerRect3.y,outerRect4.y)));
    int maximumY = max(outerRect1.y,max(outerRect2.y,max(outerRect3.y,outerRect4.y)));
    // render the rectangle
    for(int i = minimumX-1;i <= maximumX + 1;i++){
        for(int j = minimumY-1;j <= maximumY+1;j++){
            Point2d currPoint(i,j);
            if(i < 0 || j < 0 || i >= originImg.cols || j >= originImg.rows)
                continue;
            if(pointInRectangle(currPoint,innerRect1,innerRect2,innerRect3,innerRect4)){
                targetImg.at<Vec3b>(j,i).val[0] = currColor.val[0];
                targetImg.at<Vec3b>(j,i).val[1] = currColor.val[1];
                targetImg.at<Vec3b>(j,i).val[2] = currColor.val[2];
            }
            else if(pointInRectangle(currPoint,outerRect1,outerRect2,outerRect3,outerRect4)){
                // calculate alpha value, the distanc depends on the distance between the
                // point and the line
                Point2d temp(startPoint.x- i,startPoint.y-j);
                double dotProduct = temp.x * perpendicular.x +  temp.y * perpendicular.y;
                double dist = abs(dotProduct / (cv::norm(perpendicular)));
                double alpha = (dist - brushRadius)/ FALLOFF;
                Vec3b pixelColor = targetImg.at<Vec3b>(j,i);
                targetImg.at<Vec3b>(j,i).val[0] = (uchar)((alpha)*((int)(pixelColor.val[0])) + (1-alpha) * ((int)(currColor.val[0])));
                targetImg.at<Vec3b>(j,i).val[1] = (uchar)((alpha)*((int)(pixelColor.val[1])) + (1-alpha) * ((int)(currColor.val[1])));
                targetImg.at<Vec3b>(j,i).val[2] = (uchar)((alpha)*((int)(pixelColor.val[2])) + (1-alpha) * ((int)(currColor.val[2])));
            }
        }
    }

}


void renderCircle(Point2d center, double brushRadius, const Mat& originImg, Mat& targetImg, Vec3b currColor){

    /*
    circle(targetImg,center,brushRadius,Scalar(currColor.val[0],currColor.val[1],currColor.val[2]),-1);
    */
    int minimumX = center.x - brushRadius - FALLOFF;
    int maximumX = center.x + brushRadius + FALLOFF;
    int minimumY = center.y - brushRadius - FALLOFF;
    int maximumY = center.y + brushRadius + FALLOFF;

    for(int i = minimumX-1;i <= maximumX+1;i++){
        for(int j = minimumY-1;j<=maximumY+1;j++){
            if(i < 0 || j < 0 || i >= originImg.cols || j >= originImg.rows){
                continue;
            }
            double dist = cv::norm(Point2d(i-center.x,j-center.y));
            if(dist <= brushRadius){
                targetImg.at<Vec3b>(j,i).val[0] = currColor.val[0];
                targetImg.at<Vec3b>(j,i).val[1] = currColor.val[1];
                targetImg.at<Vec3b>(j,i).val[2] = currColor.val[2];
            }
            else if(dist <= brushRadius + FALLOFF){
                double alpha = (dist - brushRadius) / FALLOFF;
                Vec3b pixelColor = targetImg.at<Vec3b>(j,i);
                targetImg.at<Vec3b>(j,i).val[0] = (uchar)((alpha)*((int)(pixelColor.val[0])) + (1-alpha) * ((int)(currColor.val[0])));
                targetImg.at<Vec3b>(j,i).val[1] = (uchar)((alpha)*((int)(pixelColor.val[1])) + (1-alpha) * ((int)(currColor.val[1])));
                targetImg.at<Vec3b>(j,i).val[2] = (uchar)((alpha)*((int)(pixelColor.val[2])) + (1-alpha) * ((int)(currColor.val[2])));
            }
        }
    }
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
/**
 * Given a position, calculate the orientation of the stroke at that position.
 * @param gradientX gradient at x direction
 * @param gradientY gradient at y direction
 * @return direction of the stroke
 */
Point2d calDirection(const Mat& gradientX, const Mat& gradientY, double centerX, double centerY){
    // discretization the angle, make a histogram
    // 30 bins, each bins covers 12 degrees
    int bins[30];

    memset(bins,0,sizeof(bins));

    for(int i = centerX - GRADIENT_RADIUS;i < centerY + GRADIENT_RADIUS;i++){
        for(int j = centerY - GRADIENT_RADIUS;j < centerY + GRADIENT_RADIUS;j++){
            //in circle test
            if(i < 0 || j < 0 || i >= gradientX.cols || j >= gradientX.rows)
                continue;
            double dist = cv::norm(Point2d(i-centerX,j-centerY));
            if(dist > GRADIENT_RADIUS)
                continue;

            double gradX = gradientX.at<short int>(j,i);
            double gradY = gradientY.at<short int>(j,i);
            if(abs(gradX) < 3.0 || abs(gradY) < 3.0)
                continue;
            double angle = atan(gradY/gradX) * 180 /PI;

            if(gradX > 0 && gradY < 0){
                angle = 360 + angle;
            }
            else if(gradX < 0 && gradY > 0){
                angle = 180 + angle;
            }
            else if(gradX < 0 && gradY < 0){
                angle = 180 + angle;
            }

            int index = angle/12;

            bins[index] = bins[index] + 1;
        }
    }
    int maxValue = -1;
    int maxIndex = 0;
    for(int i = 0;i < 30;i++){
        if(bins[i] > maxValue){
            maxValue = bins[i];
            maxIndex = i;
        }
    }
    Point2d direction;
    double angle = 12 * (maxIndex);
    int randomAngle = randomBrushLength(0,30);
    angle = angle + randomAngle - 15 + 90;
    angle = (int)angle % 360;
    if(angle < 90){
        direction.x = 1;
        direction.y = tan(angle*PI/180.0);
    }
    else if(angle < 180){
        direction.x = -1;
        direction.y = abs(tan(angle*PI/180.0));
    }
    else if(angle < 270){
        direction.x = -1;
        direction.y = -abs(tan(angle*PI/180.0));
    }
    else if(angle < 360){
        direction.x = 1;
        direction.y = -abs(tan(angle*PI/180.0));
    }

    direction = direction / cv::norm(direction);
    return direction;
}

Point2d operator/(const Point2d& p1, double t){
    return Point2d(p1.x/t,p1.y/t);
}

/*
Mat renderImage(const Mat& originImg, vector<vector<Point2d> > region){
    Mat targetImg;
    // render region by region
    // for each region, find the position with maximum gradients,
    // start from these positions to render the whole image.
    for(int i = 0;i < region.size();i++){
        vector<Point2d> currRegion = region[i];
        //find the points with greatest region

    }
}
*/

/**
 * Given the gradient in x direction, for each region, we find positions
 * with maximum graident, and for other positions, their gradient value
 * are got through interpolation;
 * In the paper, this is replaced with "thin plate spline";
 */
void calDirectionWithTPS(const Mat& gradientX, Mat& smoothGradientX,
            const vector<vector<Point2d> >& region){

    smoothGradientX = Mat::zeros(gradientX.rows,gradientX.cols,CV_32FC1);

    for(int i = 0;i < region.size();i++){
        vector<Point2d> currRegion = region[i];
        vector<pair<double, Point2d> > currGradientX;
        for(int j = 0;j < currRegion.size();j++){
            int x = currRegion[i].x;
            int y = currRegion[i].y;
            pair temp;
            temp.first = gradientX.at<short int>(y,x);
            temp.second = currRegion[j];
            currGradientX.push_back(temp);
        }
        //sort gradient and find maximum
        sort(currGradientX.begin(),currGradientX.end());

        vector<pair<double, Point2d> > interpolationPoints;
        for(int j = 0;j < currGradientX.size();i++){
            //get current point
            if(interpolationPoints.size() > MAXINTERPOLATIONNUM)
                break;
            pair<double, Point2d> temp = currGradientX[j];
            bool flag  = true;
            for(int k = 0;k < interpolationPoints.size();k++){
                // seleced points should not be too close, therefore we set a minimum distance
                if(norm::(interpolationPoints[i].second - temp.second) < INTERPOLATIONRANGE){
                    flag = false;
                    break;
                }
            }
            if(flag){
                interpolationPoints.push_back(temp);
            }
        }

        for(int j = 0;j < interpolationPoints.size();j++){
            int x = interpolationPoints[j].second.x;
            int y = interpolationPoints[j].second.y;
            double value = interpolationPoints[j].first;
            smoothGradientX.at<float>(y,x) = value;
        }

        for(int j = 0;j < currRegion.size();j++){
            vector<double> efficient;
            double distanceSum = 0;
            bool flag = false;
            int x = currRegion[i].x;
            int y = currRegion[i].y;
            for(int k = 0;k < interpolationPoints.size();k++){
                double temp = norm(interpolationPoints[i].second - currRegion[i]);
                if(temp < 1e-5){
                    flag = true;
                    break;
                }
                efficient.push_back(1.0/temp);
                distanceSum += 1.0/temp;
            }

            if(flag)
                continue;
            double gradientValue = 0;
            for(int k = 0;k < efficient.size();k++){
                gradientValue = gradientValue + interpolationPoints[k].first * efficient[i]/distanceSum;
            }
            smoothGradientX.at<float>(y,x) = gradientValue;
        }

    }
}

/**
 * Operator overload for comparing two "pairs";
 */
bool operator<(const pair<double,Point2d>& p1, const pair<double,Point2d>& p2){
    if(p1.first < p2.first)
        return false;
    return true;
}


void renderByRegion(const vector<vector<Point2d> >& region){
    //read image
    Mat img = imread("./lena.png");
    Mat blurImage = smoothImage(img,cv::Size(21,21),0,0);

    Mat gradientX;
    Mat gradientY;
    Mat sobelFilteredImage;
    Mat intensity = calIntensityImage(blurImage);
    intensity = smoothImage(intensity,cv::Size(11,11),0,0);
    sobelFilter(intensity,gradientX,gradientY,sobelFilteredImage);

    Mat smoothGradientX;
    Mat smoothGradientY;
    calDirectionWithTPS(gradientX,smoothGradientX,region);
    calDirectionWithTPS(gradientY,smoothGradientY,region);

    //render region by region
    for(int i = 0;i < region.size();i++){
        // find start points of brush
        vector<Point2d> currRegion = region[i];
        vector<pair<double,Point2d> > regionPair;
        for(int j = 0;j < currRegion.size();j++){
            int x = currRegion[j].x;
            int y = currRegion[j].y;
            pair<double,Point2d> temp;
            double temp1 = smoothGradientX.at<float>(y,x);
            double temp2 = smoothGradientY.at<float>(y,x);
            temp.first = sqrt(temp1*temp1 + temp2*temp2);
            temp.second = currRegion[j];
            regionPair.push_back(temp);
        }
        sort(regionPair.begin(),regionPair.end());

        // find some points
        vector<pair<double,Point2d> > selecedStart;
        for(int j = 0;j < regionPair.size();i++){
            bool flag = true;
            for(int k = 0;k < selecedStart.size();k++){
                if(norm(regionPair[j].second - selecedStart[k].second) < PLACEBRUHSPOS){
                    flag = false;
                    break;
                }
                selecedStart.push_back(regionPair[j]);
            }
        }

        //start put brushes
        for(int j = 0;j < selecedStart.size();j++){
            Point2d currPosition = selecedStart[i].second;
            int brushCount = 0;
            int maxBrushNum = randomInt(MINBRUSHNUM,MAXBRUSHNUM);
            vector<Point2d> brushPoints;
            vector<Point2d> brushPointsWithBlend; 
            Point2d startPoint;
            Point2d endPoint;
            while(brushCount < maxBrushNum){
                Point2d direction;
                direction.x = smoothGradientX.at<float>(y,x);
                direction.y = smoothGradientY.at<float>(y,x); 
                direction = direction / norm(direction);
                double strokeLength = randomBrushLength()
                clipStroke(sobelFilteredImage,currPosition.x,currPosition.y,direction,startPoint,endPoint,
                        direction,);
                renderTriangle(startPoint,endPoint,originImg,)  
            } 
        }
    }
}

