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
    double lastIntensity = bilinearIntensitySample(sobelFilteredImage, tempX, tempY);
    while(true){
        tempX = tempX + direction.x;
        tempY = tempY + direction.y;
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
         
    while(true){
        tempX = tempX + direction.x;
        tempY = tempY + direction.y;
        if(norm(Point2d(tempX - centerX, tempY - centerY)) > strokeLength/2)
           break;
        double newIntensity = bilinearIntensitySample(sobelFilteredImage,tempX,tempY);
        if(newIntensity < lastIntensity)
           break;
        lastIntensity = newIntensity; 
    }
    endX = tempX;
    endY = tempY;

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
    int leftX = int(x);
    int leftY = int(y);
    
    double leftupIntensity = (int)(sobelFilteredImage.at<uchar>(leftY,leftX));
    double leftdownIntensity = (int)(sobelFilteredImage.at<uchar>(leftY+1,leftX));
    double rightUpIntensity = (int)(sobelFilteredImage.at<uchar>(leftY,leftX+1));
    double rightDownIntensity = (int)(sobelFilteredImage.at<uchar>(leftY+1,leftX+1));

    double temp1 = leftupIntensity * (x-leftX) + rightUpIntensity * (leftX+1-x);
    double temp2 = leftdownIntensity * (x-leftX) + rightDownIntensity *(leftX+1-x);

    double result = temp1 * (y-leftY) + temp2(leftY+1-y);

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
    
    double brushRadius = randomBrushRadius(BRUSH_RADIUS_1, BRUSH_RADIUS_2);

    //color to be filled
    Vec3b currColor = originImg.at<Vec3b>(centerY,centerX);
        
    Point2d lineDirection = endPoint - startPoint;
    Point2d perpendicular(lineDirection.y,lineDirection.x);
    perpendicular = perpendicular/cv::norm(perpendicular);

    //calculate four points of the rectangle
    Point2d innerRect1 = startPoint - brushRadius * perpendicular;
    Point2d innerRect2 = startPoint + brushRadius * perpendicular;
    Point2d innerRect3 = startPoint + brushRadius * perpendicular;
    Point2d innerRect4 = startPoint - brushRadius * perpendicular;
     
    Point2d outerRect1 = startPoint - (brushRadius+FALLOFF) * perpendicular;
    Point2d outerRect2 = startPoint + (brushRadius+FALLOFF) * perpendicular;
    Point2d outerRect3 = startPoint + (brushRadius+FALLOFF) * perpendicular;
    Point2d outerRect4 = startPoint - (brushRadius+FALLOFF) * perpendicular;
    
    int minimumX = min(outerRect1.x,min(outerRect2.x,min(outerRect3.x,outerRect4.x)));
    int maximumX = max(outerRect1.x,max(outerRect2.x,max(outerRect3.x,outerRect4.x)));
    int minimumY = min(outerRect1.y,min(outerRect2.y,min(outerRect3.y,outerRect4.y)));
    int maximumY = max(outerRect1.y,max(outerRect2.y,max(outerRect3.y,outerRect4.y)));
    
    //render the circle
    int minimumX = startPoint.x - brushRadius - FALLOFF;
    int maximumX = startPoint.x + brushRadius + FALLOFF;
    int minimumY = startPoint.y - brushRadius - FALLOFF;
    int maximumY = startPoint.y + brushRadius + FALLOFF;

    for(int i = minimumX-1;i <= maximumX+1;i++){
        for(int j = minimumY-1;j<=maximumY+1;j++){
            if(i < 0 || j < 0 || i >= originImg.cols || j >= originImg.rows){
                continue;
            }
            double dist = cv::norm(Point2d(i-centerX,j-centerY));
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


    minimumX = min(outerRect1.x,min(outerRect2.x,min(outerRect3.x,outerRect4.x)));
    maximumX = max(outerRect1.x,max(outerRect2.x,max(outerRect3.x,outerRect4.x)));
    minimumY = min(outerRect1.y,min(outerRect2.y,min(outerRect3.y,outerRect4.y)));
    maximumY = max(outerRect1.y,max(outerRect2.y,max(outerRect3.y,outerRect4.y)));
    // render the rectangle
    for(int i = minimumX-1;i <= maximumX + 1;i++){
        for(int j = minimumY-1;j <= maximumY+1;j++){
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
                Point2d temp(centerX - i,centerY-j);
                double dotProduct = temp.x * perpendicular.x +  temp.y * perpendicular.y;
                double dist = dotProduct / (cv::norm(perpendicular));
                double alpha = 1 - dist / falloff;
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
bool pointInRectangle(const Point2d& point, const Point2d& rectangle1, const Point2d& rectangle2
                    const Point2d& rectangle3, const Point2d& rectangle4){
    double temp1 = (point - rectangle1) * (rectangle2 - rectangle1);
    double temp2 = (rectangle2 - rectangle1) * (rectangle2 - rectangle1);
    if(temp1 > temp2 || temp1 < 0)
        return false;
    double temp3 = (point - rectangle) * (rectangle4 - rectangle1);
    double temp4 = (rectangle4 - rectangle1) * (rectangle4 - rectangle1);
    if(temp3 < 0 || temp3 > temp4)
        return false;

    return true;    
}

/**
 * Operator overload for multiply two points.
 */
dobule operator*(const Point2d& p1, const Point2d& p2){
    return (p1.x * p2.x +  p1.y * p2.y);
}
/**
 * Given a position, calculate the orientation of the stroke at that position.
 * @param gradientX gradient at x direction
 * @param gradientY gradient at y direction
 * @return direction of the stroke
 */ 
Point2d calDirection(const Mat& gradientX, const Mat& gradientY, double centerX, double centerY){
    //discretization the angle, make a histogram
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
            double gradX = gradientX.at<float>(j,i);
            double gradY = gradientY.at<float>(j,i);
            
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
                angle = 180 - angle;  
            }
            
            int index = angle/12;
            bins[index] = bins[index] + 1;
        }
    }
    
    int minValue = 1000000;
    int minIndex = 0;
    for(int i = 0;i < 30;i++){
        if(bins[i] < minValue){
            minValue = bins[i];
            minIndex = i;
        }
    }
    Point2d direction;
    double angle = 12 * (minIndex);
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


