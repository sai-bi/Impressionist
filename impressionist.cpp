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
/*
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
}
*/
/**
 * Given a point(x,y), calculate the intensity at that point using bilinear interpolation.
 * @param sobelFilteredImage image after sobel filter
 * @param x 
 * @param y
 */
/*
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

void renderStroke(const Mat& originImg, const Mat& targetImg,int centerX, int centerY){
       
}
*/
