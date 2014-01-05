/**
 * @author 
 * @version 2014/01/03
 */

#include "impressionist.h"


int main(){
    Mat img = imread("./test.png");
    Mat blurImage = smoothImage(img,cv::Size(21,21),0,0);
    imshow("Blur Image", blurImage);
    
    Mat gradientX;
    Mat gradientY;
    Mat sobelFilteredImg;
    Mat targetImg = Mat::zeros(img.rows,img.cols,CV_8UC3);
    Mat intensity = calIntensityImage(blurImage);
    intensity = smoothImage(intensity,cv::Size(21,21),0,0); 
    sobelFilter(intensity,gradientX,gradientY,sobelFilteredImg);
    sobelFilteredImg = sobelFilteredImg > 128;
    imshow("Image Gradient",sobelFilteredImg);
    for(int i = 0;i < img.cols;i=i+5){
        for(int j = 0;j < img.rows;j=j+5){
            Point2d direction = calDirection(gradientX,gradientY,i,j);
            direction = Point2d(direction.y,0 - direction.x);
             
            direction = direction / cv::norm(direction);
            // cout<<direction.x<<" "<<direction.y<<endl;
            renderStroke(img,targetImg,sobelFilteredImg,i,j,direction); 
            // imshow("Target image",temp);
            // waitKey(0);
        }
    }
    imshow("Target image",targetImg);
    waitKey(0); 

    return 0; 
}


