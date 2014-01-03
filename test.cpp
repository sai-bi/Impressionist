/**
 * @author 
 * @version 2014/01/03
 */

#include "impressionist.h"


int main(){
    Mat img = imread("./shading.bmp");
    Mat blurImage = smoothImage(img,cv::Size(21,21),1,1);
    imshow("Blur Image", blurImage);
    imwrite("smoothShading.bmp",blurImage); 
    Mat gradientX;
    Mat gradientY;
    Mat gradient;
    Mat intensity;
    
    // intensity = calIntensityImage(blurImage);
    // sobelFilter(intensity,gradientX,gradientY,gradient);

    // imshow("Graident",gradient);    

        
    waitKey(0); 
    return 0; 
}


