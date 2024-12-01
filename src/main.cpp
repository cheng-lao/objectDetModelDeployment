#include "../include/armorDetector.hpp"

using namespace std;

int main()
{
    Nanodet nanodet("/home/chengzi/Desktop/nanodet_RM/source/nanodet_RMDataset.onnx",
                    {8, 16, 32},
                    9, 320, (float)0.3, (float)0.3);
    // cv::Mat img = cv::imread("/home/chengzi/Desktop/nanodet_RM/image/2925.jpg");
    cv::VideoCapture video("/home/chengzi/Desktop/nanodet_RM/video/video_zhuangjiaban.mp4");
    while(1){
        cv::Mat resized_img;
        object_rect effect_roi;
        cv::Mat frame;
        video>>frame;
        if(frame.empty()) break;

        nanodet.resize_uniform(frame, resized_img, cv::Size(320, 320), effect_roi);
        auto results = nanodet.detect(resized_img);
        cv::Mat image = nanodet.draw_bboxes(frame, results, effect_roi);
        imshow("video", image);
        int k = waitKey(1);
        if(k==27) break;
        
    }
    /*
    cv::Mat resized_img;
    nanodet.resize_uniform(img, resized_img, cv::Size(320, 320), effect_roi);
    auto results = nanodet.detect(resized_img);
    cv::Mat image = nanodet.draw_bboxes(img, results, effect_roi);
    imshow("result", image);
    waitKey(0);
    */

    return 0;
}
