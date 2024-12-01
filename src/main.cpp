#include "../include/armorDetector.hpp"
#include <string>
using namespace std;

void image_display(const string& path, Nanodet& nanodet)
{
    // image display
    object_rect effect_roi;
    cv::Mat img = cv::imread(path , -1);
    cv::Mat resized_img;
    nanodet.resize_uniform(img, resized_img, cv::Size(320, 320), effect_roi);
    auto results = nanodet.detect(resized_img);
    cv::Mat image = nanodet.draw_bboxes(img, results, effect_roi);
    imshow("result", image);
    waitKey(0);
}

void video_dispaly(const string& path, Nanodet& nanodet){
    cv::VideoCapture video(path);
    if (!video.isOpened()) {
        printf("Could not open the video\n");
        return;
    }
    while (1) {
        cv::Mat resized_img;
        object_rect effect_roi;
        cv::Mat frame;
        video >> frame;
        if (frame.empty())
            break;

        nanodet.resize_uniform(frame, resized_img, cv::Size(320, 320), effect_roi);
        auto results = nanodet.detect(resized_img);
        cv::Mat image = nanodet.draw_bboxes(frame, results, effect_roi);
        imshow("video", image);
        int k = waitKey(1);
        if (k == 27)
            break;
    }
}

int main(int argc, char const *argv[])
{
    if (argc != 3) {
        printf("usage: [mode] [path/camera port] \n mode = 1 represents image display \n mode = 2 represents video display \n mode = 3 represents camera display :) \n");
        return -1;
    }
    int mode = std::stoi(argv[1]);
    std:: string path = argv[2];

    Nanodet nanodet("/home/chengyingjie666/code/objectDetModelDeployment/source/nanodet_RMDataset.onnx",
                    {8, 16, 32},
                    9, 320, (float)0.3, (float)0.3);
    // cv::Mat img = cv::imread("/home/chengyingjie666/code/objectDetModelDeployment/image/2925.jpg");
    // cv::VideoCapture video("/home/chengyingjie666/code/objectDetModelDeployment/video/video_zhuangjiaban.mp4");

    switch (mode) {
        case 1:
            image_display(path, nanodet);
            break;
        case 2:
            video_dispaly(path, nanodet);
            break;
        case 3:
            printf("To be implement!");
            break;
        default:
            printf("Invalid mode. Use 1 for image or 2 for video.\n");
            return -1;
    }
    return 0;
}
