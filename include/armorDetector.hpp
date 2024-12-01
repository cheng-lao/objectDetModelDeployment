#include <iostream>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

struct CenterPrior
{
    int x;
    int y;
    int stride;
}; // 每个中心框的信息 x,y中心坐标和stride

struct object_rect
{
    int x;
    int y;
    int width;
    int height;
}; // 目标框的信息

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score; // confidence
    int label;   // class
} BoxInfo;

const int color_list[9][3] =
    {
        //{255 ,255 ,255}, //bg
        {216, 82, 24},
        {236, 176, 31},
        {118, 171, 47},
        {76, 189, 237},
        {238, 19, 46},
        {76, 76, 76},
        {153, 153, 153},
        {255, 0, 0},
};

class Nanodet
{

public:
    Nanodet(const char *Path,
            const vector<int> &stride,
            int num_class, int input_shape,
            float iou_threshold, float score_threshold);
    cv::dnn::Net net;
    int num_class;
    float input_shape;
    float iou_threshold;
    vector<int> strides;
    float score_threshold;

    std::vector<BoxInfo> detect(cv::Mat &img);
    void resize_uniform(cv::Mat &src, cv::Mat &resized_img, cv::Size dstsize, object_rect &effect_roi);
    cv::Mat draw_bboxes(const cv::Mat &bgr, std::vector<BoxInfo> &bboxes, object_rect effect_roi);

private:
    const int reg_max = 7;
    const float mean[3] = {103.53, 116.28, 123.675};
    const float std[3] = {57.375, 57.12, 58.395};
    std::vector<BoxInfo> post_process(cv::Mat &out, cv::Mat &srcimg, int newh, int neww);
    void normalize(cv::Mat &img);
    void generate_grid_center_priors(int newh, int neww, vector<int> strides, std::vector<CenterPrior> &center_priors);
    void decode_infer(cv::Mat &, std::vector<CenterPrior> &center_priors, float threshold, std::vector<std::vector<BoxInfo>> &results);
    // BoxInfo disPred2Bbox(const float *bbox_pred, int cur_label, float score, const int x, const int y, const int stride);
    BoxInfo disPred2Bbox(const float *bbox_pred, int cur_label, float score, const int x, const int y, const int stride);
    // void NMSopera(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);
    int activation_function_softmax(const float *src, float *dst, int length);
};
