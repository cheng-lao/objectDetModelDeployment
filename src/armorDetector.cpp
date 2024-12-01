#include "armorDetector.hpp"

void NMS(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);

Nanodet::Nanodet(const char *Path, const vector<int> &stride,
                 int num_class, int input_shape, float iou_threshold, float score_threshold)
{

    this->net = cv::dnn::readNetFromONNX(Path);
    this->num_class = num_class;
    this->input_shape = input_shape;
    this->iou_threshold = iou_threshold;
    this->score_threshold = score_threshold;
    // this->net = cv::dnn::readNetFromONNX("nanodet.onnx");
    for (int i = 0; i < stride.size(); i++)
    {
        this->strides.push_back(stride[i]);
    }
}

std::vector<BoxInfo> Nanodet::detect(cv::Mat &resized_img)
{
    this->normalize(resized_img); // normalize image
    // imshow("srcimg_normalize", srcimg);
    // waitKey(0);
    int newh = this->input_shape;
    int neww = this->input_shape;
    Mat blob = cv::dnn::blobFromImage(resized_img);

    this->net.setInput(blob);
    vector<Mat> outs;
    this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

    return this->post_process(outs[0], resized_img, newh, neww);
}

std::vector<BoxInfo> Nanodet::post_process(cv::Mat &out, cv::Mat &srcimg, int newh, int neww)
{
    std::vector<CenterPrior> center_priors;
    this->generate_grid_center_priors(newh, neww, this->strides, center_priors);
    // cout << "the size of center_prious is " << center_priors.size() << endl;

    std::vector<std::vector<BoxInfo>> results;
    results.resize(center_priors.size() + 5);
    // 调用函数之前 center_priors填充了所有的中心点的先验框 out是网络的输出但是是空的 results是一个二维数组 里面存放的是每个类别的检测结果 也是空的
    // this->prob_threshold = 0.35
    this->decode_infer(out, center_priors, this->score_threshold, results);
    std::vector<BoxInfo> dets;
    for (int i = 0; i < (int)results.size(); i++)
    {
        NMS(results[i], this->iou_threshold);
        for (auto box : results[i])
        {
            dets.push_back(box);
        }
    }

    return dets;
}

cv::Mat Nanodet::draw_bboxes(const cv::Mat &bgr, std::vector<BoxInfo> &bboxes, object_rect effect_roi)
{
    static const char *class_names[] = {"car_red", "car_blue", "car_unknow", "watcher_red",
                                        "watcher_blue", "watcher_unknow", "armor_red", "armor_blue", "armor_grey"};

    cv::Mat image = bgr.clone();
    int src_w = image.cols; // 源图像的宽度
    int src_h = image.rows; // 源图像的高度
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        BoxInfo &bbox = bboxes[i]; // 每个检测框的信息 左上角的x y,右下角的x y,同时还有信度score 和 类别 label

        cv::Scalar color = cv::Scalar(color_list[bbox.label][0], color_list[bbox.label][1], color_list[bbox.label][2]);

        cv::rectangle(image, cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio), cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio)), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100); //

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (bbox.x1 - effect_roi.x) * width_ratio;
        int y = (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    return image;
    // cv::imshow("image", image);
}

/// @brief 生成格式化到固定大小的图像,并且计算在有效图像区域的左上角坐标和宽高,具体裁剪图像的方法是在保持宽高比的情况下将图像的宽或者高设置为416或者320 \
///        如果保持宽为416/320,就在图片的上下两侧填充0,扩充高度为416/320.
/// @param src  源图像
/// @param resized_img 生成的图像
/// @param dstsize 生成的图像
/// @param effect_roi   有效图像区域(x,y,width,height)左上角坐标和宽高
void Nanodet::resize_uniform(cv::Mat &src, cv::Mat &resized_img, cv::Size dstsize, object_rect &effect_roi)
{
    int w = src.cols; // 源图像的宽度
    int h = src.rows; // 源图像的高度
    int dst_w = dstsize.width;
    int dst_h = dstsize.height;

    resized_img = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;         // 源图像的宽高比
    float ratio_dst = dst_w * 1.0 / dst_h; // 目标图像的宽高比

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst)
    {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst)
    {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else
    {
        cv::resize(src, resized_img, dstsize);
        effect_roi.x = 0;
        effect_roi.y = 0;
        effect_roi.width = dst_w;
        effect_roi.height = dst_h;
        // return 0;
    }

    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h)); // 在保持了宽高比的同时 将图像的宽设置为416
    if (tmp_w != dst_w)
    {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        // std::cout << "index_w: " << index_w << std::endl;
        for (int i = 0; i < dst_h; i++)
        {
            memcpy(resized_img.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
            // memcpy函数 从tmp.data + i * tmp_w * 3 复制tmp_w * 3个字节到dst.data + i * dst_w * 3 + index_w * 3
        }
        effect_roi.x = index_w;
        effect_roi.y = 0;
        effect_roi.width = tmp_w;
        effect_roi.height = tmp_h;
    }
    else if (tmp_h != dst_h)
    {
        int index_h = floor((dst_h - tmp_h) / 2.0);

        memcpy(resized_img.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_roi.x = 0;       // 实际图像区域在dst中x轴的相对坐标是 0
        effect_roi.y = index_h; // 实际图像区域在dst中y轴的相对坐标是 index_h
        effect_roi.width = tmp_w;
        effect_roi.height = tmp_h;
    }
    else
    {
        printf("error in resize_uniform\n");
        exit(-1);
    }

    // cv::imshow("dst", dst);
    // cv::waitKey(0);
}

void Nanodet::normalize(cv::Mat &img)
{
    img.convertTo(img, CV_32F);
    int i = 0, j = 0;
    for (i = 0; i < img.rows; i++)
    {
        float *pdata = (float *)(img.data + i * img.step);
        for (j = 0; j < img.cols; j++)
        {
            pdata[0] = (pdata[0] - this->mean[0]) / this->std[0];
            pdata[1] = (pdata[1] - this->mean[1]) / this->std[1];
            pdata[2] = (pdata[2] - this->mean[2]) / this->std[2];
            pdata += 3;
        }
    }
}

void Nanodet::generate_grid_center_priors(int newh, int neww, vector<int> strides, std::vector<CenterPrior> &center_priors)
{

    for (int i = 0; i < 3; i++)
    {
        // cout << " i is " << i << endl;
        const int stride = strides[i];
        int feat_w = ceil((float)neww / stride);
        int feat_h = ceil((float)newh / stride);

        for (int y = 0; y < feat_h; y++)
        {
            for (int x = 0; x < feat_w; x++)
            {
                CenterPrior ct;
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                center_priors.push_back(ct);
            }
        }
    }
}

void Nanodet::decode_infer(cv::Mat &feats, std::vector<CenterPrior> &center_priors, float threshold, std::vector<std::vector<BoxInfo>> &results)
{
    const int num_points = center_priors.size(); // 2100个标框
    // printf("num_points:%d\n", num_points);

    for (int idx = 0; idx < num_points; idx++)
    {
        // idx 的实际意义就是当前这个框的索引

        const int ct_x = center_priors[idx].x;        // 先验框的中心点x
        const int ct_y = center_priors[idx].y;        // 先验框的中心点y
        const int stride = center_priors[idx].stride; // 先验框的步长

        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < this->num_class; label++) // 遍历一遍所有的类别 找到最大的类别
        {

            float tmp = feats.at<float>(0, idx, label); //
            if (tmp > score)
            {
                score = tmp;       // 找到最大的分数
                cur_label = label; // 找到最大的类别
            }
        }

        if (score > threshold) // 如果最大的分数大于阈值
        {
            // std::cout << "label:" << cur_label << " score:" << score << "!!!!!" << std::endl;
            /*
             * feats是一个包含模型预测结果的矩阵，每一行对应一个中心点先验框。
             * 每一行的前num_class个元素是类别预测的分数，后面的元素是边界框预测。
             */
            const float *bbox_pred = feats.ptr<float>(0, idx) + this->num_class; // 所以这里是跳过了36个类别的概率预测 直接计算边界框的预测值
            auto tmp = this->disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride);
            results[cur_label].push_back(tmp); //
                                               // BoxInfo disPred2Bbox(const float*, float cur_label, float score, const int ct_x, const int ct_y, const int stride);
        }
    }
    // cout << "decode_infer is ok!!!" << endl;
}

BoxInfo Nanodet::disPred2Bbox(const float *bbox_pred, int cur_label, float score, const int x, const int y, const int stride)
{
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++)
    {
        // 这部分的代码不好解释,我暂时也没有弄懂,回头会补充回来的!
        float dis = 0;
        float *dis_after_sm = new float[this->reg_max + 1];
        activation_function_softmax(bbox_pred + i * (this->reg_max + 1), dis_after_sm, this->reg_max + 1);
        for (int j = 0; j < this->reg_max + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        // std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)this->input_shape);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)this->input_shape);
    // 总之通过这个函数 可以计算出一个实际预测框的坐标 包括左上角和右下角的坐标 预测度 以及类别
    // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return BoxInfo{xmin, ymin, xmax, ymax, score, cur_label};
}

void NMS(std::vector<BoxInfo> &input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b)
              { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

inline float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

int Nanodet::activation_function_softmax(const float *src, float *dst, int length)
{
    const float alpha = *std::max_element(src, src + length);
    float denominator{0};

    for (int i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }

    return 0;
}
