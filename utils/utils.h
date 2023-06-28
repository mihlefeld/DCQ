//
// Created by thesh on 28/06/2023.
//

#ifndef DCQ_UTILS_H
#define DCQ_UTILS_H
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace dcq::utils {
    torch::Tensor image_to_tensor(const cv::Mat &image);

    cv::Mat tensor_to_image(const torch::Tensor &tensor);
}

#endif //DCQ_UTILS_H
