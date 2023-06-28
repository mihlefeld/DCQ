//
// Created by thesh on 28/06/2023.
//
#include "utils.h"

torch::Tensor dcq::utils::image_to_tensor(const cv::Mat &image) {
    int h = image.rows;
    int w = image.cols;
    int c = image.channels();
    auto tensor = torch::from_blob(image.data, {h, w, c},
                                   torch::TensorOptions().dtype(torch::kU8));
    tensor = tensor.permute({2, 0, 1});
    return tensor.to(torch::kFloat32) / 255;
}

cv::Mat dcq::utils::tensor_to_image(const torch::Tensor &tensor) {
    auto c = tensor.size(0);
    auto h = tensor.size(1);
    auto w = tensor.size(2);
    auto tensorU8 = (tensor.permute({1, 2, 0}) * 255).to(torch::kU8);
    auto mat = cv::Mat(h, w, CV_8UC(c), tensorU8.data_ptr<uint8_t>());
    return mat.clone();
}