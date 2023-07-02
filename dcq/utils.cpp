//
// Created by thesh on 28/06/2023.
//
#include "utils.h"
#include <opencv2/highgui.hpp>

using namespace torch::indexing;

torch::Tensor dcq::utils::image_to_tensor(const cv::Mat &image) {
    int h = image.rows;
    int w = image.cols;
    int c = image.channels();
    auto tensor = torch::from_blob(image.data, {h, w, c},torch::kU8);
    return tensor.to(torch::kFloat32) / 255;
}

cv::Mat dcq::utils::tensor_to_image(const torch::Tensor &tensor) {
    auto h = tensor.size(0);
    auto w = tensor.size(1);
    auto c = tensor.size(2);
    auto tensorU8 = (tensor.clip(0, 1) * 255);
    tensorU8 = tensorU8.flatten().to(torch::kU8);
    auto mat = cv::Mat(h, w, CV_8UC(c), tensorU8.data_ptr<uint8_t>());
    return mat.clone();
}

torch::Tensor dcq::utils::to_batched(const torch::Tensor & tensor) {
    return tensor.permute({2, 0, 1}).index({None});
}

torch::Tensor dcq::utils::to_conv_kernel(const torch::Tensor & tensor) {
    return tensor.permute({2, 0, 1}).index({Slice(), None});
}

torch::Tensor dcq::utils::from_batched(const torch::Tensor & tensor) {
    int c = tensor.size(1);
    int h = tensor.size(2);
    int w = tensor.size(3);

    return tensor.permute({0, 2, 3, 1}).flatten().reshape({h, w, c});
}

void dcq::utils::imshow(const std::string &window_name, const torch::Tensor &tensor, bool add_shape, bool normalize) {
    auto x = tensor.clone();
    if (normalize) {
        x -= x.min();
        x /= x.max();
    }
    std::stringstream  buffer;
    buffer << window_name;
    if (add_shape) {
        buffer << " " << tensor.sizes();
    }
    auto image = dcq::utils::tensor_to_image(x);
    cv::Mat out_image;
    float factor = 1;
    if (image.rows < 400) {
        factor = 400.0f / (float) image.rows;
    }
    cv::resize(image, out_image, cv::Size(), factor, factor, cv::INTER_NEAREST);
    cv::imshow(buffer.str(), out_image);
    cv::waitKey(0);
}

void dcq::utils::palette_show(const std::string &window_name, const torch::Tensor &palette, bool add_shape) {
    int k = palette.size(0);
    int c = palette.size(1);
    int h = 900;
    int w = 300;
    auto palette_vis = torch::zeros({h, w, c});
    auto color_positions = torch::linspace(0, h, k + 1, torch::kInt32);

    for (int i = 0; i < k; i++) {
        auto from = color_positions.index({i}).item<int>();
        auto to = color_positions.index({i + 1}).item<int>() - 5;
        auto color = palette.index({i}).index({None, None});
        palette_vis.index_put_({Slice(from, to, 1)}, color);
    }

    imshow(window_name, palette_vis, add_shape);
}

void dcq::utils::compare_show(const std::string &window_name, const torch::Tensor &a, const torch::Tensor &b, bool add_shape, bool normalize) {
    dcq::utils::imshow(window_name, torch::concat({a, b}, 1), add_shape, normalize);
}

void dcq::utils::count_assignments(const std::string &prefix, torch::Tensor &M, int K) {
    int max = M.max().item<int>();
    std::cout << prefix << " ";
    auto total = M.size(0) * M.size(1);
    for (int i = 0; i < K; i++) {
        std::cout << ((M == i).sum() / total).item<float>() << " ";
    }
    std::cout << std::endl;
}

bool dcq::utils::is_close(float a, float b, float atol, float rtol) {
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
}

