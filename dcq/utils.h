//
// Created by thesh on 28/06/2023.
//

#ifndef DCQ_UTILS_H
#define DCQ_UTILS_H

#define PRINT_SHAPE(x) std::cout << #x".sizes() = " << x.sizes() << std::endl;
#define DEBUG_SHOW(x) dcq::utils::imshow(#x, x, true);
#define DEBUG_NSHOW(x) dcq::utils::imshow(#x, x, true, true);
#define DEBUG_PALETTE(x) dcq::utils::palette_show(#x, x, true);
#define DEBUG_ASSIG(x) dcq::utils::count_assignments(#x, x);
#define DEBUG_COMPARE(x, y) dcq::utils::compare_show(#x" vs "#y, x, y, true);
#define TIME_IT(x) { \
auto start = std::chrono::high_resolution_clock::now(); \
x; \
auto stop = std::chrono::high_resolution_clock::now();  \
std::cout << #x" took " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl; \
}
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <chrono>

namespace dcq::utils {
    torch::Tensor image_to_tensor(const cv::Mat &image);

    cv::Mat tensor_to_image(const torch::Tensor &tensor);

    torch::Tensor to_batched(const torch::Tensor & tensor);

    torch::Tensor to_conv_kernel(const torch::Tensor & tensor);

    torch::Tensor from_batched(const torch::Tensor & tensor);

    void imshow(const std::string &window_name, const torch::Tensor &tensor, bool add_shape = false, bool normalize = false);

    void palette_show(const std::string &window_name, const torch::Tensor &palette, bool  add_shape = false);

    void compare_show(const std::string &window_name, const torch::Tensor &a, const torch::Tensor &b, bool add_shape = false, bool normalize = false);

    void count_assignments(const std::string &prefix, torch::Tensor &M);

    bool is_close(float a, float b, float atol=1e-8, float rtol=1e-5);
}

#endif //DCQ_UTILS_H
