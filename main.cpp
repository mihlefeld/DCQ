#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <filesystem>
#include <torch/torch.h>
#include "utils/utils.h"

namespace fs = std::filesystem;

int main() {
    int h = 500;
    int w = 900;
    
    auto img = cv::imread("../_data/pictures/natural/blue macaw back turned.jpg");
    auto tensor = dcq::utils::image_to_tensor(img);

    tensor = tensor.pow(2);
    tensor /= tensor.max() - tensor.min();
    tensor = tensor.clip(0, 1);

    auto image = dcq::utils::tensor_to_image(tensor);
    cv::imshow("Test", image);
    cv::waitKey(0);

    return 0;
}
