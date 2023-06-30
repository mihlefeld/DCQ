#include <opencv2/imgcodecs.hpp>
#include "dcq/algorithm.h"
#include "dcq/utils.h"
#include "dcq/init.h"


int test_M() {
    auto M = torch::zeros({5, 5, 2}, torch::kFloat32);
    M.index_put_({Slice(None, None, 2), Slice(), 0}, 1);
    M.index_put_({Slice(1, None, 2), Slice(), 1}, 1);

    std::cout << "M[:, :, 0]" << std::endl;
    std::cout << M.index({Slice(), Slice(), 0}) << std::endl;
    std::cout << "M[:, :, 1]" << std::endl;
    std::cout << M.index({Slice(), Slice(), 1}) << std::endl;
    auto Y = torch::tensor({0.25, 0.5, 0.75, 0.8, 0.4, 0.0}, torch::kFloat32);
    auto p = torch::tensor(
            {
                    -0.41, -1.38, -0.50, -0.50, 0.22, -0.38, -0.71, -1.29, -0.30, -0.62, -0.19, 0.09, -0.80, -0.81,
                    0.38, -0.53, -1.03, -0.11, -0.40, -0.57, -1.37, -0.02, 0.36, -0.74, -0.97, -0.85, -0.15, -1.08,
                    -1.13, -0.55, 0.25, -0.82, 0.08, -0.36, 0.31, -1.10, -0.92, -0.39, -0.76, 0.18, -0.05, -0.18, -0.34,
                    -0.77, -0.05, 0.09, -0.20, 0.22, -1.25, -1.33, 0.48, -1.09, 0.36, -0.39, 0.30, -1.20, -0.91, 0.46,
                    -1.24, -0.18, -0.00, 0.39, 0.28, -1.00, -0.76, -0.22, 0.35, -0.12, 0.25, -0.77, -0.45, -1.16, -0.03,
                    -0.88, -1.04
            }, torch::kFloat32
    );
    auto a = torch::tensor(
            {
                    -1.21, -0.16, -1.67, -0.52, -0.38, -0.78, -0.56, -0.74, -0.51, -1.18, -1.23, -1.89,
                    -1.82, -1.50, -1.08, -0.47, -1.54, -1.49, -1.11, -0.66, -0.14, -1.15, -1.30, -0.59,
                    -1.86, -1.39, -1.01, -1.20, -0.71, -0.93, -1.19, -1.42, -1.21, -0.51, -0.50, -1.05,
                    -0.37, -1.67, -1.41, -1.27, -1.60, -1.27, -0.37, -0.15, -0.58, -0.78, -0.67, -0.79,
                    -0.42, -0.33, -1.46, -0.48, -1.23, -1.88, -0.22, -0.26, -1.72, -1.48, -1.14, -0.76,
                    -0.55, -1.90, -0.65, -1.57, -1.88, -1.05, -0.18, -0.67, -1.55, -1.55, -1.11, -0.35,
                    -1.20, -0.24, -0.59
            }, torch::kFloat32);
    Y = Y.reshape({2, 3});
    a = a.reshape({5, 5, 3});
    p = p.reshape({5, 5, 3});
    bool changed = false;
    {
        auto kernels = dcq::init::init_kernels(3, 3);
        auto M_data = M.data_ptr<float>();
        auto Y_data = Y.data_ptr<float>();
        auto bii_data = kernels.b.index({5 / 2, 5 / 2}).data_ptr<float>();
        auto a_data = a.data_ptr<float>();
        auto p_data = p.data_ptr<float>();
        auto b0_data = kernels.b0.data_ptr<float>();
        changed = dcq::algorithm::update_M(2, 2, M_data, Y_data, p_data,
                                                bii_data, 3, 5, 5, 3, 2);

        dcq::algorithm::update_p(2, 2, M_data, Y_data, a_data, b0_data, p_data, 5, 5, 5, 3, 2);
    }

    std::cout << "changed " << changed << std::endl;
    std::cout << "M[:, :, 0]" << std::endl;
    std::cout << M.index({Slice(), Slice(), 0}) << std::endl;
    std::cout << "M[:, :, 1]" << std::endl;
    std::cout << M.index({Slice(), Slice(), 1}) << std::endl;

    std::cout << "P[:, :, 0]" << std::endl;
    std::cout << p.index({Slice(), Slice(), 0}) << std::endl;
    std::cout << "P[:, :, 1]" << std::endl;
    std::cout << p.index({Slice(), Slice(), 1}) << std::endl;
    std::cout << "P[:, :, 2]" << std::endl;
    std::cout << p.index({Slice(), Slice(), 2}) << std::endl;
    return 0;
}

int main() {
//    test_M();
    auto img = cv::imread("../_data/pictures/natural/blue macaw back turned.jpg");
    auto tensor = dcq::utils::image_to_tensor(img);

    auto params = dcq::algorithm::solve(tensor, 3, 8);
    auto recon = dcq::utils::tensor_to_image(params.reconstruct());
    cv::imwrite("../_data/qpictures/blue macaw back turned.png", recon);
    return 0;
}
