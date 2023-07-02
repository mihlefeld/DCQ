#include <opencv2/imgcodecs.hpp>
#include "dcq/algorithm.h"
#include "dcq/utils.h"
#include "dcq/init.h"
#include <filesystem>

namespace fs = std::filesystem;

void test_update_M() {
    int iy = 4;
    int ix = 3;
    int K = 2;
    int h = 5;
    int w = 5;
    int c = 3;
    int ks = 5;

    auto kernels = dcq::init::init_kernels(3, c);
    auto M = torch::zeros({h, w}, torch::kInt32);
    M.index_put_({Slice(1, None, 2)}, 0);
    M.index_put_({Slice(None, None, 2)}, 1);
    auto Y = torch::tensor({0.25, 0.5, 0.75, 0.8, 0.4, 0.0}, torch::kFloat32);
    Y = Y.reshape({K, c});
    auto p = torch::tensor(
            {
                    -0.41, -1.38, -0.50, -0.50, 0.22, -0.38, -0.71, -1.29, -0.30, -0.62, -0.19, 0.09, -0.80, -0.81,
                    0.38, -0.53, -1.03, -0.11, -0.40, -0.57, -1.37, -0.02, 0.36, -0.74, -0.97, -0.85, -0.15, -1.08,
                    -1.13, -0.55, 0.25, -0.82, 0.08, -0.36, 0.31, -1.10, -0.92, -0.39, -0.76, 0.18, -0.05, -0.18, -0.34,
                    -0.77, -0.05, 0.09, -0.20, 0.22, -1.25, -1.33, 0.48, -1.09, 0.36, -0.39, 0.30, -1.20, -0.91, 0.46,
                    -1.24, -0.18, -0.00, 0.39, 0.28, -1.00, -0.76, -0.22, 0.35, -0.12, 0.25, -0.77, -0.45, -1.16, -0.03,
                    -0.88, -1.04
            }, torch::kFloat32);
    p = p.reshape({h, w, c});
    auto a = torch::tensor(
            {
                    -1.21, -0.16, -1.67, -0.52, -0.38, -0.78, -0.56, -0.74, -0.51, -1.18, -1.23, -1.89, -1.82, -1.50,
                    -1.08, -0.47, -1.54, -1.49, -1.11, -0.66, -0.14, -1.15, -1.30, -0.59, -1.86, -1.39, -1.01, -1.20,
                    -0.71, -0.93, -1.19, -1.42, -1.21, -0.51, -0.50, -1.05, -0.37, -1.67, -1.41, -1.27, -1.60, -1.27,
                    -0.37, -0.15, -0.58, -0.78, -0.67, -0.79, -0.42, -0.33, -1.46, -0.48, -1.23, -1.88, -0.22, -0.26,
                    -1.72, -1.48, -1.14, -0.76, -0.55, -1.90, -0.65, -1.57, -1.88, -1.05, -0.18, -0.67, -1.55, -1.55,
                    -1.11, -0.35, -1.20, -0.24, -0.59
            }, torch::kFloat32);
    auto x = torch::tensor(
            {
                    0.0721, 0.1843, 0.8856, 0.1182, 0.4976, 0.1564, 0.2443, 0.0237, 0.2811, 0.1920, 0.4230, 0.1428,
                    0.3455, 0.3952, 0.7878, 0.8015, 0.1011, 0.4480, 0.3168, 0.9081, 0.6837, 0.2651, 0.6752, 0.5815,
                    0.5762, 0.4514, 0.6201, 0.3511, 0.7690, 0.5081, 0.9407, 0.2130, 0.2319, 0.4279, 0.5215, 0.5252,
                    0.1774, 0.0248, 0.9875, 0.6560, 0.2960, 0.3006, 0.8096, 0.8147, 0.4144, 0.2833, 0.4363, 0.8448,
                    0.7329, 0.0499, 0.4984, 0.4632, 0.6147, 0.6515, 0.6173, 0.1583, 0.8844, 0.5403, 0.5415, 0.6337,
                    0.8666, 0.6863, 0.4975, 0.9175, 0.9149, 0.3938, 0.3141, 0.1558, 0.2654, 0.8294, 0.8587, 0.7454,
                    0.7688, 0.3770, 0.8897
            }, torch::kFloat32
    );
    x = x.reshape({h, w, c});
    a = a.reshape({h, w, c});

    std::cout << kernels.b.index({Slice(), Slice(), 0}) << std::endl;
    std::cout << kernels.b0.index({Slice(), Slice(), 0}) << std::endl;

    auto M_data = M.data_ptr<int>();
    auto Y_data = Y.data_ptr<float>();
    auto bii_data = kernels.b.index({ks / 2, ks / 2}).data_ptr<float>();
    auto a_data = a.data_ptr<float>();
    auto p_data = p.data_ptr<float>();
    auto b0_data = kernels.b0.data_ptr<float>();

    auto changed = dcq::algorithm::update_M(iy, ix, M_data, Y_data, p_data, bii_data, ks, h, w, c, K);

    dcq::algorithm::update_p(iy, ix, M_data, Y_data, a_data, b0_data, p_data, ks, h, w, c, K);

    std::cout << "changed=" << changed << std::endl;

    std::cout << "updated_M" << std::endl;
    std::cout << M << std::endl;

    std::cout << "update_p 0" << std::endl;
    std::cout << p.index({Slice(), Slice(), 0}) << std::endl;
    std::cout << "update_p 1" << std::endl;
    std::cout << p.index({Slice(), Slice(), 1}) << std::endl;
    std::cout << "update_p 2" << std::endl;
    std::cout << p.index({Slice(), Slice(), 2}) << std::endl;

    dcq::Parameters params{M, Y};
    dcq::algorithm::compute_colors(params, kernels.b, a);
    std::cout << params.Y << std::endl;

    dcq::algorithm::add_color(x, params);

    std::cout << "M argmax" << std::endl;
    std::cout << params.M << std::endl;

    std::cout << "Y added" << std::endl;
    std::cout << params.Y << std::endl;

}

int main(int argc, char *argv[]) {
//    test_update_M();
    int ks = 3;
    int palette = 5;
    if (argc < 3) {
        std::cout << "Not enough arguments" << std::endl;
        return 1;
    }
    auto input_path = fs::path(argv[1]);
    auto output_path = fs::path(argv[2]);
    ks = std::stoi(argv[3]);
    palette = std::stoi(argv[4]);

    auto name = input_path.filename().string();
    name = name.substr(0, name.size() - 4);

    if (!fs::is_directory(output_path)) {
        fs::create_directories(output_path);
    }

    auto img = cv::imread(input_path.string(), cv::IMREAD_UNCHANGED);
    std::cout << "Reading image " << input_path << "with size " << img.rows << "x" << img.cols << "x" << img.channels()
              << std::endl;
    auto tensor = dcq::utils::image_to_tensor(img);

    dcq::Parameters params;
    params = dcq::algorithm::solve(tensor, ks, palette);
    dcq::utils::count_assignments("Final Assignments", params.M, params.Y.size(0));
    auto recon = dcq::utils::tensor_to_image(params.reconstruct());
    cv::imwrite((output_path / (name + ".png")).string(), recon);
    return 0;
}
