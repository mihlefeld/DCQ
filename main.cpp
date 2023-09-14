#include <opencv2/imgcodecs.hpp>
#include "dcq/algorithm.h"
#include "dcq/utils.h"
#include "dcq/init.h"
#include "dcq/floyd_steinberg.h"
#include "dcq/alpha.h"
#include <omp.h>
#include <filesystem>

namespace fs = std::filesystem;

/*
 * Program expects system arguments in specified order:
 * input_path
 * output_path
 * kernel_size: size of the kernel used for the Dithered Color Quantization approach
 * palette_size: maximum number of colors to use
 * mode: how to quantize the image, one of the following: DCQ, FS, FS-ICM, FS-ICM-DCQ, FS-DCQ
 * alpha_mode: how to handle the alpha channel, one of: quantize, divide, none
 * cluster_mode: how to compute the starting palette for Floyd-Steinberg: k_means, median_cuts
 */
int main(int argc, char *argv[]) {
    int ks = 3;
    int palette = 5;
    std::string mode = "DCQ";
    std::string alpha_mode = "noop";
    std::string cluster_mode = "kmeans";
    if (argc < 3) {
        std::cout << "Not enough arguments" << std::endl;
        return 1;
    }
    auto input_path = fs::path(argv[1]);
    auto output_path = fs::path(argv[2]);
    if (argc >= 4) {
        ks = std::stoi(argv[3]);
    }
    if (argc >= 5) {
        palette = std::stoi(argv[4]);
    }
    if (argc >= 6) {
        mode = argv[5];
    }
    if (argc >= 7) {
        alpha_mode = argv[6];
    }
    if (argc >= 8) {
        cluster_mode = argv[7];
    }

    auto name = input_path.filename().string();
    name = name.substr(0, name.size() - 4);

    if (!fs::is_directory(output_path)) {
        fs::create_directories(output_path);
    }

    auto img = cv::imread(input_path.string(), cv::IMREAD_UNCHANGED);
    std::cout << "Reading image " << input_path << "with size " << img.rows << "x" << img.cols << "x" << img.channels()
              << std::endl;
    auto tensor_ = dcq::utils::image_to_tensor(img);
    dcq::alpha::preprocess(tensor_, alpha_mode);
    auto tensor = tensor_.clone();
    dcq::Parameters params;

    if (mode == "DCQ") {
        params = dcq::algorithm::solve(tensor, ks, palette);
    }

    if (mode == "FS") {
        params = dcq::fs::solve(tensor, palette, cluster_mode);
    }

    if (mode == "FS-ICM") {
        params = dcq::fs::solve_icm(tensor, palette, cluster_mode);
    }

    if (mode == "FS-ICM-DCQ") {
        params = dcq::fs::solve_icm(tensor, palette, cluster_mode);
        params = dcq::algorithm::solve(tensor, ks, palette, params);
    }

    if (mode == "FS-DCQ") {
        params = dcq::fs::solve(tensor, palette, cluster_mode);
        params = dcq::algorithm::solve(tensor, ks, palette, params);
    }

    auto result = params.reconstruct();
    dcq::alpha::postprocess(result, alpha_mode);
    auto recon = dcq::utils::tensor_to_image(result);
    cv::imwrite((output_path / (name + ".png")).string(), recon);
    return 0;
}
