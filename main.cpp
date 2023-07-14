#include <opencv2/imgcodecs.hpp>
#include "dcq/algorithm.h"
#include "dcq/utils.h"
#include "dcq/init.h"
#include "dcq/floyd_steinberg.h"
#include <omp.h>
#include <filesystem>

namespace fs = std::filesystem;


int main(int argc, char *argv[]) {
    int ks = 3;
    int palette = 5;
    std::string mode = "DCQ";
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

    if (mode == "DCQ") {
        params = dcq::algorithm::solve(tensor, ks, palette);
    }

    if (mode == "FS") {
        params = dcq::fs::solve(tensor, palette);
    }

    if (mode == "FS-DCQ") {
        params = dcq::fs::solve(tensor, palette);
        params = dcq::algorithm::solve(tensor, ks, palette, params);
    }

    auto recon = dcq::utils::tensor_to_image(params.reconstruct());
    cv::imwrite((output_path / (name + ".png")).string(), recon);
    return 0;
}
