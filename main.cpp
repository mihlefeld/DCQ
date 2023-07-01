#include <opencv2/imgcodecs.hpp>
#include "dcq/algorithm.h"
#include "dcq/utils.h"
#include "dcq/init.h"
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
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

    auto img = cv::imread(input_path.string());
    auto tensor = dcq::utils::image_to_tensor(img);

    dcq::Parameters params;
    TIME_IT(params = dcq::algorithm::solve(tensor, ks, palette);)
    auto recon = dcq::utils::tensor_to_image(params.reconstruct());
    cv::imwrite((output_path / (name + ".png")).string(), recon);
    return 0;
}
