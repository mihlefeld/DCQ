//
// Created by thesh on 23/08/2023.
//

#include "alpha.h"
#include <torch/torch.h>
using namespace torch::indexing;

void quantize_alpha(torch::Tensor &X) {
    if (X.size(-1) != 4) {
        return;
    }

    auto alpha_values = X.index({Slice(), Slice(), -1});
    auto unique_counts = torch::unique_dim(alpha_values.flatten(), -1, true, false, true);
    auto unique = std::get<0>(unique_counts);
    auto counts = std::get<2>(unique_counts);

    if (unique.size(0) < 4) {
        return;
    }

    auto sorted = std::get<0>(torch::sort(alpha_values.flatten()));
    auto zeros = counts.index({0}).item<long>();
    auto ones = counts.index({-1}).item<long>();
    auto n = sorted.size(0) - zeros - ones;
    auto colors = torch::linspace(0, 1, 4);

    for (int i = 1; i < 3; i++) {
        auto sorted_index = (long) ((float) i / 4.0 * ((float) n - 1)) + zeros;
        colors.index_put_({i}, sorted.index({sorted_index}));
    }

    auto color_index = (alpha_values.index({Slice(), Slice(), None}) - colors.index({None, None})).abs().argmin(-1);
    X.index_put_({Slice(), Slice(), -1}, colors.index({color_index}));
}

void divide_alpha(torch::Tensor &X) {
    X.index({Slice(), Slice(), -1}) /= 10;
}

void multiply_alpha(torch::Tensor &X) {
    X.index({Slice(), Slice(), -1}) *= 10;
}

void dcq::alpha::preprocess(torch::Tensor &X, const std::string &mode) {
    if (X.size(-1) < 4) {
        return;
    }
    if (mode == "quantize") {
        quantize_alpha(X);
        return;
    }
    if (mode == "divide") {
        divide_alpha(X);
        return;
    }

}

void dcq::alpha::postprocess(torch::Tensor &X, const std::string &mode) {
    if (X.size(-1) < 4) {
        return;
    }
    if (mode == "divide") {
        multiply_alpha(X);
        return;
    }

}