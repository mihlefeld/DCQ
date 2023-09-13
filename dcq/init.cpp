//
// Created by thesh on 28/06/2023.
//

#include "init.h"
#include <cmath>
#include <torch/torch.h>
#include <iostream>
#include "utils.h"

using namespace torch::indexing;
namespace F = torch::nn::functional;

dcq::Parameters dcq::init::init_parameters(const torch::Tensor &Xl, const torch::Tensor &Xs) {
    int h = Xs.size(0);
    int w = Xs.size(1);
    int c = Xs.size(2);
    auto M = torch::zeros({h, w}, torch::kInt32);
    auto Y = Xl.mean({1, 0}).reshape({1, c});
    return {M, Y};
}

dcq::Parameters dcq::init::init_parameters(const torch::Tensor &X) {
    return init_parameters(X, X);
}

dcq::Kernels dcq::init::init_kernels(int ks, int c) {
    int ksh = ks / 2;
    float sigma_squared = -1 / std::log(0.125);
    auto W = torch::zeros({ks, ks, c});
    for (int chan = 0; chan < c; chan++) {
        for (int i = 0; i < ks; i++) {
            for (int j = 0; j < ks; j++) {
                float iks = std::pow(i - ksh, 2);
                float jks = std::pow(j - ksh, 2);
                float dist = std::sqrt(iks + jks);
                float gaussian = std::exp(-dist / sigma_squared);
                W.index_put_({i, j, chan}, gaussian);
            }
        }
    }

    W /= W.sum({1, 0}).index({None, None});

    auto b = F::conv2d(dcq::utils::to_batched(W), dcq::utils::to_conv_kernel(W),
                       F::Conv2dFuncOptions().groups(c).padding({ks - 1, ks - 1})
    );
    b = dcq::utils::from_batched(b);
    auto b0 = b.clone();
    b0.index_put_({ks - 1, ks - 1}, 0);

    return {W, b, b0};
}

dcq::Constants dcq::init::init_constants(const torch::Tensor &X, const torch::Tensor &b, int max_K) {
    int h = X.size(0);
    int w = X.size(1);
    int c = X.size(2);
    int kernel_pad = b.size(0) / 2;
    int hk = h + 2 * kernel_pad;
    int wk = w + 2 * kernel_pad;
    int ksh = b.size(0) / 2;

    auto colors_mixed = X.index({Slice(), Slice(), 0}).clone();
    for (int i = 1; i < c; i++) {
        colors_mixed += std::pow(10, i) * X.index({Slice(), Slice(), i});
    }
    auto unique_colors = std::get<0>(torch::_unique(colors_mixed));
    int max_K_reduced = std::min((int) unique_colors.size(0), max_K);

    // compute the maximal number of max_level possible without making the image too small
    auto max_level = (int) std::floor(std::log2(std::min(hk, wk) / 5));

    // construct geometric space from logspace
    auto space = torch::logspace(std::log2(max_K_reduced), std::log2(2), max_level + 1, 2);
    space = space.floor().to(torch::kInt32);

    // pad the image
    int div = 2 << max_level;
    int new_h = (int) (std::ceil((float) hk / div) * div);
    int new_w = (int) (std::ceil((float) wk / div) * div);
    int diff_h = new_h - h;
    int diff_w = new_w - w;
    int pad_top = diff_h / 2;
    int pad_bottom = diff_h - pad_top;
    int pad_left = diff_w / 2;
    int pad_right = diff_w - pad_left;

    auto X_pad = F::pad(X.index({None}),
                        F::PadFuncOptions({0, 0, pad_left, pad_right, pad_top, pad_bottom})
                                .mode(torch::kReflect)
    ).index({0});
    auto a = dcq::utils::from_batched(
            -2 * F::conv2d(dcq::utils::to_batched(X_pad), dcq::utils::to_conv_kernel(b),
                           F::Conv2dFuncOptions().padding({ksh, ksh}).groups(c))
    );

    std::vector<torch::Tensor> Xs;
    std::vector<torch::Tensor> as;
    Xs.push_back(X_pad);
    as.push_back(a);

    for (int i = 0; i < max_level; i++) {
        auto last_X = Xs.back();
        auto last_a = as.back();
        auto new_X = dcq::utils::from_batched(
                F::avg_pool2d(dcq::utils::to_batched(last_X), F::AvgPool2dFuncOptions({2, 2}))
        );
        auto new_a = dcq::utils::from_batched(
                F::avg_pool2d(dcq::utils::to_batched(last_a), F::AvgPool2dFuncOptions({2, 2}))
        );
        Xs.push_back(new_X);
        as.push_back(new_a);
    }
    return {Xs, as, space, h, w, max_K_reduced, max_level};
}