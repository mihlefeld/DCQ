//
// Created by thesh on 12/07/2023.
//

#ifndef DCQ_FLOYD_STEINBERG_H
#define DCQ_FLOYD_STEINBERG_H

#include <torch/torch.h>
#include "PBar.h"
#include "init.h"
#include "types.h"


namespace dcq::fs {
    torch::Tensor median_cuts(const torch::Tensor &X, int palette_size, dcq::PBar &pbar);

    torch::Tensor k_means(const torch::Tensor &X, int palette_size, dcq::PBar &pbar);

    torch::Tensor cluster(const torch::Tensor &X, int palette_size, const std::string &mode, dcq::PBar &pbar);

    dcq::Parameters solve(const torch::Tensor &X, int palette_size, const std::string &mode);

    dcq::Parameters solve_icm(torch::Tensor &X, int palette_size, const std::string &mode);
}

#endif //DCQ_FLOYD_STEINBERG_H
