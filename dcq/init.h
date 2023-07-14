//
// Created by thesh on 28/06/2023.
//

#ifndef DCQ_INIT_H
#define DCQ_INIT_H

#include <torch/torch.h>
#include "types.h"

namespace dcq::init {
    dcq::Parameters init_parameters(const torch::Tensor &Xl, const torch::Tensor &Xs);
    dcq::Parameters init_parameters(const torch::Tensor &X);

    dcq::Kernels init_kernels(int ks, int c);

    dcq::Constants init_constants(const torch::Tensor &X, const torch::Tensor &b, int max_K);
}

#endif //DCQ_INIT_H
