//
// Created by thesh on 12/07/2023.
//

#ifndef DCQ_FLOYD_STEINBERG_H
#define DCQ_FLOYD_STEINBERG_H

#include <torch/torch.h>
#include "init.h"
#include "types.h"


namespace dcq::fs {
    torch::Tensor median_cuts(const torch::Tensor &X, int palette_size);

    dcq::Parameters solve(const torch::Tensor &X, int palette_size);
}

#endif //DCQ_FLOYD_STEINBERG_H
