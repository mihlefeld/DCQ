//
// Created by thesh on 28/06/2023.
//

#ifndef DCQ_TYPES_H
#define DCQ_TYPES_H
#include <torch/torch.h>
#include <vector>
using namespace torch::indexing;

namespace dcq {
    struct Kernels {
        torch::Tensor W;
        torch::Tensor b;
        torch::Tensor b0;
    };

    struct Constants {
        std::vector<torch::Tensor> Xs;
        std::vector<torch::Tensor> as;
        torch::Tensor max_Ks;
        int original_h;
        int original_w;
        int max_K;
        int max_level;
    };

    struct LConst {
        torch::Tensor X;
        torch::Tensor a;
        int max_K;
    };

    struct Parameters {
        torch::Tensor M;
        torch::Tensor Y;

        torch::Tensor reconstruct() {
            return torch::einsum("hwk,kc->hwc", {M, Y});
        }
    };

    struct Region {
        int t;
        int b;
        int l;
        int r;
    };
}

#endif //DCQ_TYPES_H
