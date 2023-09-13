//
// Created by thesh on 23/08/2023.
//

#ifndef DCQ_ALPHA_H
#define DCQ_ALPHA_H

#include <torch/torch.h>

namespace dcq::alpha {
    void preprocess(torch::Tensor &X, const std::string &mode);
    void postprocess(torch::Tensor &X, const std::string &mode);
}

#endif //DCQ_ALPHA_H
