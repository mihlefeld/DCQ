//
// Created by thesh on 28/06/2023.
//

#ifndef DCQ_ALGORITHM_H
#define DCQ_ALGORITHM_H

#include "types.h"

namespace dcq::algorithm {
    dcq::Parameters solve(torch::Tensor &X, int ks, int max_K);

    dcq::Parameters icm(dcq::LConst &constants, dcq::Parameters &params,
                        dcq::Kernels &kernels, torch::Tensor &p);

    void add_color(torch::Tensor &X, dcq::Parameters &params);

    int compute_assignments(dcq::Parameters &params, dcq::LConst &constants, dcq::Kernels &kernels, torch::Tensor &p);

    void compute_colors(dcq::Parameters &params, torch::Tensor &b, torch::Tensor &a);

    bool update_M(int iy, int ix, int *M_data, float *Y_data, float *p_data, float *bii_data,
                  int ks, int h, int w, int c, int K);

    void update_p(int iy, int ix, int *M_data, float *Y_data, float *a_data, float *b0_data, float *p_data,
                  int ks, int h, int w, int c, int K);

    torch::Tensor compute_p(dcq::Parameters &params, torch::Tensor &a, torch::Tensor &b);

    float compute_loss(torch::Tensor &X, dcq::Parameters &params, torch::Tensor &W);

    void propagate_M(dcq::Parameters &params);

    dcq::Region get_neighborhood(int iy, int ix, int ks, torch::Tensor &a);
}

#endif //DCQ_ALGORITHM_H
