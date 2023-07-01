//
// Created by thesh on 29/06/2023.
//
#include "algorithm.h"
#include "init.h"
#include "utils.h"
#include <algorithm>

using namespace std::chrono;
using namespace torch::indexing;
namespace F = torch::nn::functional;

dcq::Parameters dcq::algorithm::solve(
        torch::Tensor &X,
        int ks,
        int max_K
) {
    auto pbar = PBar(2, 0);
    int c = X.size(2);
    auto kernels = dcq::init::init_kernels(ks, c);
    auto constants = dcq::init::init_constants(X, kernels.b, max_K);
    auto params = dcq::init::init_parameters(constants.Xs.front(), constants.Xs.back());
    pbar.max_L = constants.max_level + 1;
    pbar.max_K = constants.max_K;

    for (int i = constants.max_level; i >= 0; i--) {
        LConst lconst{constants.Xs.at(i), constants.as.at(i),
                      constants.max_Ks.index({i}).item<int>()};
        pbar.l = constants.max_level + 1 - i;
        pbar.update();
        auto p = compute_p(params, lconst.a, kernels.b0);

        do {
            if (params.Y.size(0) < lconst.max_K) {
                pbar.start(pbar_timers::ADD_TIMER);
                add_color(lconst.X, params);
                pbar.K = params.Y.size(0);
                pbar.stop(pbar_timers::ADD_TIMER);
            }
            icm(lconst, params, kernels, p, pbar);
        } while (params.Y.size(0) < lconst.max_K);
        if (i != 0) {
            propagate_M(params);
        }
    }

    int h = params.M.size(0);
    int w = params.M.size(1);
    int diff_h = h - constants.original_h;
    int diff_w = w - constants.original_w;
    int top = diff_h / 2;
    int bottom = h - (diff_h - top);
    int left = diff_w / 2;
    int right = w - (diff_w - left);

    auto sub_M = params.M.index({Slice(top, bottom), Slice(left, right)});
    return {sub_M, params.Y};
}

dcq::Parameters dcq::algorithm::icm(
        dcq::LConst &constants,
        dcq::Parameters &params,
        dcq::Kernels &kernels,
        torch::Tensor &p,
        PBar &pbar
) {
    int h = params.M.size(0);
    int w = params.M.size(1);
    int K = params.Y.size(0);
    float loss = compute_loss(constants.X, params, kernels.W);
    float old_loss = INFINITY;
    int changed;
    do {
        old_loss = loss;

        pbar.start(pbar_timers::ASSIGN_TIMER);
        changed = compute_assignments(params, constants, kernels, p);
        pbar.stop(pbar_timers::ASSIGN_TIMER);

        pbar.start(pbar_timers::COMPUTE_TIMER);
        compute_colors(params, kernels.b, constants.a);
        pbar.stop(pbar_timers::COMPUTE_TIMER);

        loss = compute_loss(constants.X, params, kernels.W);
        pbar.loss = loss;
        pbar.update();
    } while (changed > 0 && loss < old_loss);
    return params;
}

void dcq::algorithm::add_color(
        torch::Tensor &X,
        dcq::Parameters &params
) {
    int K = params.Y.size(0);
    auto new_M = params.M.clone();
    auto new_Y = F::pad(params.Y.index({None}), F::PadFuncOptions({0, 0, 0, 1})).index({0});

    auto comp = torch::arange(K).reshape({1, 1, K});
    auto one_hot_M = params.M.index({Slice(), Slice(), None}) == comp;

    auto MY = params.reconstruct();
    auto diff = (X - MY).pow(2).sum({-1}, true);
    auto distortions = (diff * one_hot_M).sum({1, 0});
    auto v = distortions.argmax().item<int>();

    auto bool_M = params.M == v;
    auto colors = X.index({bool_M});
    auto center_a_idx = diff.index({bool_M}).argmax();
    auto center_b_idx = diff.index({bool_M}).argmin();

    auto center_a = colors.index({center_a_idx});
    auto center_b = colors.index({center_b_idx});

    auto center_diff_a = (center_a.index({None, None}) - X).pow(2).sum({-1});
    auto center_diff_b = (center_b.index({None, None}) - X).pow(2).sum({-1});
    auto index = ((center_diff_b <= center_diff_a) * bool_M) == 1;

    new_M.index_put_({index}, K);

    auto v_sum = X.index({new_M == v}).sum(0);
    auto v_mean = v_sum / (new_M == v).sum();
    auto k_sum = X.index({new_M == K}).sum(0);
    auto k_mean = k_sum / (new_M == K).sum();

    new_Y.index_put_({v}, v_mean);
    new_Y.index_put_({K}, k_mean);

    params.M = new_M;
    params.Y = new_Y;
}

int dcq::algorithm::compute_assignments(
        dcq::Parameters &params,
        dcq::LConst &constants,
        dcq::Kernels &kernels,
        torch::Tensor &p
) {
    int ks = kernels.b.size(0);
    int ksh = ks / 2;
    auto M_data = params.M.data_ptr<int>();
    auto Y_data = params.Y.data_ptr<float>();
    auto bii_data = kernels.b.index({ks / 2, ks / 2}).data_ptr<float>();
    auto a_data = constants.a.data_ptr<float>();
    auto p_data = p.data_ptr<float>();
    auto b0_data = kernels.b0.data_ptr<float>();

    int h = p.size(0);
    int w = p.size(1);
    int c = p.size(2);
    int K = params.Y.size(0);
    int changed = 0;

    auto changed_map = new bool[h * w];

    for (int i = 0; i < h * w; i++) {
        changed_map[i] = true;
    }

    int total_changed = 0;
    do {
        changed = 0;
        for (int iy = 0; iy < h - 1; iy++) {
            for (int ix = 0; ix < w - 1; ix++) {
                if (!changed_map[iy * w + ix]) continue;

                if (update_M(iy, ix, M_data, Y_data, p_data, bii_data, ks, h, w, c, K)) {
                    update_p(iy, ix, M_data, Y_data, a_data, b0_data, p_data, ks, h, w, c, K);
                    changed += 1;
                    int ktop = std::max(iy - ksh, 0);
                    int kbottom = std::min(iy + ksh, h - 1) + 1;
                    int kleft = std::max(ix - ksh, 0);
                    int kright = std::min(ix + ksh, w - 1) + 1;

                    for (int ky = ktop; ky < kbottom; ky++) {
                        for (int kx = kleft; kx < kright; kx++) {
                            changed_map[ky * w + kx] = true;
                        }
                    }
                }
                changed_map[iy * w + ix] = false;
            }
        }
        total_changed += changed;
    } while (changed > 0);

    delete[] changed_map;

    return total_changed;
}

void dcq::algorithm::compute_colors(
        dcq::Parameters &params,
        torch::Tensor &b,
        torch::Tensor &a
) {
    int K = params.Y.size(0);
    int c = params.Y.size(1);
    int h = params.M.size(0);
    int w = params.M.size(1);
    int ks = b.size(0);
    int pad = b.size(0) / 2;

    auto Sn = torch::zeros({K, K, c});
    auto Rn = torch::zeros({K, c});
    int *M_data = params.M.data_ptr<int>();
    float *a_data = a.data_ptr<float>();
    float *b_data = b.data_ptr<float>();
    float *S_data = Sn.data_ptr<float>();
    float *R_data = Rn.data_ptr<float>();

    for (int iy = 0; iy < h; iy++) {
        for (int ky = 0; ky < ks; ky++) {
            for (int ix = 0; ix < w; ix++) {
                int v = M_data[iy * w + ix];
                int kiy = iy + ky - ks / 2;
                for (int kx = 0; kx < ks; kx++) {
                    int kix = ix + kx - ks / 2;
                    if (kiy < 0 || kix < 0 || kiy >= h || kix >= w) continue;
                    int al = M_data[kiy * w + kix];
                    float *bxy = &b_data[ky * ks * c + kx * c];
                    for (int ci = 0; ci < c; ci++) {
                        S_data[v * K * c + al * c + ci] += bxy[ci];
                    }
                }
            }
        }
    }

    for (int iy = 0; iy < h; iy++) {
        for (int ix = 0; ix < w; ix++) {
            int v = M_data[iy * w + ix];
            for (int ci = 0; ci < c; ci++) {
                R_data[v * c + ci] += a_data[iy * w * c + ix * c + ci];
            }
        }
    }

    auto new_Y = torch::linalg::solve(-2 * Sn.permute({2, 0, 1}), Rn.permute({1, 0}), true);
    params.Y = new_Y.permute({1, 0}).flatten().reshape({K, c});
}


bool dcq::algorithm::update_M(int iy, int ix, int *M_data, float *Y_data, float *p_data, float *bii_data,
                              int ks, int h, int w, int c, int K) {
    float *pi = &p_data[iy * w * c + ix * c];
    int *mi = &M_data[iy * w + ix];
    float min = INFINITY;
    int amin = 0;
    for (int v = 0; v < K; v++) {
        float *yv = &Y_data[v * c];
        float sum = 0;
        for (int ci = 0; ci < c; ci++) {
            float pic = pi[ci];
            float biic = bii_data[ci];
            float yvc = yv[ci];
            sum += yvc * (pic + biic * yvc);
        }
        if (sum < min) {
            min = sum;
            amin = v;
        }
    }
    bool changed = *mi != amin;
    *mi = amin;
    return changed;
}


void
dcq::algorithm::update_p(int iy, int ix, int *M_data, float *Y_data, float *a_data, float *b0_data, float *p_data,
                         int ks, int h, int w, int c, int K) {
    int ksh = ks / 2;
    int top = std::max(iy - ksh, 0);
    int bottom = std::min(iy + ksh, h - 1) + 1;
    int left = std::max(ix - ksh, 0);
    int right = std::min(ix + ksh, w - 1) + 1;

    float sum[4];

    for (int i0 = top; i0 < bottom; i0++) {
        for (int i1 = left; i1 < right; i1++) {

            for (int ci = 0; ci < c; ci++) {
                sum[ci] = 0;
            }

            for (int ki = 0; ki < ks; ki++) {
                int iy = i0 + ksh - ki;
                for (int kj = 0; kj < ks; kj++) {
                    int ix = i1 + ksh - kj;
                    if (ix < 0 || iy < 0 || iy >= h || ix >= w) {
                        continue;
                    }
                    int amax = M_data[iy * w + ix];
                    float *MY = &Y_data[amax * c];
                    float *b0 = &b0_data[ki * ks * c + kj * c];
                    for (int ci = 0; ci < c; ci++) {
                        sum[ci] += MY[ci] * b0[ci];
                    }
                }
            }

            for (int ci = 0; ci < c; ci++) {
                p_data[i0 * w * c + i1 * c + ci] = 2 * sum[ci] + a_data[i0 * w * c + i1 * c + ci];
            }
        }
    }

}

torch::Tensor dcq::algorithm::compute_p(
        dcq::Parameters &params,
        torch::Tensor &a,
        torch::Tensor &b0
) {
    int c = params.Y.size(1);
    auto conved = dcq::utils::from_batched(
            F::conv2d(
                    dcq::utils::to_batched(params.reconstruct()),
                    dcq::utils::to_conv_kernel(b0),
                    F::Conv2dFuncOptions().groups(c).padding(torch::kSame)
            )
    );
    return 2 * conved.index({0}) + a;
}

float dcq::algorithm::compute_loss(
        torch::Tensor &X,
        dcq::Parameters &params,
        torch::Tensor &W
) {
    auto kernel = dcq::utils::to_conv_kernel(W);
    int c = params.Y.size(1);
    auto pred = F::conv2d(
            dcq::utils::to_batched(params.reconstruct()),
            kernel,
            F::Conv2dFuncOptions().groups(c).padding(torch::kSame)
    );
    auto gt = F::conv2d(
            dcq::utils::to_batched(X),
            kernel,
            F::Conv2dFuncOptions().groups(c).padding(torch::kSame)
    );
    return (gt - pred).pow(2).mean().item<float>();
}

void dcq::algorithm::propagate_M(dcq::Parameters &params) {
    int h = params.M.size(0);
    int w = params.M.size(1);
    auto new_M = torch::zeros({h * 2, w * 2}, torch::kInt32);

    new_M.index_put_({Slice(None, None, 2), Slice(None, None, 2)}, params.M);
    new_M.index_put_({Slice(None, None, 2), Slice(1, None, 2)}, params.M);
    new_M.index_put_({Slice(1, None, 2), Slice(None, None, 2)}, params.M);
    new_M.index_put_({Slice(1, None, 2), Slice(1, None, 2)}, params.M);

    params.M = new_M;
}