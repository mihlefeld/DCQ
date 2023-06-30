//
// Created by thesh on 29/06/2023.
//
#include "algorithm.h"
#include "init.h"
#include "utils.h"
#include <chrono>

using namespace std::chrono;
using namespace torch::indexing;
namespace F = torch::nn::functional;

dcq::Parameters dcq::algorithm::solve(
        torch::Tensor &X,
        int ks,
        int max_K
) {
    int c = X.size(2);
    auto kernels = dcq::init::init_kernels(ks, c);
    auto constants = dcq::init::init_constants(X, kernels.b, max_K);
    auto params = dcq::init::init_parameters(constants.Xs.front(), constants.Xs.back());

    for (int i = constants.max_level; i >= 0; i--) {
        LConst lconst{constants.Xs.at(i), constants.as.at(i),
                      constants.max_Ks.index({i}).item<int>()};
        auto p = compute_p(params, lconst.a, kernels.b0);

        do {
            if (params.M.size(2) < lconst.max_K) {
                add_color(lconst.X, params);
            }
            icm(lconst, params, kernels, p);
            DEBUG_COMPARE(lconst.X, params.reconstruct())
        } while (params.M.size(2) < lconst.max_K);
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
        torch::Tensor &p
) {
    float loss = compute_loss(constants.X, params, kernels.W);
    std::cout << "loss = " << loss << std::endl;
    float old_loss = INFINITY;
    do {
        old_loss = loss;
        TIME_IT(compute_assignments(params, constants, kernels, p));
        compute_colors(params, kernels.b, constants.a);
        loss = compute_loss(constants.X, params, kernels.W);
        std::cout << "loss = " << loss << std::endl;
    } while (!dcq::utils::is_close(loss, old_loss) && loss < old_loss);
    return params;
}

void dcq::algorithm::add_color(
        torch::Tensor &X,
        dcq::Parameters &params
) {
    int K = params.M.size(2);
    auto new_M = F::pad(params.M, F::PadFuncOptions({0, 1}));
    auto new_Y = F::pad(params.Y.index({None}), F::PadFuncOptions({0, 0, 0, 1})).index({0});

    auto MY = params.reconstruct();
    auto diff = (X - MY).pow(2).sum({-1}, true);
    auto distortions = (diff * params.M).sum({1, 0});
    auto v = distortions.argmax().item<int>();

    auto bool_M = params.M.index({Slice(), Slice(), v}) == 1;
    auto colors = X.index({bool_M});
    auto center_a_idx = diff.index({bool_M}).argmax();
    auto center_b_idx = diff.index({bool_M}).argmin();

    auto center_a = colors.index({center_a_idx});
    auto center_b = colors.index({center_b_idx});

    auto center_diff_a = (center_a.index({None, None}) - X).pow(2).sum({-1});
    auto center_diff_b = (center_b.index({None, None}) - X).pow(2).sum({-1});
    auto index = ((center_diff_b <= center_diff_a) * params.M.index({Slice(), Slice(), v})) == 1;

    new_M.index({Slice(), Slice(), v}).index_put_({index}, 0);
    new_M.index({Slice(), Slice(), K}).index_put_({index}, 1);

    auto v_sum = X.index({new_M.index({Slice(), Slice(), v}) == 1}).sum(0);
    auto v_mean = v_sum / new_M.index({Slice(), Slice(), v}).sum();
    auto k_sum = X.index({new_M.index({Slice(), Slice(), K}) == 1}).sum(0);
    auto k_mean = k_sum / new_M.index({Slice(), Slice(), K}).sum();

    new_Y.index_put_({v}, v_mean);
    new_Y.index_put_({K}, k_mean);

    params.M = new_M;
    params.Y = new_Y;
}

void dcq::algorithm::compute_assignments(
        dcq::Parameters &params,
        dcq::LConst &constants,
        dcq::Kernels &kernels,
        torch::Tensor &p
) {
    int ks = kernels.b.size(0);
    auto M_data = params.M.data_ptr<float>();
    auto Y_data = params.Y.data_ptr<float>();
    auto bii_data = kernels.b.index({ks / 2, ks / 2}).data_ptr<float>();
    auto a_data = constants.a.data_ptr<float>();
    auto p_data = p.data_ptr<float>();
    auto b0_data = kernels.b0.data_ptr<float>();

    int h = p.size(0);
    int w = p.size(1);
    int c = p.size(2);
    int K = params.M.size(2);


    for (int iy = 0; iy < h; iy++) {
        for (int ix = 0; ix < w; ix++) {
            if (update_M(iy, ix, M_data, Y_data, p_data, bii_data, ks, h, w, c, K)) {
                update_p(iy, ix, M_data, Y_data, a_data, b0_data, p_data, ks, h, w, c, K);
            }
        }
    }
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
    int padding = b.size(1) / 2;
    auto M_pad = F::pad(
            dcq::utils::to_batched(params.M),
            F::PadFuncOptions({padding, padding, padding, padding})
                    .mode(torch::kReflect)
    );
    auto b_rep = dcq::utils::to_conv_kernel(b).repeat({K, 1, 1, 1});
    auto M_alpha = F::conv2d(M_pad, b_rep,
                             F::Conv2dFuncOptions().groups(K)).reshape({K, c, h, w});
    auto S = torch::einsum("hwv,achw->cav", {params.M, M_alpha});
    auto R = torch::einsum("hwv,hwc->cv", {params.M, a});
    auto new_Y = torch::linalg::solve(-2 * S, R, true);
    params.Y = new_Y.permute({1, 0}).flatten().reshape({K, c});
}

bool dcq::algorithm::update_M(
        int iy, int ix,
        dcq::Parameters &params,
        dcq::Kernels &kernels,
        torch::Tensor &p
) {
    int ks = kernels.b.size(0);
    auto bii = kernels.b.index({ks / 2, ks / 2, None});
    auto pi = p.index({iy, ix, None});
    auto mingv_i = (params.Y * (pi + bii * params.Y)).sum(1).argmin().item<int>();
    bool changed = (params.M.index({iy, ix, mingv_i}) == 0).item<bool>();
    params.M.index_put_({iy, ix}, 0);
    params.M.index_put_({iy, ix, mingv_i}, 1);
    return changed;
}

bool dcq::algorithm::update_M(int iy, int ix, float *M_data, float *Y_data, float *p_data, float *bii_data,
                          int ks, int h, int w, int c, int K) {
    float *pi = &p_data[iy * w * c + ix * c];
    float *mi = &M_data[iy * w * K + ix * K];
    float min = INFINITY;
    int amin = 0;
    for (int i = 0; i < K; i++) {
        float *yv = &Y_data[i * c];
        float sum = 0;
        for (int j = 0; j < c; j++) {
            float pic = pi[j];
            float biic = bii_data[j];
            float yvc = yv[j];
            sum += yvc * (pic + biic * yvc);
        }
        if (sum < min) {
            min = sum;
            amin = i;
        }
    }
    bool changed = mi[amin] == 0.0f;
    for (int i = 0; i < K; i++) {
        mi[i] = 0;
    }
    mi[amin] = 1;
    return changed;
}

void dcq::algorithm::update_p(
        int iy, int ix,
        dcq::Parameters &params,
        torch::Tensor &a,
        torch::Tensor &b0,
        torch::Tensor &p
) {
    int ks = b0.size(0);
    int ksh = ks / 2;
    int h = a.size(0);
    int w = a.size(1);
    int c = a.size(2);
    dcq::Region region = get_neighborhood(iy, ix, ks, a);

    auto windows = torch::zeros({(region.b - region.t) * (region.r - region.l), ks * ks, c});
    for (int i0 = region.t; i0 < region.b; i0++) {
        for (int i1 = region.l; i1 < region.r; i1++) {
            for (int ki = 0; ki < ks; ki++) {
                int iy = i0 + ksh - ki;
                for (int kj = 0; kj < ks; kj++) {
                    int ix = i1 + ksh - kj;
                    if (h > iy && iy >= 0 && w > ix && ix >= 0) {
                        auto v = params.M.index({iy, ix}).argmax().item<int>();
                        auto Yv = params.Y.index({v});
                        int w_index = (i0 - region.t) * (region.r - region.l) + (i1 - region.l);
                        int k_index = ki * ks + kj;
                        windows.index_put_({w_index, k_index}, Yv);
                    }
                }
            }
        }
    }
    auto s = a.index({Slice(region.t, region.b), Slice(region.l, region.r)}).clone();
    auto b02 = (b0 * 2).reshape({1, ks * ks, c});
    s += (b02 * windows).sum(1).reshape({region.b - region.t, region.r - region.l, c});
    p.index_put_({Slice(region.t, region.b), Slice(region.l, region.r)}, s);
}

int argmax(float *data, int len) {
    float max = -INFINITY;
    int amax = 0;
    for (int i = 0; i < len; i++) {
        if (data[i] > max) {
            amax = i;
            max = data[i];
        }
    }
    return amax;
}


void dcq::algorithm::update_p(int iy, int ix, float *M_data, float *Y_data, float *a_data, float *b0_data, float *p_data,
              int ks, int h, int w, int c, int K) {
    int ksh = ks / 2;
    int top = std::max(iy - ksh, 0);
    int bottom = std::min(iy + ksh, h - 1) + 1;
    int left = std::max(ix - ksh, 0);
    int right =  std::min(ix + ksh, w - 1) + 1;

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
                    int amax = argmax(&M_data[iy * w * K + ix * K], K);
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
    int k = params.M.size(2);
    auto new_M = torch::zeros({h * 2, w * 2, k});

    new_M.index_put_({Slice(None, None, 2), Slice(None, None, 2)}, params.M);
    new_M.index_put_({Slice(None, None, 2), Slice(1, None, 2)}, params.M);
    new_M.index_put_({Slice(1, None, 2), Slice(None, None, 2)}, params.M);
    new_M.index_put_({Slice(1, None, 2), Slice(1, None, 2)}, params.M);

    params.M = new_M;
}

dcq::Region dcq::algorithm::get_neighborhood(int iy, int ix, int ks, torch::Tensor &a) {
    int ksh = ks / 2;
    int h = a.size(0);
    int w = a.size(1);
    return {std::max(iy - ksh, 0), std::min(iy + ksh, h - 1) + 1, std::max(ix - ksh, 0), std::min(ix + ksh, w - 1) + 1};
}