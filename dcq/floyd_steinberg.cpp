//
// Created by thesh on 12/07/2023.
//

#include "floyd_steinberg.h"
#include "utils.h"
#include "algorithm.h"
#include <algorithm>

namespace F = torch::nn::functional;


int find_closest_color(int k, int c, float *pixel, float *palette) {
    float error = INFINITY;
    int min_index = 0;
    for (int ik = 0; ik < k; ik++) {
        float sum = 0;
        auto p_pixel = &palette[ik * c];
        for (int ic = 0; ic < c; ic++) {
            float pixel_ic = std::max(std::min(pixel[ic], 1.0f), 0.0f);
            float diff = p_pixel[ic] - pixel[ic];
            sum += diff * diff;
        }
        if (sum < error) {
            min_index = ik;
            error = sum;
        }
    }
    return min_index;
}

torch::Tensor median_cut_r(const torch::Tensor &bucket, int buckets) {
    if (buckets == 1) {
        return bucket.mean(0, true);
    }

    if (buckets < 1) {
        return torch::zeros({0, bucket.size(1)});
    }

    auto bucket_ranges = std::get<0>(bucket.max(0)) - std::get<0>(bucket.min(0));
    auto max_range = bucket_ranges.argmax().item<int>();
    auto sort_index = torch::argsort(bucket.index({Slice(), max_range}));
    auto sorted = bucket.index({sort_index});
    auto size = bucket.size(0);
    auto buckets_lower = buckets / 2;
    auto buckets_upper = buckets - buckets_lower;
    auto lower = sorted.index({Slice(0, size / 2)});
    auto upper = sorted.index({Slice(size / 2)});

    auto palette_lower = median_cut_r(lower, buckets_lower);
    auto palette_upper = median_cut_r(upper, buckets_upper);

    return torch::concat({palette_lower, palette_upper}, 0);
}

torch::Tensor dcq::fs::median_cuts(const torch::Tensor &X, int palette_size, dcq::PBar &pbar) {
    auto h = X.size(0);
    auto w = X.size(1);
    auto c = X.size(2);
    auto palette = median_cut_r(X.reshape({h * w, c}), palette_size);
    return palette;
}

void adjust_centroids(const torch::Tensor &X, torch::Tensor &palette, torch::Tensor &assignments) {
    for (int i = 0; i < palette.size(0); i++) {
        auto color = X.index({assignments == i}).mean({0});
        palette.index_put_({i}, color);
    }
}

int reassign_clusters(const torch::Tensor &X, torch::Tensor &palette, torch::Tensor &assignments) {
    int c = X.size(-1);
    int h = X.size(0);
    int w = X.size(1);
    int palette_size = palette.size(0);

    float *pixels = X.data_ptr<float>();
    float *colors = palette.data_ptr<float>();
    int *assign = assignments.data_ptr<int>();
    int changes = 0;

#pragma omp parallel for num_threads(omp_get_max_threads()) default(none) firstprivate(c, h, w, palette_size) reduction(+: changes) shared(pixels, colors, assign)
    for (int iy = 0; iy < h; iy++) {
        float *pixel_row = &pixels[iy * w * c];
        int *assig_row = &assign[iy * w];
        for (int ix = 0; ix < w; ix++) {
            int color_idx = find_closest_color(palette_size, c, &pixel_row[ix * c], colors);
            if (assig_row[ix] != color_idx) {
                changes += 1;
                assig_row[ix] = color_idx;
            }
        }
    }
    return changes;
}

torch::Tensor dcq::fs::k_means(const torch::Tensor &X, int palette_size, dcq::PBar &pbar) {
    int c = X.size(-1);
    int h = X.size(0);
    int w = X.size(1);

    pbar.update();
    auto palette = torch::rand({palette_size, c});
    pbar.update();

    auto assignments = torch::zeros({h, w}, torch::kInt32) - 1;
    int max_iterations = 10;
    pbar.max_K = max_iterations;
    for (int i = 0; i < max_iterations; i++) {
        int changed = reassign_clusters(X, palette, assignments);
        adjust_centroids(X, palette, assignments);
        pbar.K = i + 1;
        pbar.l = changed;
        pbar.update();
        if (changed == 0) {
            break;
        }
    }
    return palette;
}

torch::Tensor dcq::fs::cluster(const torch::Tensor &X, int palette_size, const std::string &mode, dcq::PBar &pbar) {
    if (mode == "k_means") {
        return k_means(X, palette_size, pbar);
    }
    if (mode == "median_cuts") {
        return dcq::fs::median_cuts(X, palette_size);
    }
}

void set_error(float *error, const float *old_pixel, const float *new_pixel, int c) {
    for (int i = 0; i < c; i++) {
        error[i] = old_pixel[i] - new_pixel[i];
    }
}

void add_error(float *pixel, float *error, float factor, int c) {
    for (int i = 0; i < c; i++) {
        pixel[i] += error[i] * factor;
    }
}

void dither(int h, int w, int c, int k, float *X, int *M, float *Y, dcq::PBar &pbar) {
    float q_error[4];
    for (int iy = 0; iy < h; iy++) {
        pbar.update();
        for (int ix = 0; ix < w; ix++) {
            auto old_pixel = &X[iy * w * c + ix * c];
            int color_id = find_closest_color(k, c, old_pixel, Y);
            auto new_pixel = &Y[color_id * c];
            M[iy * w + ix] = color_id;
            set_error(q_error, old_pixel, new_pixel, c);
/*
 *      pixels[x + 1][y    ] := pixels[x + 1][y    ] + quant_error × 7 / 16
        pixels[x - 1][y + 1] := pixels[x - 1][y + 1] + quant_error × 3 / 16
        pixels[x    ][y + 1] := pixels[x    ][y + 1] + quant_error × 5 / 16
        pixels[x + 1][y + 1] := pixels[x + 1][y + 1] + quant_error × 1 / 16
 */
            if (ix < w - 1) {
                add_error(&X[iy * w * c + (ix + 1) * c], q_error, 7.0f / 16.0f, c);
                if (iy < h - 1) {
                    add_error(&X[(iy + 1) * w * c + (ix + 1) * c], q_error, 1.0f / 16.0f, c);
                }
            }
            if (iy < h - 1) {
                add_error(&X[(iy + 1) * w * c + ix * c], q_error, 5.0f / 16.0f, c);
                if (ix > 0) {
                    add_error(&X[(iy + 1) * w * c + (ix - 1) * c], q_error, 1.0f / 16.0f, c);
                }
            }

        }
    }
}



dcq::Parameters dcq::fs::solve(const torch::Tensor &X, int palette_size, const std::string &mode) {
    auto pbar = PBar(0, 0);
    auto palette = dcq::fs::cluster(X, palette_size, mode, pbar);
    auto parameters = dcq::init::init_parameters(X);
    parameters.Y = palette;
    auto fs_x = X.clone();
    auto X_ptr = fs_x.data_ptr<float>();
    auto Y_ptr = parameters.Y.data_ptr<float>();
    auto M_ptr = parameters.M.data_ptr<int>();
    int h = X.size(0);
    int w = X.size(1);
    int c = X.size(2);
    dither(h, w, c, palette_size, X_ptr, M_ptr, Y_ptr, pbar);
    pbar.update();
    return parameters;
}

dcq::Parameters dcq::fs::solve_icm(torch::Tensor &X, int palette_size, const std::string &mode) {
    int h = X.size(0);
    int w = X.size(1);
    int c = X.size(2);
    auto pbar = PBar(0, 0);
    auto palette = dcq::fs::cluster(X, palette_size, mode, pbar);
    auto parameters = dcq::init::init_parameters(X);
    auto kernels = dcq::init::init_kernels(3, c);
    auto a = dcq::utils::from_batched(
            -2 * F::conv2d(dcq::utils::to_batched(X), dcq::utils::to_conv_kernel(kernels.b),
                           F::Conv2dFuncOptions().padding({2, 2}).groups(c))
    );

    parameters.Y = palette;
    auto fs_x = X.clone();
    auto X_ptr = fs_x.data_ptr<float>();
    float loss = INFINITY;
    float old_loss = INFINITY;
    int iterations = 0;
    do {
        iterations++;
        old_loss = loss;
        dither(h, w, c, palette_size, X_ptr, parameters.M.data_ptr<int>(), parameters.Y.data_ptr<float>(), pbar);
        dcq::algorithm::compute_colors(parameters, kernels.b, a);
        loss = dcq::algorithm::compute_loss(X, parameters, kernels.W);
        pbar.loss = loss;
        pbar.l = iterations;
        pbar.update();
    } while (!dcq::utils::is_close(loss, old_loss));
    return parameters;
}