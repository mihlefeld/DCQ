//
// Created by thesh on 12/07/2023.
//

#include "floyd_steinberg.h"
#include "utils.h"
#include <algorithm>

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

torch::Tensor dcq::fs::median_cuts(const torch::Tensor &X, int palette_size) {
    auto h = X.size(0);
    auto w = X.size(1);
    auto c = X.size(2);
    auto palette = median_cut_r(X.reshape({h * w, c}), palette_size);
    return palette;
}

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

void dither(int h, int w, int c, int k, float *X, int *M, float *Y) {
    float q_error[4];
    for (int iy = 0; iy < h; iy++) {
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


dcq::Parameters dcq::fs::solve(const torch::Tensor &X, int palette_size) {
    auto palette = median_cuts(X, palette_size);
    auto parameters = dcq::init::init_parameters(X);
    parameters.Y = palette;
    auto fs_x = X.clone();
    auto X_ptr = fs_x.data_ptr<float>();
    auto Y_ptr = parameters.Y.data_ptr<float>();
    auto M_ptr = parameters.M.data_ptr<int>();
    int h = X.size(0);
    int w = X.size(1);
    int c = X.size(2);
    dither(h, w, c, palette_size, X_ptr, M_ptr, Y_ptr);
    return parameters;
}