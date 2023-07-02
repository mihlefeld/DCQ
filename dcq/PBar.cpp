//
// Created by thesh on 01/07/2023.
//

#include "PBar.h"
#include <iomanip>
#include <iostream>

void PBar::update() {
    auto now = std::chrono::high_resolution_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
    float t0_s = to_s(ms);
    float add_s = to_s(add_t);
    float compute_s = to_s(compute_t);
    float assign_s = to_s(assign_t);

    std::cout << std::setprecision(3) << "T=" << t0_s << "s " << "K=" << K << "/" << max_K << " L=" << l << "/"
              << max_L << " loss=" << loss << " cmp=" << compute_s << "s asg=" << assign_s << "s add="
              << add_s << "s                       \r" << std::flush;

}

void PBar::start(pbar_timers timer) {
    hrclock *t0;
    switch (timer) {
        case ASSIGN_TIMER:
            t0 = &this->assign_t0;
            break;
        case COMPUTE_TIMER:
            t0 = &this->compute_t0;
            break;
        case ADD_TIMER:
            t0 = &this->add_t0;
            break;
    }
    *t0 = std::chrono::high_resolution_clock::now();
}

void PBar::stop(pbar_timers timer) {
    hrclock t0;
    long *ms;
    switch (timer) {
        case ASSIGN_TIMER:
            t0 = assign_t0;
            ms = &assign_t;
            break;
        case COMPUTE_TIMER:
            t0 = this->compute_t0;
            ms = &compute_t;
            break;
        case ADD_TIMER:
            t0 = this->add_t0;
            ms = &add_t;
            break;
    }
    auto now = std::chrono::high_resolution_clock::now();
    *ms += std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
    update();
}

PBar::PBar(int K, int l) {
    this->K = K;
    this->l = l;
    t0 = std::chrono::high_resolution_clock::now();
}

float PBar::to_s(long ms) {
    float seconds = (float) ms / 1000;
    return seconds;
}

PBar::~PBar() {
    update();
    std::cout << std::endl;
}
