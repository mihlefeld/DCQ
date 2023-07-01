//
// Created by thesh on 01/07/2023.
//

#ifndef DCQ_PBAR_H
#define DCQ_PBAR_H

#include <cmath>
#include <chrono>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> hrclock;

enum pbar_timers {
    ASSIGN_TIMER = 0,
    COMPUTE_TIMER = 1,
    ADD_TIMER = 2
};

class PBar {
    hrclock t0;
    hrclock assign_t0;
    hrclock compute_t0;
    hrclock add_t0;
    long assign_t{};
    long compute_t{};
    long add_t{};

    static float to_s(long ms);

public:
    float loss = INFINITY;
    int K = 0;
    int l = 0;
    int max_L = 0;
    int max_K = 0;

    ~PBar();

    void start(pbar_timers timer);

    void stop(pbar_timers timer);

    void update();

public:
    PBar(int K, int l);
};


#endif //DCQ_PBAR_H
