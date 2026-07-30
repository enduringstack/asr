// Copyright (c) 2016 Network
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "feature_extractor/wenet_fft.h"

#include <cmath>

namespace asr_frontend {

void make_sintbl(int n, float* sintbl) {
    int n2 = n / 2;
    int n4 = n / 4;
    int n8 = n / 8;
    float t = std::sin(M_PI / n);
    float dc = 2 * t * t;
    float ds = std::sqrt(dc * (2 - dc));
    t = 2 * dc;
    float c = sintbl[n4] = 1;
    float s = sintbl[0] = 0;

    for (int i = 1; i < n8; ++i) {
        c -= dc;
        dc += t * c;
        s += ds;
        ds -= t * s;
        sintbl[i] = s;
        sintbl[n4 - i] = c;
    }
    if (n8 != 0) {
        sintbl[n8] = std::sqrt(0.5f);
    }
    for (int i = 0; i < n4; ++i) {
        sintbl[n2 - i] = sintbl[i];
    }
    for (int i = 0; i < n2 + n4; ++i) {
        sintbl[i + n2] = -sintbl[i];
    }
}

void make_bitrev(int n, int* bitrev) {
    int n2 = n / 2;
    int i = 0;
    int j = 0;
    for (;;) {
        bitrev[i] = j;
        if (++i >= n) {
            break;
        }
        int k = n2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
}

int fft(const int* bitrev, const float* sintbl, float* x, float* y, int n) {
    int inverse = 0;
    if (n < 0) {
        n = -n;
        inverse = 1;
    }
    if (n == 0) {
        return 0;
    }

    int n4 = n / 4;
    for (int i = 0; i < n; ++i) {
        int j = bitrev[i];
        if (i < j) {
            float t = x[i];
            x[i] = x[j];
            x[j] = t;
            t = y[i];
            y[i] = y[j];
            y[j] = t;
        }
    }

    for (int k = 1; k < n;) {
        int h = 0;
        int k2 = k + k;
        int d = n / k2;
        for (int j = 0; j < k; ++j) {
            float c = sintbl[h + n4];
            float s = inverse ? -sintbl[h] : sintbl[h];
            for (int i = j; i < n; i += k2) {
                int ik = i + k;
                float dx = s * y[ik] + c * x[ik];
                float dy = c * y[ik] - s * x[ik];
                x[ik] = x[i] - dx;
                x[i] += dx;
                y[ik] = y[i] - dy;
                y[i] += dy;
            }
            h += d;
        }
        k = k2;
    }

    if (inverse) {
        for (int i = 0; i < n; ++i) {
            x[i] /= n;
            y[i] /= n;
        }
    }
    return 0;
}

} // namespace asr_frontend
