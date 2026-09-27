/* FP64 metric reduction only. No quantization or GEMM implementation here.
 * Eight independent comparisons; no additive error-contribution assumption.
 * Fixed lanes allow vectorization without -ffast-math or reassociation flags.
 */
#include <math.h>
#include <stddef.h>
int reduce_metrics(size_t n, const double *reference, const double *input,
                   const double *weight, const double *pair, const double *ideal,
                   const double *hardware, double *out) {
    double square[8][8] = {{0}}, absolute[8][8] = {{0}};
    double maximum[8][8] = {{0}}, energy[3][8] = {{0}};
    for (size_t base = 0; base < n; base += 8) {
        for (size_t lane = 0; lane < 8 && base + lane < n; ++lane) {
            size_t i = base + lane;
            double r = reference[i], a = input[i], w = weight[i];
            double p = pair[i], d = ideal[i], h = hardware[i];
            if (!(isfinite(r) && isfinite(a) && isfinite(w) && isfinite(p) && isfinite(d) && isfinite(h))) return 1;
            double e[8] = {a-r,w-r,p-r,d-r,h-d,h-r,d-p,h-p};
            energy[0][lane] += r*r;
            energy[1][lane] += d*d;
            energy[2][lane] += p*p;
            for (size_t k = 0; k < 8; ++k) {
                double ae = fabs(e[k]);
                square[k][lane] += e[k]*e[k];
                absolute[k][lane] += ae;
                if (ae > maximum[k][lane]) maximum[k][lane] = ae;
            }
        }
    }
    for (size_t k = 0; k < 8; ++k) {
        size_t energy_index = k == 4 ? 1 : k >= 6 ? 2 : 0;
        out[4*k] = out[4*k+1] = out[4*k+2] = out[4*k+3] = 0;
        for (size_t lane = 0; lane < 8; ++lane) {
            out[4*k] += square[k][lane];
            out[4*k+1] += absolute[k][lane];
            out[4*k+2] += energy[energy_index][lane];
            if (maximum[k][lane] > out[4*k+3]) out[4*k+3] = maximum[k][lane];
        }
    }
    return 0;
}
