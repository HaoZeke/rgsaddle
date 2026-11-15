/* A meson parent project that sees rgsaddle through its headers only
 * and loads the runtime at run time, the way gpridge does. The C
 * declarations type the dlsym lookups; RGSADDLE_LIBRARY names the
 * cdylib. A climbing band on a double well has to find the saddle. */
#include <rgsaddle.h>
#include <rgsaddle/session.hpp>

#include <dlfcn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

template <typename Fn>
bool resolve(void* lib, const char* name, Fn*& out) {
    out = reinterpret_cast<Fn*>(dlsym(lib, name));
    if (out == nullptr) {
        std::fprintf(stderr, "missing symbol %s\n", name);
    }
    return out != nullptr;
}

double well(const double* p, double* g) {
    const double x = p[0], y = p[1], z = p[2];
    g[0] = 4.0 * x * (x * x - 1.0);
    g[1] = 4.0 * y;
    g[2] = 4.0 * z;
    return (x * x - 1.0) * (x * x - 1.0) + 2.0 * y * y + 2.0 * z * z;
}

rgsaddle_status_t surface(void*, rgsaddle_surface_request_t* req) {
    if (req->version.major != RGSADDLE_ABI_MAJOR) {
        return RGSADDLE_ABI_MISMATCH;
    }
    const int64_t dof = 3 * req->n_atoms;
    const int64_t rows = (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
    for (int64_t i = 0; i < rows; ++i) {
        req->energies[i] = well(req->positions + i * dof, req->gradients + i * dof);
    }
    return RGSADDLE_OK;
}

}  // namespace

int main() {
    const char* path = std::getenv("RGSADDLE_LIBRARY");
    if (path == nullptr) {
        std::fprintf(stderr, "RGSADDLE_LIBRARY is not set\n");
        return 1;
    }
    void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }

    decltype(rgsaddle_abi_stamp)* abi_stamp = nullptr;
    decltype(rgsaddle_status_name)* status_name = nullptr;
    decltype(rgsaddle_band_create)* band_create = nullptr;
    decltype(rgsaddle_band_step)* band_step = nullptr;
    decltype(rgsaddle_band_positions)* band_positions = nullptr;
    decltype(rgsaddle_band_free)* band_free = nullptr;
    if (!resolve(lib, "rgsaddle_abi_stamp", abi_stamp) ||
        !resolve(lib, "rgsaddle_status_name", status_name) ||
        !resolve(lib, "rgsaddle_band_create", band_create) ||
        !resolve(lib, "rgsaddle_band_step", band_step) ||
        !resolve(lib, "rgsaddle_band_positions", band_positions) ||
        !resolve(lib, "rgsaddle_band_free", band_free)) {
        return 1;
    }

    rgsaddle_version_t runtime{};
    if (abi_stamp(&runtime) != RGSADDLE_OK || runtime.major != RGSADDLE_ABI_MAJOR ||
        runtime.minor < RGSADDLE_ABI_MINOR) {
        std::fprintf(stderr, "runtime ABI %u.%u, header %u.%u\n", runtime.major,
                     runtime.minor, RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR);
        return 1;
    }

    const int64_t n_images = 9, n_atoms = 1;
    std::vector<double> pos(static_cast<std::size_t>(n_images * 3));
    for (int64_t i = 0; i < n_images; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n_images - 1);
        pos[static_cast<std::size_t>(3 * i + 0)] = -1.0 + 2.0 * t;
        pos[static_cast<std::size_t>(3 * i + 1)] = 0.3 * std::sin(3.14159265358979 * t);
    }

    rgsaddle_band_config_t cfg = rgsaddle::band_config();
    cfg.method = RGSADDLE_METHOD_FIRE;
    cfg.ci_trigger_factor = 0.5;
    RgsaddleBand* band = band_create(&cfg, n_images, n_atoms, pos.data());
    if (band == nullptr) {
        std::fprintf(stderr, "rgsaddle_band_create returned NULL\n");
        return 1;
    }

    rgsaddle_report_t report{};
    for (int i = 0; i < 400; ++i) {
        const rgsaddle_status_t status = band_step(band, surface, nullptr, &report);
        if (status != RGSADDLE_OK) {
            std::fprintf(stderr, "step %d: %s\n", i, status_name(status));
            band_free(band);
            return 1;
        }
        if (report.status == RGSADDLE_STATUS_CONVERGED) {
            break;
        }
    }
    if (report.status != RGSADDLE_STATUS_CONVERGED || report.ci_index < 0) {
        std::fprintf(stderr, "band status %d, climbing image %lld\n", report.status,
                     static_cast<long long>(report.ci_index));
        band_free(band);
        return 1;
    }
    if (band_positions(band, pos.data()) != RGSADDLE_OK) {
        std::fprintf(stderr, "rgsaddle_band_positions failed\n");
        band_free(band);
        return 1;
    }
    band_free(band);

    const double* saddle = pos.data() + 3 * report.ci_index;
    double g[3];
    const double energy = well(saddle, g);
    if (std::fabs(saddle[0]) > 1e-2 || std::fabs(energy - 1.0) > 1e-3) {
        std::fprintf(stderr, "climbing image at (%g, %g, %g), energy %g\n", saddle[0],
                     saddle[1], saddle[2], energy);
        return 1;
    }
    std::printf("rgsaddle %u.%u: saddle energy %.6f after %lld iterations\n", runtime.major,
                runtime.minor, energy, static_cast<long long>(report.iteration));
    return 0;
}
