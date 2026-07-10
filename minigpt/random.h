// random.h
#pragma once
#include <cstdint>
#include <cmath>
#include <limits>
#include <random>

namespace rng {

class Xoshiro256 {
  uint64_t s[4];
public:
  explicit Xoshiro256(uint64_t seed = 0) {
    // Inisialisasi dengan splitmix64 untuk seed nol pun aman
    auto splitmix = [](uint64_t& x) {
      uint64_t z = (x += 0x9e3779b97f4a7c15);
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
      z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
      return z ^ (z >> 31);
    };
    if (seed == 0) {
      seed = std::random_device{}();
    }
    uint64_t x = seed;
    s[0] = splitmix(x);
    s[1] = splitmix(x);
    s[2] = splitmix(x);
    s[3] = splitmix(x);
  }

  inline uint64_t next() noexcept {
    const uint64_t result = s[0] + s[3];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 45) | (s[3] >> 19);
    return result;
  }

  // Uniform [0,1) double
  inline double uniform01() noexcept {
    return (next() >> 11) * 0x1.0p-53;
  }

  // Normal distribution (Box–Muller, menghasilkan dua sample tapi kita kembalikan satu per call)
  double normal() noexcept {
    // Menggunakan cache untuk sampel kedua
    static thread_local bool has_spare = false;
    static thread_local double spare;
    if (has_spare) {
      has_spare = false;
      return spare;
    }
    double u1 = uniform01();
    double u2 = uniform01();
    double r = std::sqrt(-2.0 * std::log(u1));
    double theta = 2.0 * M_PI * u2;
    has_spare = true;
    spare = r * std::sin(theta);
    return r * std::cos(theta);
  }
};

} // namespace rng