#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

namespace rarefy_detail {

constexpr double kLnSqrt2Pi = 0.918938533204672741780329736406;
constexpr int64_t kMaxExactRInteger = 9007199254740992LL; // 2^53

inline double uniform_mt(std::mt19937_64& rng) {
  double u = 0.0;
  do {
    u = std::generate_canonical<double, 53>(rng);
  } while (u <= 0.0);
  return u;
}

// Exact fallback used when parameters exceed the integer range of the HRUA port.
inline int rhyper_urn_mt(std::mt19937_64& rng, int64_t population, int64_t good, int sample) {
  if (sample <= 0 || good <= 0 || population <= 0) {
    return 0;
  }
  if (good > population) {
    good = population;
  }
  if (sample > population) {
    sample = static_cast<int>(population);
  }
  if (good == population) {
    return sample;
  }

  int x = 0;
  int64_t remaining_population = population;
  int64_t remaining_good = good;
  for (int i = 0; i < sample; ++i) {
    if (remaining_good <= 0) {
      break;
    }
    if (remaining_population <= remaining_good) {
      x += sample - i;
      break;
    }
    const double u = uniform_mt(rng);
    if (u * static_cast<double>(remaining_population) < static_cast<double>(remaining_good)) {
      ++x;
      --remaining_good;
    }
    --remaining_population;
  }
  return x;
}

// afc(i) = log(i!) with the same approximation used in R's rhyper.c.
inline double afc(int i) {
  static constexpr double kTable[8] = {
      0.0,
      0.0,
      0.69314718055994530941723212145817,
      1.79175946922805500081247735838070,
      3.17805383034794561964694160129705,
      4.78749174278204599424770093452324,
      6.57925121201010099506017829290394,
      8.52516136106541430016553103634712,
  };
  if (i < 0) {
    throw std::runtime_error("rarefy: internal hypergeometric state became negative");
  }
  if (i <= 7) {
    return kTable[i];
  }
  const double di = static_cast<double>(i);
  const double i2 = di * di;
  return (di + 0.5) * std::log(di) - di + kLnSqrt2Pi +
         (0.0833333333333333 - 0.00277777777777778 / i2) / di;
}

// Returns -1 when the HRUA path encounters numerically unstable state and the
// caller should fall back to the exact urn implementation.
inline int rhyper_hrua_int_mt(std::mt19937_64& rng, int nn1, int nn2, int kk) {
  const double N = nn1 + static_cast<double>(nn2);

  const int n1 = std::min(nn1, nn2);
  const int n2 = std::max(nn1, nn2);
  const int k = (static_cast<double>(kk) + kk >= N) ? static_cast<int>(N - kk) : kk;

  const int m = static_cast<int>((k + 1.0) * (n1 + 1.0) / (N + 2.0));
  const int minjx = std::max(0, k - n2);
  const int maxjx = std::min(n1, k);

  int ix = 0;
  if (minjx == maxjx) {
    ix = maxjx;
  } else if (m - minjx < 10) {
    constexpr double scale = 1e25;
    constexpr double con = 57.5646273248511421;

    double lw = 0.0;
    if (k < n2) {
      lw = afc(n2) + afc(n1 + n2 - k) - afc(n2 - k) - afc(n1 + n2);
    } else {
      lw = afc(n1) + afc(k) - afc(k - n2) - afc(n1 + n2);
    }
    const double w = std::exp(lw + con);
    if (!(w > 0.0) || !std::isfinite(w)) {
      return -1;
    }

    while (true) {
      double p = w;
      double u = uniform_mt(rng) * scale;
      ix = minjx;
      bool restart = false;
      while (u > p) {
        u -= p;
        p *= static_cast<double>(n1 - ix) * static_cast<double>(k - ix);
        ++ix;
        p = p / static_cast<double>(ix) / static_cast<double>(n2 - k + ix);
        if (ix > maxjx || !(p > 0.0) || !std::isfinite(p)) {
          restart = true;
          break;
        }
      }
      if (!restart) {
        break;
      }
    }
  } else {
    constexpr double deltal = 0.0078;
    constexpr double deltau = 0.0034;

    const double s =
        std::sqrt((N - k) * k * n1 * static_cast<double>(n2) / (N - 1.0) / N / N);
    const double d = static_cast<int>(1.5 * s) + 0.5;
    const double xl = m - d + 0.5;
    const double xr = m + d + 0.5;
    const double a = afc(m) + afc(n1 - m) + afc(k - m) + afc(n2 - k + m);
    const double kl = std::exp(a - afc(static_cast<int>(xl)) - afc(static_cast<int>(n1 - xl)) -
                               afc(static_cast<int>(k - xl)) -
                               afc(static_cast<int>(n2 - k + xl)));
    const double kr =
        std::exp(a - afc(static_cast<int>(xr - 1.0)) - afc(static_cast<int>(n1 - xr + 1.0)) -
                 afc(static_cast<int>(k - xr + 1.0)) -
                 afc(static_cast<int>(n2 - k + xr - 1.0)));
    const double lamdl =
        -std::log(xl * (n2 - k + xl) / (n1 - xl + 1.0) / (k - xl + 1.0));
    const double lamdr =
        -std::log((n1 - xr + 1.0) * (k - xr + 1.0) / xr / (n2 - k + xr));
    const double p1 = d + d;
    const double p2 = p1 + kl / lamdl;
    const double p3 = p2 + kr / lamdr;

    if (!(lamdl > 0.0) || !(lamdr > 0.0) || !(p3 > 0.0) || !std::isfinite(p3)) {
      return -1;
    }

    bool reject = true;
    int n_uv = 0;
    do {
      const double u = uniform_mt(rng) * p3;
      double v = uniform_mt(rng);
      ++n_uv;
      if (n_uv >= 10000) {
        return -1;
      }

      if (u < p1) {
        ix = static_cast<int>(xl + u);
      } else if (u <= p2) {
        ix = static_cast<int>(xl + std::log(v) / lamdl);
        if (ix < minjx) {
          continue;
        }
        v *= (u - p1) * lamdl;
      } else {
        ix = static_cast<int>(xr - std::log(v) / lamdr);
        if (ix > maxjx) {
          continue;
        }
        v *= (u - p2) * lamdr;
      }

      if (m < 100 || ix <= 50) {
        double f = 1.0;
        if (m < ix) {
          for (int i = m + 1; i <= ix; ++i) {
            f = f * static_cast<double>(n1 - i + 1) * static_cast<double>(k - i + 1) /
                static_cast<double>(n2 - k + i) / static_cast<double>(i);
          }
        } else if (m > ix) {
          for (int i = ix + 1; i <= m; ++i) {
            f = f * static_cast<double>(i) * static_cast<double>(n2 - k + i) /
                static_cast<double>(n1 - i + 1) / static_cast<double>(k - i + 1);
          }
        }
        reject = !(v <= f);
      } else {
        const double y = ix;
        const double y1 = y + 1.0;
        const double ym = y - m;
        const double yn = n1 - y + 1.0;
        const double yk = k - y + 1.0;
        const double nk = n2 - k + y1;
        const double r = -ym / y1;
        const double s2 = ym / yn;
        const double t = ym / yk;
        const double e = -ym / nk;
        const double g = yn * yk / (y1 * nk) - 1.0;
        const double dg = (g < 0.0) ? 1.0 + g : 1.0;
        const double gu = g * (1.0 + g * (-0.5 + g / 3.0));
        const double gl = gu - 0.25 * (g * g * g * g) / dg;
        const double xm = m + 0.5;
        const double xn = n1 - m + 0.5;
        const double xk = k - m + 0.5;
        const double nm = n2 - k + xm;
        const double ub =
            y * gu - m * gl + deltau + xm * r * (1.0 + r * (-0.5 + r / 3.0)) +
            xn * s2 * (1.0 + s2 * (-0.5 + s2 / 3.0)) +
            xk * t * (1.0 + t * (-0.5 + t / 3.0)) +
            nm * e * (1.0 + e * (-0.5 + e / 3.0));
        const double alv = std::log(v);
        if (alv > ub) {
          reject = true;
        } else {
          double dr = xm * (r * r * r * r);
          if (r < 0.0) {
            dr /= (1.0 + r);
          }
          double ds = xn * (s2 * s2 * s2 * s2);
          if (s2 < 0.0) {
            ds /= (1.0 + s2);
          }
          double dt = xk * (t * t * t * t);
          if (t < 0.0) {
            dt /= (1.0 + t);
          }
          double de = nm * (e * e * e * e);
          if (e < 0.0) {
            de /= (1.0 + e);
          }
          if (alv < ub - 0.25 * (dr + ds + dt + de) + (y + m) * (gl - gu) - deltal) {
            reject = false;
          } else {
            reject = alv > (a - afc(ix) - afc(n1 - ix) - afc(k - ix) - afc(n2 - k + ix));
          }
        }
      }
    } while (reject);
  }

  if (static_cast<double>(kk) + kk >= N) {
    if (nn1 > nn2) {
      ix = kk - nn2 + ix;
    } else {
      ix = nn1 - ix;
    }
  } else if (nn1 > nn2) {
    ix = kk - ix;
  }
  return ix;
}

} // namespace rarefy_detail

// Exact scalar hypergeometric draw with mt19937_64-backed uniforms.
inline int rhyper_mt(std::mt19937_64& rng, int64_t N, int64_t K, int n) {
  if (n <= 0 || K <= 0 || N <= 0) {
    return 0;
  }
  if (K > N) {
    K = N;
  }
  if (n > N) {
    n = static_cast<int>(N);
  }
  if (K == N) {
    return n;
  }
  if (N > rarefy_detail::kMaxExactRInteger) {
    throw std::runtime_error(
        "rarefy: counts exceed R's exact integer range (2^53); smaller totals are required");
  }

  const int64_t good = K;
  const int64_t bad = N - K;
  if (good > INT_MAX || bad > INT_MAX) {
    return rarefy_detail::rhyper_urn_mt(rng, N, K, n);
  }

  const int draw = rarefy_detail::rhyper_hrua_int_mt(
      rng, static_cast<int>(good), static_cast<int>(bad), n);
  if (draw >= 0) {
    return draw;
  }
  return rarefy_detail::rhyper_urn_mt(rng, N, K, n);
}
