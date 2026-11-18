#ifndef LADYBUG_PARALLEL_H
#define LADYBUG_PARALLEL_H

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ladybug {

// In-place exclusive prefix sum: data[i] becomes the sum of data[0..i).
// Returns the total. Parallel two-pass block scan; the blocks are sized from
// the team OpenMP actually provides, so it is correct under OMP_THREAD_LIMIT,
// OMP_DYNAMIC or nesting.
template <typename T> T ExclusiveScanInPlace(T *data, size_t n) {
  if (n == 0) {
    return T{};
  }
  if (n < (size_t{1} << 16) || omp_get_max_threads() == 1) {
    T running{};
    for (size_t i = 0; i < n; ++i) {
      T value = data[i];
      data[i] = running;
      running += value;
    }
    return running;
  }
  std::vector<T> block_sums;
  T total{};
#pragma omp parallel
  {
    const size_t threads = static_cast<size_t>(omp_get_num_threads());
    const size_t t = static_cast<size_t>(omp_get_thread_num());
#pragma omp single
    block_sums.assign(threads + 1, T{});
    // (implicit barrier after single)
    const size_t block = (n + threads - 1) / threads;
    const size_t lo = std::min(n, t * block);
    const size_t hi = std::min(n, lo + block);
    T sum{};
    for (size_t i = lo; i < hi; ++i) {
      sum += data[i];
    }
    block_sums[t + 1] = sum;
#pragma omp barrier
#pragma omp single
    {
      for (size_t b = 0; b < threads; ++b) {
        block_sums[b + 1] += block_sums[b];
      }
      total = block_sums[threads];
    }
    T running = block_sums[t];
    for (size_t i = lo; i < hi; ++i) {
      T value = data[i];
      data[i] = running;
      running += value;
    }
  }
  return total;
}

} // namespace ladybug

#endif // LADYBUG_PARALLEL_H
