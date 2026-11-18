#ifndef CANTATAS_RNG_H
#define CANTATAS_RNG_H

#include "pcg_random.hpp"
#include <cstdint>

/*
 * Deterministic random streams.
 *
 * Every random decision in the model draws from rng::Thread(), a thread-local
 * pcg32. Before each unit of work (one agent, one creation batch, one ingest
 * stage) the generator is re-seeded from (run seed, stream, a, b). Results
 * therefore depend only on the run seed, never on how agents are spread over
 * threads or MPI ranks: a 4-rank run reproduces a single-process run exactly.
 */
namespace rng {

enum class Stream : uint64_t {
  IngestNumAuthors = 1,
  IngestAuthors = 2,
  AgentCreation = 3,
  Agent = 4,
  IngestAuthorOrder = 5,
};

// SplitMix64 finalizer: strong avalanche for deriving independent seeds.
inline uint64_t Mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

inline uint64_t Combine(uint64_t a, uint64_t b) { return Mix64(a ^ Mix64(b)); }

inline pcg32 &Thread() {
  thread_local pcg32 generator(0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL);
  return generator;
}

inline void SeedThread(uint64_t run_seed, Stream stream, uint64_t a,
                       uint64_t b = 0) {
  uint64_t key = Combine(Combine(Combine(run_seed, static_cast<uint64_t>(stream)), a), b);
  Thread().seed(key, Mix64(key ^ 0x5851F42D4C957F2DULL));
}

} // namespace rng

#endif // CANTATAS_RNG_H
