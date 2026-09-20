#ifndef ACTOR_EXCHANGE_H
#define ACTOR_EXCHANGE_H

#include "ladybug/ladybug_cluster_index.h"
#include "ladybug/ladybug_csr.h"
#include "ladybug/ladybug_db.h"
#include "ladybug/ladybug_node_store.h"
#include "utils.h"
#include <span>
#include <vector>

namespace distributed {

struct RemoteCandidateRequest {
  int32_t requester_rank;
  int32_t target_rank;
  int32_t sample_k;
  uint16_t max_year;
  std::vector<int32_t> generator_nodes;
};

struct RemoteCandidateResponse {
  int32_t requester_rank;
  int32_t target_rank;
  std::vector<int32_t> candidates;
};

struct RemoteCitation {
  int32_t source_node; // citing node u
  int32_t target_node; // cited node v
  uint16_t year;
};

/*
 * ActorExchange: Distributed actor messaging and batched columnar transport engine.
 * Implements:
 * 1. Low-alpha Remote Pre-Sampling (k = 10,000): bounds 2-hop cross-rank candidate payloads
 *    to ~40 KB per boundary agent by performing sampling directly on the remote owner node.
 * 2. Batched network exchanges via MPI Alltoallv / columnar buffers.
 * 3. Annual BSP Superstep Edge Reduction: routes cross-partition citations back to target owners.
 */
class ActorExchange {
public:
  ActorExchange();
  ~ActorExchange();

  // Low-alpha Remote Pre-Sampling:
  // Samples at most k unique candidate nodes from the 2-hop neighborhood of generator_nodes
  static std::vector<int32_t> RemotePreSample(
      const std::vector<int32_t> &generator_nodes, int k, uint16_t max_year,
      pcg32 &rng, const ladybug::LadyBugBidirectionalCSR &csr,
      const ladybug::LadyBugNodeStore &node_store);

  // Sample at most k candidates from a specific cluster
  static std::vector<int32_t> RemotePreSampleFromCluster(
      int cluster_id, int k, uint16_t max_year, pcg32 &rng,
      const ladybug::LadyBugClusterIndex &cluster_index,
      const ladybug::LadyBugNodeStore &node_store);

  // Queue outbound requests/citations during the active simulation year
  void QueueRequest(int target_rank, const std::vector<int32_t> &generator_nodes,
                    int sample_k, uint16_t max_year, int my_rank);
  void QueueCitation(int target_rank, int32_t source_node, int32_t target_node,
                     uint16_t year);

  // Bulk batched exchange of candidate requests and pre-sampled responses
  std::vector<RemoteCandidateResponse> ExchangeRequestsAndResponses(
      int my_rank, int world_size, pcg32 &rng,
      const ladybug::LadyBugBidirectionalCSR &csr,
      const ladybug::LadyBugNodeStore &node_store);

  // Annual superstep citation routing: sends citations to target node owners
  size_t ExchangeAnnualCitations(int my_rank, int world_size,
                                 ladybug::LadyBugDB &db);

  // Queue inspection
  size_t GetPendingRequestCount() const noexcept { return outgoing_requests_.size(); }
  size_t GetPendingCitationCount() const noexcept { return outgoing_citations_.size(); }

  void Clear();

private:
  std::vector<RemoteCandidateRequest> outgoing_requests_;
  std::vector<RemoteCitation> outgoing_citations_;
};

} // namespace distributed

#endif // ACTOR_EXCHANGE_H
