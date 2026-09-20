#include "distributed/actor_exchange.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <unordered_set>

#if __has_include(<mpi.h>)
#define DISTRIBUTED_HAS_MPI 1
#include <mpi.h>
#else
#define DISTRIBUTED_HAS_MPI 0
#endif

namespace distributed {

ActorExchange::ActorExchange() {}

ActorExchange::~ActorExchange() {
  Clear();
}

void ActorExchange::Clear() {
  outgoing_requests_.clear();
  outgoing_citations_.clear();
}

std::vector<int32_t> ActorExchange::RemotePreSample(
    const std::vector<int32_t> &generator_nodes, int k, uint16_t max_year,
    pcg32 &rng, const ladybug::LadyBugBidirectionalCSR &csr,
    const ladybug::LadyBugNodeStore &node_store) {
  if (generator_nodes.empty() || k <= 0) {
    return {};
  }

  std::unordered_set<int32_t> candidate_set;
  size_t node_count = node_store.GetNodeCount();

  for (int32_t g : generator_nodes) {
    if (g < 0 || static_cast<size_t>(g) >= node_count) continue;

    auto hop1 = csr.GetOutNeighbors(g);
    for (int32_t h1 : hop1) {
      if (h1 < 0 || static_cast<size_t>(h1) >= node_count) continue;

      if (node_store.GetYear(h1) <= max_year) {
        candidate_set.insert(h1);
      }

      auto hop2 = csr.GetOutNeighbors(h1);
      for (int32_t h2 : hop2) {
        if (h2 < 0 || static_cast<size_t>(h2) >= node_count) continue;

        if (node_store.GetYear(h2) <= max_year) {
          candidate_set.insert(h2);
        }
      }
    }
  }

  std::vector<int32_t> all_candidates;
  all_candidates.reserve(candidate_set.size());
  for (int32_t c : candidate_set) {
    all_candidates.push_back(c);
  }

  if (all_candidates.size() <= static_cast<size_t>(k)) {
    return all_candidates;
  }

  // Reservoir / Partial Fisher-Yates shuffle to pick exactly k uniformly at random
  for (size_t i = 0; i < static_cast<size_t>(k); ++i) {
    size_t j = i + (rng() % (all_candidates.size() - i));
    std::swap(all_candidates[i], all_candidates[j]);
  }
  all_candidates.resize(k);
  return all_candidates;
}

std::vector<int32_t> ActorExchange::RemotePreSampleFromCluster(
    int cluster_id, int k, uint16_t max_year, pcg32 &rng,
    const ladybug::LadyBugClusterIndex &cluster_index,
    const ladybug::LadyBugNodeStore &node_store) {
  const auto &cluster_nodes = cluster_index.GetClusterNodes(cluster_id);
  if (cluster_nodes.empty() || k <= 0) {
    return {};
  }

  size_t node_count = node_store.GetNodeCount();
  std::vector<int32_t> filtered;
  filtered.reserve(std::min<size_t>(cluster_nodes.size(), k * 2));

  for (int32_t u : cluster_nodes) {
    if (u >= 0 && static_cast<size_t>(u) < node_count) {
      if (node_store.GetYear(u) <= max_year) {
        filtered.push_back(u);
      }
    }
  }

  if (filtered.size() <= static_cast<size_t>(k)) {
    return filtered;
  }

  for (size_t i = 0; i < static_cast<size_t>(k); ++i) {
    size_t j = i + (rng() % (filtered.size() - i));
    std::swap(filtered[i], filtered[j]);
  }
  filtered.resize(k);
  return filtered;
}

void ActorExchange::QueueRequest(
    int target_rank, const std::vector<int32_t> &generator_nodes, int sample_k,
    uint16_t max_year, int my_rank) {
  outgoing_requests_.push_back(RemoteCandidateRequest{
      .requester_rank = my_rank,
      .target_rank = target_rank,
      .sample_k = sample_k,
      .max_year = max_year,
      .generator_nodes = generator_nodes,
  });
}

void ActorExchange::QueueCitation(int target_rank, int32_t source_node,
                                  int32_t target_node, uint16_t year) {
  outgoing_citations_.push_back(RemoteCitation{
      .source_node = source_node,
      .target_node = target_node,
      .year = year,
  });
}

std::vector<RemoteCandidateResponse>
ActorExchange::ExchangeRequestsAndResponses(
    int my_rank, int world_size, pcg32 &rng,
    const ladybug::LadyBugBidirectionalCSR &csr,
    const ladybug::LadyBugNodeStore &node_store) {
  std::vector<RemoteCandidateResponse> responses;

#if DISTRIBUTED_HAS_MPI
  int flag = 0;
  MPI_Initialized(&flag);
  bool mpi_active = (flag && world_size > 1);
#else
  bool mpi_active = false;
#endif

  if (!mpi_active) {
    for (const auto &req : outgoing_requests_) {
      auto cands = RemotePreSample(req.generator_nodes, req.sample_k, req.max_year,
                                   rng, csr, node_store);
      responses.push_back(RemoteCandidateResponse{
          .requester_rank = req.requester_rank,
          .target_rank = req.target_rank,
          .candidates = std::move(cands),
      });
    }
    outgoing_requests_.clear();
    return responses;
  }

#if DISTRIBUTED_HAS_MPI
  // Multi-Rank MPI Exchange
  // 1. Pack outgoing requests into per-target flat buffers:
  // [requester_rank, target_rank, sample_k, max_year, gen_count, g0, g1, ...]
  std::vector<std::vector<int32_t>> send_req_bufs(world_size);
  for (const auto &req : outgoing_requests_) {
    int tr = req.target_rank;
    if (tr < 0 || tr >= world_size) tr = 0;
    send_req_bufs[tr].push_back(req.requester_rank);
    send_req_bufs[tr].push_back(req.target_rank);
    send_req_bufs[tr].push_back(req.sample_k);
    send_req_bufs[tr].push_back(static_cast<int32_t>(req.max_year));
    send_req_bufs[tr].push_back(static_cast<int32_t>(req.generator_nodes.size()));
    for (int32_t g : req.generator_nodes) {
      send_req_bufs[tr].push_back(g);
    }
  }

  std::vector<int> send_req_counts(world_size, 0);
  for (int r = 0; r < world_size; ++r) {
    send_req_counts[r] = static_cast<int>(send_req_bufs[r].size());
  }

  std::vector<int> recv_req_counts(world_size, 0);
  MPI_Alltoall(send_req_counts.data(), 1, MPI_INT, recv_req_counts.data(), 1,
               MPI_INT, MPI_COMM_WORLD);

  std::vector<int> sreq_displs(world_size, 0);
  std::vector<int> rreq_displs(world_size, 0);
  int total_sreq = 0;
  int total_rreq = 0;
  for (int r = 0; r < world_size; ++r) {
    sreq_displs[r] = total_sreq;
    total_sreq += send_req_counts[r];
    rreq_displs[r] = total_rreq;
    total_rreq += recv_req_counts[r];
  }

  std::vector<int32_t> flat_sreq(total_sreq);
  int offset = 0;
  for (int r = 0; r < world_size; ++r) {
    if (send_req_counts[r] > 0) {
      std::memcpy(flat_sreq.data() + offset, send_req_bufs[r].data(),
                  send_req_counts[r] * sizeof(int32_t));
      offset += send_req_counts[r];
    }
  }

  std::vector<int32_t> flat_rreq(total_rreq);
  MPI_Alltoallv(flat_sreq.data(), send_req_counts.data(), sreq_displs.data(),
                MPI_INT32_T, flat_rreq.data(), recv_req_counts.data(),
                rreq_displs.data(), MPI_INT32_T, MPI_COMM_WORLD);

  // Process incoming requests locally & pack responses into per-requester buffers
  std::vector<std::vector<int32_t>> send_resp_bufs(world_size);
  size_t req_idx = 0;
  while (req_idx < flat_rreq.size()) {
    int32_t req_rank = flat_rreq[req_idx++];
    int32_t tgt_rank = flat_rreq[req_idx++];
    int32_t sample_k = flat_rreq[req_idx++];
    uint16_t max_year = static_cast<uint16_t>(flat_rreq[req_idx++]);
    int32_t gen_count = flat_rreq[req_idx++];
    std::vector<int32_t> gens;
    gens.reserve(gen_count);
    for (int32_t i = 0; i < gen_count; ++i) {
      gens.push_back(flat_rreq[req_idx++]);
    }

    auto cands = RemotePreSample(gens, sample_k, max_year, rng, csr, node_store);

    int rr = req_rank;
    if (rr < 0 || rr >= world_size) rr = 0;
    send_resp_bufs[rr].push_back(req_rank);
    send_resp_bufs[rr].push_back(tgt_rank);
    send_resp_bufs[rr].push_back(static_cast<int32_t>(cands.size()));
    for (int32_t c : cands) {
      send_resp_bufs[rr].push_back(c);
    }
  }

  std::vector<int> send_resp_counts(world_size, 0);
  for (int r = 0; r < world_size; ++r) {
    send_resp_counts[r] = static_cast<int>(send_resp_bufs[r].size());
  }

  std::vector<int> recv_resp_counts(world_size, 0);
  MPI_Alltoall(send_resp_counts.data(), 1, MPI_INT, recv_resp_counts.data(), 1,
               MPI_INT, MPI_COMM_WORLD);

  std::vector<int> sresp_displs(world_size, 0);
  std::vector<int> rresp_displs(world_size, 0);
  int total_sresp = 0;
  int total_rresp = 0;
  for (int r = 0; r < world_size; ++r) {
    sresp_displs[r] = total_sresp;
    total_sresp += send_resp_counts[r];
    rresp_displs[r] = total_rresp;
    total_rresp += recv_resp_counts[r];
  }

  std::vector<int32_t> flat_sresp(total_sresp);
  offset = 0;
  for (int r = 0; r < world_size; ++r) {
    if (send_resp_counts[r] > 0) {
      std::memcpy(flat_sresp.data() + offset, send_resp_bufs[r].data(),
                  send_resp_counts[r] * sizeof(int32_t));
      offset += send_resp_counts[r];
    }
  }

  std::vector<int32_t> flat_rresp(total_rresp);
  MPI_Alltoallv(flat_sresp.data(), send_resp_counts.data(), sresp_displs.data(),
                MPI_INT32_T, flat_rresp.data(), recv_resp_counts.data(),
                rresp_displs.data(), MPI_INT32_T, MPI_COMM_WORLD);

  // Unpack responses
  size_t resp_idx = 0;
  while (resp_idx < flat_rresp.size()) {
    int32_t req_rank = flat_rresp[resp_idx++];
    int32_t tgt_rank = flat_rresp[resp_idx++];
    int32_t cand_count = flat_rresp[resp_idx++];
    std::vector<int32_t> cands;
    cands.reserve(cand_count);
    for (int32_t i = 0; i < cand_count; ++i) {
      cands.push_back(flat_rresp[resp_idx++]);
    }
    responses.push_back(RemoteCandidateResponse{
        .requester_rank = req_rank,
        .target_rank = tgt_rank,
        .candidates = std::move(cands),
    });
  }

  outgoing_requests_.clear();
  return responses;
#endif
}

size_t ActorExchange::ExchangeAnnualCitations(int my_rank, int world_size,
                                               ladybug::LadyBugDB &db) {
  if (outgoing_citations_.empty() && world_size <= 1) {
    return 0;
  }

#if DISTRIBUTED_HAS_MPI
  int flag = 0;
  MPI_Initialized(&flag);
  bool mpi_active = (flag && world_size > 1);
#else
  bool mpi_active = false;
#endif

  if (!mpi_active) {
    size_t count = outgoing_citations_.size();
    for (const auto &cit : outgoing_citations_) {
      db.CSR().RecordDeltaIncomingCitation(0, cit.target_node, cit.source_node);
    }
    outgoing_citations_.clear();
    return count;
  }

#if DISTRIBUTED_HAS_MPI
  // Multi-Rank MPI Exchange
  std::vector<std::vector<int32_t>> send_bufs(world_size);
  for (const auto &cit : outgoing_citations_) {
    int target_cluster = db.Nodes().GetClusterId(cit.target_node);
    int target_rank = (target_cluster >= 0) ? (target_cluster % world_size) : 0;
    if (target_rank < 0 || target_rank >= world_size) target_rank = 0;
    send_bufs[target_rank].push_back(cit.source_node);
    send_bufs[target_rank].push_back(cit.target_node);
    send_bufs[target_rank].push_back(static_cast<int32_t>(cit.year));
  }

  std::vector<int> send_counts(world_size, 0);
  for (int r = 0; r < world_size; ++r) {
    send_counts[r] = static_cast<int>(send_bufs[r].size());
  }

  std::vector<int> recv_counts(world_size, 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               MPI_COMM_WORLD);

  std::vector<int> sdispls(world_size, 0);
  std::vector<int> rdispls(world_size, 0);
  int total_send = 0;
  int total_recv = 0;
  for (int r = 0; r < world_size; ++r) {
    sdispls[r] = total_send;
    total_send += send_counts[r];
    rdispls[r] = total_recv;
    total_recv += recv_counts[r];
  }

  std::vector<int32_t> flat_send(total_send);
  int offset = 0;
  for (int r = 0; r < world_size; ++r) {
    if (send_counts[r] > 0) {
      std::memcpy(flat_send.data() + offset, send_bufs[r].data(),
                  send_counts[r] * sizeof(int32_t));
      offset += send_counts[r];
    }
  }

  std::vector<int32_t> flat_recv(total_recv);
  MPI_Alltoallv(flat_send.data(), send_counts.data(), sdispls.data(), MPI_INT32_T,
                flat_recv.data(), recv_counts.data(), rdispls.data(), MPI_INT32_T,
                MPI_COMM_WORLD);

  size_t citations_ingested = 0;
  for (size_t i = 0; i + 2 < flat_recv.size(); i += 3) {
    int32_t src = flat_recv[i];
    int32_t tgt = flat_recv[i + 1];
    db.CSR().RecordDeltaIncomingCitation(0, tgt, src);
    citations_ingested++;
  }

  outgoing_citations_.clear();
  return citations_ingested;
#endif

  outgoing_citations_.clear();
  return 0;
}

} // namespace distributed
