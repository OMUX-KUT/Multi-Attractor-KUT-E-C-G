#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kManifoldDim = 110;
constexpr int kPatchSize = 2;
constexpr int kNumRoles = 5;
constexpr int kTopK = 256;
constexpr int kNumTokens = 514;
constexpr int kDtypeBytes = 2;
constexpr std::size_t kL3LimitBytes = 3840000;
constexpr double kMinTps = 500.0;

struct VocabBenchRow {
  int vocab_size;
  std::string mode;
  std::size_t head_state_bytes;
  std::size_t bitpacked_mask_bytes;
  std::size_t logit_or_topk_bytes;
  std::size_t total_layer1_bytes;
  double l3_utilization_ratio;
  double measured_tps;
  bool gate_l3_pass;
  bool gate_tps_pass;
  bool phase2a_pass;
};

std::size_t BaseHeadStateBytes(int n_latents) {
  const std::size_t latent_bytes = static_cast<std::size_t>(n_latents) * kManifoldDim * 8ULL * kDtypeBytes;
  const std::size_t proj_bytes = static_cast<std::size_t>(kManifoldDim * kManifoldDim * 4) * kDtypeBytes;
  return latent_bytes + proj_bytes;
}

VocabBenchRow RunMode(int vocab_size, bool sparse_topk_mode, int iters) {
  const int n_latents = (kNumTokens + kPatchSize - 1) / kPatchSize;
  const int words_per_role = (vocab_size + 63) / 64;

  std::vector<uint64_t> bitmasks(static_cast<std::size_t>(kNumRoles) * words_per_role, 0ULL);
  for (int r = 0; r < kNumRoles; ++r) {
    const int stride = (r == 0) ? 512 : (r == 1 ? 2048 : (r == 2 ? 1024 : 2));
    for (int v = 0; v < vocab_size; v += stride) {
      bitmasks[static_cast<std::size_t>(r) * words_per_role + (v >> 6)] |= (1ULL << (v & 63));
    }
  }

  std::vector<float> logits(static_cast<std::size_t>(vocab_size));
  for (int v = 0; v < vocab_size; ++v) {
    logits[v] = std::sin(static_cast<float>(v) * 0.0017f);
  }

  std::vector<uint32_t> topk_ids(kTopK);
  std::vector<float> topk_scores(kTopK);
  for (int k = 0; k < kTopK; ++k) {
    topk_ids[k] = static_cast<uint32_t>((k * 499) % vocab_size);
    topk_scores[k] = logits[topk_ids[k]];
  }

  volatile uint32_t sink = 0;
  double total_sec = 0.0;

  for (int run = 0; run < iters; ++run) {
    auto t0 = std::chrono::steady_clock::now();
    uint32_t acc = 0;

    for (int t = 0; t < kNumTokens; ++t) {
      const int role = t % kNumRoles;
      const std::size_t row_off = static_cast<std::size_t>(role) * words_per_role;
      float best_val = -1e9f;
      uint32_t best_id = 0;

      if (sparse_topk_mode) {
        for (int k = 0; k < kTopK; ++k) {
          const uint32_t vid = topk_ids[k];
          const uint64_t word = bitmasks[row_off + (vid >> 6)];
          if ((word >> (vid & 63)) & 1ULL) {
            const float sc = topk_scores[k];
            if (sc > best_val) {
              best_val = sc;
              best_id = vid;
            }
          }
        }
      } else {
        for (int w = 0; w < words_per_role; ++w) {
          uint64_t bits = bitmasks[row_off + w];
          const int base_v = w << 6;
          while (bits) {
            const int bit = __builtin_ctzll(bits);
            const int vid = base_v + bit;
            if (vid < vocab_size) {
              const float sc = logits[vid];
              if (sc > best_val) {
                best_val = sc;
                best_id = static_cast<uint32_t>(vid);
              }
            }
            bits &= (bits - 1);
          }
        }
      }
      acc ^= best_id;
    }

    sink = acc;
    auto t1 = std::chrono::steady_clock::now();
    total_sec += std::chrono::duration<double>(t1 - t0).count();
  }
  (void)sink;

  const double avg_sec = std::max(1e-9, total_sec / static_cast<double>(iters));
  const double tps = static_cast<double>(kNumTokens) / avg_sec;
  const std::size_t head_bytes = BaseHeadStateBytes(n_latents);
  const std::size_t mask_bytes = bitmasks.size() * sizeof(uint64_t);
  const std::size_t logit_bytes = sparse_topk_mode
      ? static_cast<std::size_t>(kTopK) * (sizeof(uint32_t) + kDtypeBytes)
      : static_cast<std::size_t>(vocab_size) * kDtypeBytes;
  const std::size_t total_bytes = head_bytes + mask_bytes + logit_bytes;
  const bool l3_ok = total_bytes <= kL3LimitBytes;
  const bool tps_ok = tps >= kMinTps;

  return VocabBenchRow{
      vocab_size,
      sparse_topk_mode ? "BitPacked-SparseTopK256" : "BitPacked-ExactFullV",
      head_bytes,
      mask_bytes,
      logit_bytes,
      total_bytes,
      static_cast<double>(total_bytes) / static_cast<double>(kL3LimitBytes),
      tps,
      l3_ok,
      tps_ok,
      l3_ok && tps_ok,
  };
}

}  // namespace

int main() {
  const std::vector<int> vocabs = {32768, 65536, 131072};
  std::vector<VocabBenchRow> rows;
  for (int v : vocabs) {
    rows.push_back(RunMode(v, false, 15));
    rows.push_back(RunMode(v, true, 100));
  }

  std::cout << std::fixed << std::setprecision(4) << "[\n";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    std::cout << "  {\n"
              << "    \"vocab_size\": " << r.vocab_size << ",\n"
              << "    \"mode\": \"" << r.mode << "\",\n"
              << "    \"head_state_bytes\": " << r.head_state_bytes << ",\n"
              << "    \"bitpacked_mask_bytes\": " << r.bitpacked_mask_bytes << ",\n"
              << "    \"logit_or_topk_bytes\": " << r.logit_or_topk_bytes << ",\n"
              << "    \"total_layer1_bytes\": " << r.total_layer1_bytes << ",\n"
              << "    \"l3_utilization_ratio\": " << r.l3_utilization_ratio << ",\n"
              << "    \"measured_tps\": " << r.measured_tps << ",\n"
              << "    \"gate_l3_pass\": " << (r.gate_l3_pass ? "true" : "false") << ",\n"
              << "    \"gate_tps_pass\": " << (r.gate_tps_pass ? "true" : "false") << ",\n"
              << "    \"phase2a_pass\": " << (r.phase2a_pass ? "true" : "false") << "\n"
              << "  }" << (i + 1 < rows.size() ? "," : "") << "\n";
  }
  std::cout << "]\n";
  return 0;
}
