#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
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
  std::string vocab_label;
  std::string mask_source;
  int vocab_size;
  std::string mode;
  std::size_t head_state_bytes;
  std::size_t bitpacked_mask_bytes;
  std::size_t logit_or_topk_bytes;
  std::size_t total_layer1_bytes;
  double l3_utilization_ratio;
  double active_bit_density;
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

std::vector<uint64_t> BuildSyntheticMask(int vocab_size) {
  const int words_per_role = (vocab_size + 63) / 64;
  std::vector<uint64_t> bitmasks(static_cast<std::size_t>(kNumRoles) * words_per_role, 0ULL);
  for (int r = 0; r < kNumRoles; ++r) {
    const int stride = (r == 0) ? 512 : (r == 1 ? 2048 : (r == 2 ? 1024 : 2));
    for (int v = 0; v < vocab_size; v += stride) {
      bitmasks[static_cast<std::size_t>(r) * words_per_role + (v >> 6)] |= (1ULL << (v & 63));
    }
  }
  return bitmasks;
}

bool LoadExternalMask(const std::string& path, int vocab_size, std::vector<uint64_t>& out) {
  const std::size_t words_per_role = static_cast<std::size_t>((vocab_size + 63) / 64);
  const std::size_t expected_words = static_cast<std::size_t>(kNumRoles) * words_per_role;
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) return false;
  out.resize(expected_words);
  ifs.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(expected_words * sizeof(uint64_t)));
  return static_cast<std::size_t>(ifs.gcount()) == expected_words * sizeof(uint64_t);
}

VocabBenchRow RunMode(
    const std::string& vocab_label,
    const std::string& mask_source,
    int vocab_size,
    const std::vector<uint64_t>& bitmasks,
    bool sparse_topk_mode,
    int iters) {
  const int n_latents = (kNumTokens + kPatchSize - 1) / kPatchSize;
  const int words_per_role = (vocab_size + 63) / 64;

  uint64_t set_bits = 0;
  for (uint64_t w : bitmasks) {
    set_bits += static_cast<uint64_t>(__builtin_popcountll(w));
  }
  const double density = static_cast<double>(set_bits) /
      static_cast<double>(static_cast<uint64_t>(kNumRoles) * static_cast<uint64_t>(vocab_size));

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
  std::vector<double> run_secs;
  run_secs.reserve(static_cast<std::size_t>(iters));
  const int warmup = 5;

  for (int run = -warmup; run < iters; ++run) {
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
    if (run >= 0) {
      run_secs.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
  }
  (void)sink;

  std::sort(run_secs.begin(), run_secs.end());
  const double avg_sec = std::max(1e-9, run_secs[run_secs.size() / 2]);
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
      vocab_label,
      mask_source,
      vocab_size,
      sparse_topk_mode ? "BitPacked-SparseTopK256" : "BitPacked-ExactFullV",
      head_bytes,
      mask_bytes,
      logit_bytes,
      total_bytes,
      static_cast<double>(total_bytes) / static_cast<double>(kL3LimitBytes),
      density,
      tps,
      l3_ok,
      tps_ok,
      l3_ok && tps_ok,
  };
}

void PrintRows(const std::vector<VocabBenchRow>& rows) {
  std::cout << std::fixed << std::setprecision(4) << "[\n";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    std::cout << "  {\n"
              << "    \"vocab_label\": \"" << r.vocab_label << "\",\n"
              << "    \"mask_source\": \"" << r.mask_source << "\",\n"
              << "    \"vocab_size\": " << r.vocab_size << ",\n"
              << "    \"mode\": \"" << r.mode << "\",\n"
              << "    \"head_state_bytes\": " << r.head_state_bytes << ",\n"
              << "    \"bitpacked_mask_bytes\": " << r.bitpacked_mask_bytes << ",\n"
              << "    \"logit_or_topk_bytes\": " << r.logit_or_topk_bytes << ",\n"
              << "    \"total_layer1_bytes\": " << r.total_layer1_bytes << ",\n"
              << "    \"l3_utilization_ratio\": " << r.l3_utilization_ratio << ",\n"
              << "    \"active_bit_density\": " << r.active_bit_density << ",\n"
              << "    \"measured_tps\": " << r.measured_tps << ",\n"
              << "    \"gate_l3_pass\": " << (r.gate_l3_pass ? "true" : "false") << ",\n"
              << "    \"gate_tps_pass\": " << (r.gate_tps_pass ? "true" : "false") << ",\n"
              << "    \"phase2a_pass\": " << (r.phase2a_pass ? "true" : "false") << "\n"
              << "  }" << (i + 1 < rows.size() ? "," : "") << "\n";
  }
  std::cout << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<VocabBenchRow> rows;
  if (argc == 4 && std::string(argv[1]) == "--real-mask") {
    const std::string label_and_v = argv[2];
    const std::string path = argv[3];
    const auto sep = label_and_v.find(':');
    const std::string label = label_and_v.substr(0, sep);
    const int vocab_size = std::stoi(label_and_v.substr(sep + 1));
    std::vector<uint64_t> mask;
    if (!LoadExternalMask(path, vocab_size, mask)) {
      std::cerr << "Failed to load real mask: " << path << "\n";
      return 2;
    }
    rows.push_back(RunMode(label, "real_tokenizer_vocab", vocab_size, mask, false, 15));
    rows.push_back(RunMode(label, "real_tokenizer_vocab", vocab_size, mask, true, 100));
    PrintRows(rows);
    return 0;
  }

  const std::vector<std::pair<std::string, int>> vocabs = {
      {"32k", 32768},
      {"64k", 65536},
      {"128k", 131072},
      {"200k_o200k_scale", 200019},
      {"256k_gemma_gemini_scale", 256000},
  };
  for (const auto& [label, v] : vocabs) {
    auto mask = BuildSyntheticMask(v);
    rows.push_back(RunMode(label, "synthetic_role_density", v, mask, false, 15));
    rows.push_back(RunMode(label, "synthetic_role_density", v, mask, true, 100));
  }
  PrintRows(rows);
  return 0;
}
