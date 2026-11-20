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
constexpr int kSemanticDims = 78;
constexpr int kBlockSize = 16;
constexpr int kPatchSize = 2;
constexpr int kInferSteps = 8;
constexpr int kVocabMaskClasses = 5;
constexpr int kDemoVocabSize = 512;  // Resident role mask table for head verification
constexpr double kCfgScale = 5.0;
constexpr double kLogitNormalLoc = 1.0;
constexpr int kDtypeBytes = 2;
constexpr std::size_t kL3LimitBytes = 3840000;
constexpr std::size_t kLayer2Ref2BBytes = 4000000000ULL;
constexpr double kMinTps = 500.0;

struct BenchResult {
  std::string condition;
  int num_tokens;
  int num_latents;
  std::size_t head_working_set_bytes;
  std::size_t typeclass_mask_bytes;
  std::size_t backbone_ref_bytes;
  double measured_tps;
  double latent_l2_residual;
  bool boundary_aligned;
  bool gate_b_head_l3_pass;
  bool fallback_to_cram;
};

double LogitNormalSigma(int step, int total_steps, double loc) {
  double u = static_cast<double>(total_steps - step) / static_cast<double>(total_steps + 1);
  u = std::max(1e-6, std::min(1.0 - 1e-6, u));
  double z = std::log(u / (1.0 - u)) + loc;
  return 1.0 / (1.0 + std::exp(-z));
}

std::size_t EstimateHeadWorkingSetBytes(int num_latents, bool with_typeclass_mask) {
  const std::size_t per_latent = static_cast<std::size_t>(kManifoldDim) * 8ULL;
  const std::size_t latent_bytes = static_cast<std::size_t>(num_latents) * per_latent * kDtypeBytes;
  const std::size_t proj_bytes = static_cast<std::size_t>(kManifoldDim * kManifoldDim * 4) * kDtypeBytes;
  const std::size_t mask_bytes = with_typeclass_mask
      ? static_cast<std::size_t>(kVocabMaskClasses * kDemoVocabSize)
      : 0ULL;
  return latent_bytes + proj_bytes + mask_bytes;
}

BenchResult RunCondition(
    const std::string& cond,
    int num_tokens,
    bool aligned,
    bool with_typeclass_mask,
    int iters) {
  const int n_latents = (num_tokens + kPatchSize - 1) / kPatchSize;
  const int n_blocks = (n_latents + kBlockSize - 1) / kBlockSize;
  const double boundary_bias = aligned ? 0.0 : 0.85;

  std::vector<float> clean_sem(static_cast<std::size_t>(n_latents) * kSemanticDims);
  std::vector<float> state(static_cast<std::size_t>(n_latents) * kSemanticDims);
  std::vector<uint8_t> role_mask(static_cast<std::size_t>(kVocabMaskClasses * kDemoVocabSize), 1);
  for (int i = 0; i < n_latents; ++i) {
    for (int d = 0; d < kSemanticDims; ++d) {
      clean_sem[static_cast<std::size_t>(i) * kSemanticDims + d] =
          static_cast<float>(std::cos((i * 13 + d) * 0.013));
    }
  }

  double total_sec = 0.0;
  double last_l2 = 0.0;
  volatile int sink_token = 0;

  for (int run = 0; run < iters; ++run) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_latents; ++i) {
      const float init_noise = (i < n_latents / 2) ? 0.0f : static_cast<float>(0.25 + boundary_bias);
      for (int d = 0; d < kSemanticDims; ++d) {
        const std::size_t idx = static_cast<std::size_t>(i) * kSemanticDims + d;
        state[idx] = clean_sem[idx] + init_noise;
      }
    }

    for (int step = 1; step <= kInferSteps; ++step) {
      const double sigma = LogitNormalSigma(step, kInferSteps, kLogitNormalLoc);
      const float alpha = static_cast<float>(std::min(0.95, 0.12 * kCfgScale * (1.0 - 0.5 * sigma)));
      for (int blk = 0; blk < n_blocks; ++blk) {
        const int start = blk * kBlockSize;
        const int end = std::min(n_latents, start + kBlockSize);
        for (int i = start; i < end; ++i) {
          if (i < n_latents / 2) continue;
          const std::size_t base = static_cast<std::size_t>(i) * kSemanticDims;
          for (int d = 0; d < kSemanticDims; ++d) {
            const float target = clean_sem[base + d] + static_cast<float>(boundary_bias);
            state[base + d] -= alpha * (state[base + d] - target);
          }
        }
      }
    }

    // Apply Layer-1 Type-Class Logit Masking pass over decoded token slots
    if (with_typeclass_mask) {
      int acc = 0;
      for (int t = 0; t < num_tokens; ++t) {
        const int role = t % kVocabMaskClasses;
        const std::size_t row = static_cast<std::size_t>(role) * kDemoVocabSize;
        float best_val = -1e9f;
        int best_id = 0;
        const float seed = state[static_cast<std::size_t>(t / kPatchSize) * kSemanticDims];
        for (int v = 0; v < kDemoVocabSize; ++v) {
          if (!role_mask[row + v]) continue;
          const float score = seed * static_cast<float>((v & 7) - 3);
          if (score > best_val) {
            best_val = score;
            best_id = v;
          }
        }
        acc ^= best_id;
      }
      sink_token = acc;
    }

    auto t1 = std::chrono::steady_clock::now();
    total_sec += std::chrono::duration<double>(t1 - t0).count();

    if (run == iters - 1) {
      double sq = 0.0;
      for (std::size_t k = 0; k < state.size(); ++k) {
        const double diff = static_cast<double>(state[k]) - static_cast<double>(clean_sem[k]);
        sq += diff * diff;
      }
      last_l2 = std::sqrt(sq / static_cast<double>(state.size()));
    }
  }
  (void)sink_token;

  const double avg_sec = std::max(1e-9, total_sec / static_cast<double>(iters));
  const double tps = static_cast<double>(num_tokens) / avg_sec;
  const std::size_t mask_bytes = with_typeclass_mask
      ? static_cast<std::size_t>(kVocabMaskClasses * kDemoVocabSize)
      : 0ULL;
  const std::size_t ws = EstimateHeadWorkingSetBytes(n_latents, with_typeclass_mask);
  const bool gate_b = (ws <= kL3LimitBytes) && (tps >= kMinTps);
  return BenchResult{cond, num_tokens, n_latents, ws, mask_bytes, kLayer2Ref2BBytes, tps, last_l2, aligned, gate_b, !gate_b};
}

}  // namespace

int main() {
  const int kIters = 200;
  std::vector<BenchResult> results = {
      RunCondition("Mod0", 512, true, false, kIters),
      RunCondition("Mod1", 513, false, false, kIters),
      RunCondition("AST-Sync", 514, true, false, kIters),
      RunCondition("AST-TypeClass-Sync", 514, true, true, kIters),
  };

  std::cout << std::fixed << std::setprecision(4) << "[\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    std::cout << "  {\n"
              << "    \"condition\": \"" << r.condition << "\",\n"
              << "    \"num_tokens\": " << r.num_tokens << ",\n"
              << "    \"num_latents\": " << r.num_latents << ",\n"
              << "    \"layer1_head_working_set_bytes\": " << r.head_working_set_bytes << ",\n"
              << "    \"typeclass_mask_bytes\": " << r.typeclass_mask_bytes << ",\n"
              << "    \"layer1_l3_utilization_ratio\": "
              << static_cast<double>(r.head_working_set_bytes) / static_cast<double>(kL3LimitBytes) << ",\n"
              << "    \"layer2_backbone_ref_bytes\": " << r.backbone_ref_bytes << ",\n"
              << "    \"layer1_measured_tps\": " << r.measured_tps << ",\n"
              << "    \"latent_l2_residual\": " << r.latent_l2_residual << ",\n"
              << "    \"boundary_aligned\": " << (r.boundary_aligned ? "true" : "false") << ",\n"
              << "    \"gate_b_head_l3_pass\": " << (r.gate_b_head_l3_pass ? "true" : "false") << ",\n"
              << "    \"fallback_to_cram\": " << (r.fallback_to_cram ? "true" : "false") << "\n"
              << "  }" << (i + 1 < results.size() ? "," : "") << "\n";
  }
  std::cout << "]\n";
  return 0;
}
