#include <torch/extension.h>

void launch_fused_entropy_temperature(
    torch::Tensor& logits,
    torch::Tensor& prev_entropies,
    torch::Tensor& patience_counts,
    torch::Tensor& output_entropies,
    float base_temp,
    float beta,
    int patience_cap,
    float entropy_floor,
    float marginal_threshold
);

void apply_geometric_annealing(
    torch::Tensor logits,
    torch::Tensor prev_entropies,
    torch::Tensor patience_counts,
    torch::Tensor output_entropies,
    double base_temp,
    double beta,
    int64_t patience_cap,
    double entropy_floor,
    double marginal_threshold
) {
    launch_fused_entropy_temperature(
        logits,
        prev_entropies,
        patience_counts,
        output_entropies,
        static_cast<float>(base_temp),
        static_cast<float>(beta),
        static_cast<int>(patience_cap),
        static_cast<float>(entropy_floor),
        static_cast<float>(marginal_threshold)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_geometric_annealing", &apply_geometric_annealing, "Fused Entropy and Geometric Annealing Kernel");
}
