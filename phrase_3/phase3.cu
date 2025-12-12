
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <chrono>
#include <cmath>

constexpr int IN_DIM  = 64;
constexpr int H_DIM   = 8;
constexpr int OUT_DIM = 10;

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// ------------------- small helpers -------------------
static inline std::string trim(const std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool load_int8_mem(const std::string &path, std::vector<int8_t> &out, int expected_len) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }
    out.clear();
    std::string line;
    while (std::getline(fin, line)) {
        line = trim(line);
        if (line.empty()) continue;
        long v = std::strtol(line.c_str(), nullptr, 16);
        out.push_back(static_cast<int8_t>(static_cast<int32_t>(v)));
    }
    if (expected_len > 0 && static_cast<int>(out.size()) != expected_len) {
        std::cerr << "Warning: file " << path << " expected "
                  << expected_len << " entries, got " << out.size() << std::endl;
    }
    return true;
}

bool load_int32_mem(const std::string &path, std::vector<int32_t> &out, int expected_len) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }
    out.clear();
    std::string line;
    while (std::getline(fin, line)) {
        line = trim(line);
        if (line.empty()) continue;
        unsigned long u = std::strtoul(line.c_str(), nullptr, 16);
        out.push_back(static_cast<int32_t>(u));
    }
    if (expected_len > 0 && static_cast<int>(out.size()) != expected_len) {
        std::cerr << "Warning: file " << path << " expected "
                  << expected_len << " entries, got " << out.size() << std::endl;
    }
    return true;
}

// ------------------- CPU reference -------------------
void mlp_cpu(const int8_t* x,
             const int8_t* W1, const int32_t* b1,
             const int8_t* W2, const int32_t* b2,
             int32_t* z2_out) {
    int32_t z1[H_DIM];

    // layer1: 64 -> 8
    for (int j = 0; j < H_DIM; ++j) {
        int32_t acc = b1[j];
        for (int i = 0; i < IN_DIM; ++i) {
            int idx = i * H_DIM + j;
            acc += static_cast<int32_t>(x[i]) *
                   static_cast<int32_t>(W1[idx]);
        }
        if (acc < 0) acc = 0;  // ReLU
        z1[j] = acc;
    }

    // layer2: 8 -> 10
    for (int k = 0; k < OUT_DIM; ++k) {
        int32_t acc = b2[k];
        for (int j = 0; j < H_DIM; ++j) {
            int idx = j * OUT_DIM + k;
            acc += z1[j] * static_cast<int32_t>(W2[idx]);
        }
        z2_out[k] = acc;
    }
}

// ------------------- Single-sample CUDA kernels -------------------
__global__
void linear1_kernel(const int8_t* x,
                    const int8_t* W1,
                    const int32_t* b1,
                    int32_t* z1) {
    int j = threadIdx.x;
    if (j >= H_DIM) return;

    int32_t acc = b1[j];
    for (int i = 0; i < IN_DIM; ++i) {
        int idx = i * H_DIM + j;
        acc += static_cast<int32_t>(x[i]) *
               static_cast<int32_t>(W1[idx]);
    }
    if (acc < 0) acc = 0;
    z1[j] = acc;
}

__global__
void linear2_kernel(const int32_t* z1,
                    const int8_t* W2,
                    const int32_t* b2,
                    int32_t* z2) {
    int k = threadIdx.x;
    if (k >= OUT_DIM) return;

    int32_t acc = b2[k];
    for (int j = 0; j < H_DIM; ++j) {
        int idx = j * OUT_DIM + k;
        acc += z1[j] * static_cast<int32_t>(W2[idx]);
    }
    z2[k] = acc;
}


void mlp_gpu_single(const int8_t* h_x,
                    const int8_t* h_W1, const int32_t* h_b1,
                    const int8_t* h_W2, const int32_t* h_b2,
                    int32_t* h_z2,
                    float &kernel_ms) {

    int8_t  *d_x  = nullptr;
    int8_t  *d_W1 = nullptr;
    int32_t *d_b1 = nullptr;
    int32_t *d_z1 = nullptr;
    int8_t  *d_W2 = nullptr;
    int32_t *d_b2 = nullptr;
    int32_t *d_z2 = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x,  IN_DIM * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_W1, IN_DIM * H_DIM * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_b1, H_DIM * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_z1, H_DIM * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_W2, H_DIM * OUT_DIM * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_b2, OUT_DIM * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_z2, OUT_DIM * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(d_x,  h_x,  IN_DIM * sizeof(int8_t),          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1, h_W1, IN_DIM * H_DIM * sizeof(int8_t),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, h_b1, H_DIM * sizeof(int32_t),          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2, H_DIM * OUT_DIM * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2, h_b2, OUT_DIM * sizeof(int32_t),        cudaMemcpyHostToDevice));

    dim3 block1(H_DIM);
    dim3 block2(OUT_DIM);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    linear1_kernel<<<1, block1>>>(d_x, d_W1, d_b1, d_z1);
    CHECK_CUDA(cudaGetLastError());

    linear2_kernel<<<1, block2>>>(d_z1, d_W2, d_b2, d_z2);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_z2, d_z2, OUT_DIM * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_x);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_z1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_z2);
}

// ------------------- Batch kernels (many blocks) -------------------
// layer1 for batch: grid.x = batch_size, block.x = H_DIM
__global__
void linear1_batch_kernel(const int8_t* x_batch,   // [B, IN_DIM]
                          const int8_t* W1,        // [IN_DIM, H_DIM]
                          const int32_t* b1,       // [H_DIM]
                          int32_t* z1_batch,       // [B, H_DIM]
                          int B) {
    int n = blockIdx.x;      // sample index
    int j = threadIdx.x;     // neuron index
    if (n >= B || j >= H_DIM) return;

    const int8_t* x = x_batch + n * IN_DIM;
    int32_t* z1 = z1_batch + n * H_DIM;

    int32_t acc = b1[j];
    for (int i = 0; i < IN_DIM; ++i) {
        int idx = i * H_DIM + j;
        acc += static_cast<int32_t>(x[i]) *
               static_cast<int32_t>(W1[idx]);
    }
    if (acc < 0) acc = 0;
    z1[j] = acc;
}

// layer2 for batch: grid.x = B, block.x = OUT_DIM
__global__
void linear2_batch_kernel(const int32_t* z1_batch, // [B, H_DIM]
                          const int8_t* W2,        // [H_DIM, OUT_DIM]
                          const int32_t* b2,       // [OUT_DIM]
                          int32_t* z2_batch,       // [B, OUT_DIM]
                          int B) {
    int n = blockIdx.x;  // sample
    int k = threadIdx.x; // output neuron
    if (n >= B || k >= OUT_DIM) return;

    const int32_t* z1 = z1_batch + n * H_DIM;
    int32_t* z2 = z2_batch + n * OUT_DIM;

    int32_t acc = b2[k];
    for (int j = 0; j < H_DIM; ++j) {
        int idx = j * OUT_DIM + k;
        acc += z1[j] * static_cast<int32_t>(W2[idx]);
    }
    z2[k] = acc;
}


void batch_test_parallel(int batch_size,
                         const int8_t* h_W1, const int32_t* h_b1,
                         const int8_t* h_W2, const int32_t* h_b2) {

    std::cout << "\n==== Batch Test Parallel (B = " << batch_size << ") ====\n";

    
    std::vector<int8_t> x_batch(batch_size * IN_DIM);
    for (int i = 0; i < batch_size * IN_DIM; ++i) {
        x_batch[i] = static_cast<int8_t>((rand() % 7) - 3); // [-3,3]
    }

    // --------- CPU baseline ----------
    std::vector<int32_t> z2_cpu(batch_size * OUT_DIM);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < batch_size; ++n) {
        mlp_cpu(&x_batch[n * IN_DIM],
                h_W1, h_b1, h_W2, h_b2,
                &z2_cpu[n * OUT_DIM]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --------- GPU memory ----------
    int8_t  *d_x_batch = nullptr;
    int32_t *d_z1_batch = nullptr;
    int32_t *d_z2_batch = nullptr;
    int8_t  *d_W1 = nullptr;
    int32_t *d_b1 = nullptr;
    int8_t  *d_W2 = nullptr;
    int32_t *d_b2 = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x_batch, batch_size * IN_DIM  * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_z1_batch, batch_size * H_DIM  * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_z2_batch, batch_size * OUT_DIM * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_W1, IN_DIM * H_DIM * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_b1, H_DIM * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_W2, H_DIM * OUT_DIM * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_b2, OUT_DIM * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(d_x_batch, x_batch.data(),
                          batch_size * IN_DIM * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1, h_W1,
                          IN_DIM * H_DIM * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, h_b1,
                          H_DIM * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2,
                          H_DIM * OUT_DIM * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2, h_b2,
                          OUT_DIM * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    dim3 block1(H_DIM);
    dim3 block2(OUT_DIM);
    dim3 grid(batch_size);

    linear1_batch_kernel<<<grid, block1>>>(d_x_batch, d_W1, d_b1,
                                           d_z1_batch, batch_size);
    CHECK_CUDA(cudaGetLastError());

    linear2_batch_kernel<<<grid, block2>>>(d_z1_batch, d_W2, d_b2,
                                           d_z2_batch, batch_size);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));

    
    std::vector<int32_t> z2_gpu(batch_size * OUT_DIM);
    CHECK_CUDA(cudaMemcpy(z2_gpu.data(), d_z2_batch,
                          batch_size * OUT_DIM * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < batch_size * OUT_DIM; ++i) {
        if (z2_cpu[i] != z2_gpu[i]) mismatches++;
    }

    std::cout << "CPU total time       : " << cpu_ms << " ms\n";
    std::cout << "GPU pure-kernel time : " << gpu_ms << " ms\n";
    if (gpu_ms > 0.0f) {
        std::cout << "Speedup (CPU/GPU)    : " << (cpu_ms / gpu_ms) << "x\n";
        std::cout << "GPU throughput       : "
                  << (batch_size / (gpu_ms / 1000.0f))
                  << " samples/sec\n";
    }
    std::cout << "Mismatch count       : " << mismatches << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x_batch);
    cudaFree(d_z1_batch);
    cudaFree(d_z2_batch);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
}

// ------------------- print helper -------------------
template <typename T>
void print_vec(const T* v, int n, const std::string &name) {
    std::cout << name << " = [";
    for (int i = 0; i < n; ++i) {
        std::cout << v[i];
        if (i + 1 < n) std::cout << ", ";
    }
    std::cout << "]\n";
}

// ------------------- main -------------------
int main() {
    std::vector<int8_t>  W1_q, W2_q, x_q;
    std::vector<int32_t> b1_q, b2_q;

    if (!load_int8_mem("W1_q.mem", W1_q, IN_DIM * H_DIM)) return 1;
    if (!load_int8_mem("W2_q.mem", W2_q, H_DIM * OUT_DIM)) return 1;
    if (!load_int32_mem("b1_q.mem", b1_q, H_DIM)) return 1;
    if (!load_int32_mem("b2_q.mem", b2_q, OUT_DIM)) return 1;
    if (!load_int8_mem("x_sample.mem", x_q, IN_DIM)) return 1;

    std::cout << "Loaded params: W1_q(" << W1_q.size()
              << "), b1_q(" << b1_q.size()
              << "), W2_q(" << W2_q.size()
              << "), b2_q(" << b2_q.size()
              << "), x_q("  << x_q.size()  << ")\n\n";

    // ---------- simple CPU vs GPU ----------
    int32_t z2_cpu[OUT_DIM];

    auto t0 = std::chrono::high_resolution_clock::now();
    mlp_cpu(x_q.data(), W1_q.data(), b1_q.data(),
            W2_q.data(), b2_q.data(), z2_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int32_t z2_gpu[OUT_DIM];
    float kernel_ms = 0.0f;
    mlp_gpu_single(x_q.data(),
                   W1_q.data(), b1_q.data(),
                   W2_q.data(), b2_q.data(),
                   z2_gpu, kernel_ms);

    print_vec(z2_cpu, OUT_DIM, "z2_cpu");
    print_vec(z2_gpu, OUT_DIM, "z2_gpu");

    int pred_cpu = 0, pred_gpu = 0;
    int32_t best = z2_cpu[0];
    for (int i = 1; i < OUT_DIM; ++i)
        if (z2_cpu[i] > best) { best = z2_cpu[i]; pred_cpu = i; }
    best = z2_gpu[0];
    for (int i = 1; i < OUT_DIM; ++i)
        if (z2_gpu[i] > best) { best = z2_gpu[i]; pred_gpu = i; }

    int32_t max_diff = 0;
    for (int i = 0; i < OUT_DIM; ++i) {
        int32_t d = std::abs(z2_cpu[i] - z2_gpu[i]);
        if (d > max_diff) max_diff = d;
    }

    std::cout << "\nPrediction (CPU): " << pred_cpu << "\n";
    std::cout << "Prediction (GPU): " << pred_gpu << "\n";
    std::cout << "Max |CPU - GPU| diff: " << max_diff << "\n\n";

    std::cout << "CPU time   : " << cpu_ms    << " ms (single sample)\n";
    std::cout << "GPU kernels: " << kernel_ms << " ms (single sample)\n";

    int BATCH = 32768;  
    batch_test_parallel(BATCH,
                        W1_q.data(), b1_q.data(),
                        W2_q.data(), b2_q.data());

    return 0;
}

