/**
 * InfiniCCL Example: Reduce (OpenMPI + CCL Hybrid)
 *
 * This example verifies Reduce through an OpenMPI fallback, then initializes
 * a native CCL communicator and performs Reduce on GPU memory.
 */

#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "backend_manifest.h"
#include "device.h"
#include "infiniccl.h"
#include "runtime.h"
#include "traits.h"
#include "utils.h"

using namespace infini::ccl;

constexpr int kRoot = 0;
constexpr float kSentinel = -1.0f;

float ExpectedValue(int size) {
  return static_cast<float>(size) * (size + 1) / 2.0f;
}

bool ValidateResult(const std::vector<float> &result, float expected) {
  for (size_t i = 0; i < result.size(); ++i) {
    if (!std::isfinite(result[i]) || std::fabs(result[i] - expected) > 1e-3f) {
      std::cerr << "Reduce mismatch at index " << i << ": expected " << expected
                << ", got " << result[i] << std::endl;
      return false;
    }
  }
  return true;
}

void PrintMetrics(double elapsed_ms, size_t total_bytes) {
  const double seconds = elapsed_ms / 1000.0;
  const double gigabytes =
      static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
  const double algorithm_bandwidth = gigabytes / seconds;

  // Reduce has a bus-bandwidth correction factor of 1.
  std::cout << "Time:           " << std::fixed << std::setprecision(3)
            << elapsed_ms << " ms" << std::endl;
  std::cout << "Throughput:     " << std::fixed << std::setprecision(2)
            << algorithm_bandwidth << " GB/s (Bus BW)" << std::endl;
  std::cout << "Alg Bandwidth:  " << std::fixed << std::setprecision(2)
            << algorithm_bandwidth << " GB/s" << std::endl;
}

bool RunReduceExample(int argc, char **argv, int warmup_iter, int profile_iter,
                      size_t num_elements) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_INFINI(infinicclInit(&argc, &argv));

  int rank, size;
  CHECK_INFINI(infinicclGetRank(&rank));
  CHECK_INFINI(infinicclGetSize(&size));
  if (size <= 0) {
    std::cerr << "Invalid world size for Reduce." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const char *local_rank_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  const int local_rank =
      local_rank_str == nullptr ? 0 : std::atoi(local_rank_str);
  CHECK_RT(Rt, Rt::SetDevice(local_rank));

  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  // Verify Reduce through the OpenMPI communicator before native CCL setup.
  const float bootstrap_send = static_cast<float>(rank + 1);
  float bootstrap_recv = kSentinel;
  float *d_bootstrap_send = nullptr;
  float *d_bootstrap_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_bootstrap_send),
                          sizeof(float)));
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_bootstrap_recv),
                            sizeof(float)));
  }
  CHECK_RT(Rt, Rt::Memcpy(d_bootstrap_send, &bootstrap_send, sizeof(float),
                          Rt::MemcpyHostToDevice));
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Memcpy(d_bootstrap_recv, &bootstrap_recv, sizeof(float),
                            Rt::MemcpyHostToDevice));
  }
  CHECK_INFINI(infinicclReduce(d_bootstrap_send, d_bootstrap_recv, 1,
                               infinicclFloat32, infinicclSum, kRoot, comm,
                               nullptr));
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Memcpy(&bootstrap_recv, d_bootstrap_recv, sizeof(float),
                            Rt::MemcpyDeviceToHost));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  int32_t validation_status =
      rank != kRoot ||
              (std::isfinite(bootstrap_recv) &&
               std::fabs(bootstrap_recv - ExpectedValue(size)) <= 1e-3f)
          ? 1
          : 0;
  CHECK_INFINI(infinicclBroadcast(&validation_status, &validation_status, 1,
                                  infinicclInt32, kRoot, comm, nullptr));

  CHECK_RT(Rt, Rt::Free(d_bootstrap_send));
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Free(d_bootstrap_recv));
  }

  infinicclUniqueId id;
  if (rank == kRoot) {
    CHECK_INFINI(infinicclGetUniqueId(&id));
  }
  CHECK_INFINI(infinicclBroadcast(&id, &id, sizeof(id), infinicclChar, kRoot,
                                  comm, nullptr));

  std::cout << "[Rank " << rank << "] Host: " << hostname
            << " | GPU: " << Device::StringFromType(kDevType) << " "
            << " | Device " << local_rank << std::endl;

  CHECK_INFINI(infinicclCommInitRank(&comm, size, id, rank));

  const bool is_root = rank == kRoot;
  const size_t total_bytes = num_elements * sizeof(float);
  std::vector<float> h_send(num_elements, static_cast<float>(rank + 1));
  std::vector<float> h_recv(is_root ? num_elements : 0, kSentinel);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), total_bytes));
  if (is_root) {
    CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), total_bytes));
  }
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  if (is_root) {
    std::cout << "\n=== Performing Reduce on GPU Memory ===" << std::endl;
    std::cout << "Data size: " << num_elements << " floats ("
              << total_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Operation: Sum" << std::endl;
    std::cout << "Root Rank: " << kRoot << std::endl;
    std::cout << "Warm-up iterations: " << warmup_iter << std::endl;
    std::cout << "Profile iterations: " << profile_iter << std::endl;
  }

  for (int i = 0; i < warmup_iter; ++i) {
    CHECK_INFINI(infinicclReduce(d_send, d_recv, num_elements, infinicclFloat32,
                                 infinicclSum, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < profile_iter; ++i) {
    CHECK_INFINI(infinicclReduce(d_send, d_recv, num_elements, infinicclFloat32,
                                 infinicclSum, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed_ms =
      timer.ElapsedMs() / static_cast<double>(profile_iter);

  // Validate a separate untimed reduction with a sentinel-filled root output.
  if (is_root) {
    CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                            Rt::MemcpyHostToDevice));
  }
  CHECK_INFINI(infinicclReduce(d_send, d_recv, num_elements, infinicclFloat32,
                               infinicclSum, kRoot, comm, nullptr));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  if (is_root) {
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    const bool correct =
        validation_status == 1 && ValidateResult(h_recv, ExpectedValue(size));
    validation_status = correct ? 1 : 0;

    std::cout << "\n=== Hybrid CCL Reduce Results ===" << std::endl;
    std::cout << "Correct: " << (correct ? "YES" : "NO") << std::endl;
    PrintMetrics(elapsed_ms, total_bytes);
  }

  CHECK_INFINI(infinicclBroadcast(&validation_status, &validation_status, 1,
                                  infinicclInt32, kRoot, comm, nullptr));

  CHECK_RT(Rt, Rt::Free(d_send));
  if (is_root) {
    CHECK_RT(Rt, Rt::Free(d_recv));
  }
  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (is_root) {
    std::cout << "InfiniCCL finalized." << std::endl;
  }
  return validation_status == 1;
}

int main(int argc, char **argv) {
  constexpr int kWarmupIterations = 2;
  constexpr int kProfileIterations = 20;
  constexpr size_t kNumElements = 1 << 20;

  if (kNumElements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "Reduce buffer size is invalid." << std::endl;
    return EXIT_FAILURE;
  }

  return RunReduceExample(argc, argv, kWarmupIterations, kProfileIterations,
                          kNumElements)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
