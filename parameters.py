
from dataclasses import dataclass

@dataclass
class GpuParameters:
    architecture: str
    cuda_cores_fp32: int
    tensor_cores_gen: int
    memory_type: str
    memory_capacity_gb: int
    memory_bandwidth_tb_s: float
    l2_cache_mb: int
    nvlink_bandwidth_gb_s: float
    pcie_gen: int
    # SM specific parameters (simplified for now, will refine later)
    num_sms: int
    max_warps_per_sm: int # Max resident warps per SM
    warp_size: int = 32 # Threads per warp

@dataclass
class StorageParameters:
    sequential_read_bandwidth_gb_s: float
    sequential_write_bandwidth_gb_s: float
    random_read_iops_k: float # in thousands
    random_write_iops_k: float # in thousands
    latency_us: float # in microseconds

# Defined Parameters
H100_PARAMS = GpuParameters(
    architecture="Hopper",
    cuda_cores_fp32=16896,
    tensor_cores_gen=4,
    memory_type="HBM3",
    memory_capacity_gb=80,
    memory_bandwidth_tb_s=3.35,
    l2_cache_mb=50,
    nvlink_bandwidth_gb_s=900,
    pcie_gen=5,
    num_sms=132, # Based on 16896 CUDA cores / 128 cores per SM
    max_warps_per_sm=64 # Typical for Hopper
)

B200_PARAMS = GpuParameters(
    architecture="Blackwell",
    cuda_cores_fp32=36000, # Placeholder
    tensor_cores_gen=5, # Placeholder
    memory_type="HBM3e", # Placeholder
    memory_capacity_gb=192, # Placeholder
    memory_bandwidth_tb_s=8.0, # Placeholder
    l2_cache_mb=128, # Placeholder
    nvlink_bandwidth_gb_s=1800, # Placeholder
    pcie_gen=5,
    num_sms=288, # Placeholder
    max_warps_per_sm=128 # Placeholder (assuming higher for Blackwell)
)

PCIe5_NVME_SSD_PARAMS = StorageParameters(
    sequential_read_bandwidth_gb_s=14.5,
    sequential_write_bandwidth_gb_s=14.0,
    random_read_iops_k=2300, # 2.3 million
    random_write_iops_k=1600, # 1.6 million
    latency_us=50 # Average latency
)
