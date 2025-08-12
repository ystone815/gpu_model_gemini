

import simpy
from parameters import H100_PARAMS, B200_PARAMS, PCIe5_NVME_SSD_PARAMS
from system_components import System

def run_simulation(gpu_params, storage_params, simulation_time=1000):
    env = simpy.Environment()
    system = System(env, gpu_params, storage_params)

    print(f"\n--- Running simulation for {gpu_params.architecture} GPU ---")

    # --- Workload Scenarios ---

    # 1. VectorDB Search Scenario
    env.process(system.run_workload(
        "VectorDB_Search",
        index_size_gb=10, # 10 GB index
        query_batch_size=16,
        num_warps_per_query=100, # Simplified: 100 warps per query
        compute_cycles_per_warp=1000000, # 1M cycles for coarse search
        memory_access_gb_per_warp=0.001, # Small memory access
        candidate_data_gb=5 # 5 GB candidate data
    ))

    # 2. LLM KV Cache Offloading Scenario
    env.process(system.run_workload(
        "LLM_KV_Cache_Offloading",
        cache_page_size_gb=0.004, # 4MB page size
        num_pages_to_swap=100, # Swap 100 pages (400MB)
        compute_cycles_per_token=500000 # 500k cycles for token generation
    ))

    # 3. GNN Training Scenario
    env.process(system.run_workload(
        "GNN_Training",
        graph_data_gb=1, # 1 GB graph structure data
        feature_data_gb=0.5, # 0.5 GB features per mini-batch
        num_mini_batches=5,
        compute_cycles_per_batch=20000, # 20k cycles per batch
        memory_access_gb_per_batch=0.01 # Small memory access per batch
    ))

    env.run(until=simulation_time)
    print(f"--- Simulation for {gpu_params.architecture} GPU finished at {env.now:.2f} ---")

if __name__ == '__main__':
    # Run simulation for H100
    run_simulation(H100_PARAMS, PCIe5_NVME_SSD_PARAMS, simulation_time=1000)

    # Run simulation for B200
    run_simulation(B200_PARAMS, PCIe5_NVME_SSD_PARAMS, simulation_time=1000)


