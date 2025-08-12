

import simpy
from parameters import GpuParameters, StorageParameters

class StorageSystem:
    def __init__(self, env: simpy.Environment, params: StorageParameters):
        self.env = env
        self.params = params
        self.read_bandwidth_resource = simpy.Resource(env, capacity=1) # Simplified for now
        self.write_bandwidth_resource = simpy.Resource(env, capacity=1) # Simplified for now
        print(f"[{env.now:.2f}] StorageSystem initialized with {params.sequential_read_bandwidth_gb_s} GB/s read BW")

    def read_data(self, size_gb: float):
        # Simulate latency
        yield self.env.timeout(self.params.latency_us / 1_000_000) # Convert us to seconds

        # Simulate bandwidth limited transfer
        transfer_time = size_gb / self.params.sequential_read_bandwidth_gb_s
        with self.read_bandwidth_resource.request() as req:
            yield req
            yield self.env.timeout(transfer_time)
        print(f"[{self.env.now:.2f}] StorageSystem: Read {size_gb} GB completed.")

    def write_data(self, size_gb: float):
        yield self.env.timeout(self.params.latency_us / 1_000_000)
        transfer_time = size_gb / self.params.sequential_write_bandwidth_gb_s
        with self.write_bandwidth_resource.request() as req:
            yield req
            yield self.env.timeout(transfer_time)
        print(f"[{self.env.now:.2f}] StorageSystem: Write {size_gb} GB completed.")

class HostMemory:
    def __init__(self, env: simpy.Environment):
        self.env = env
        # Host memory is assumed to be large enough and fast enough not to be a bottleneck for now
        print(f"[{env.now:.2f}] HostMemory initialized.")

    def transfer_data(self, size_gb: float):
        # Placeholder for host memory access time if needed
        yield self.env.timeout(0.001) # Very small delay
        print(f"[{self.env.now:.2f}] HostMemory: Data transfer of {size_gb} GB completed.")

class PCIeInterface:
    def __init__(self, env: simpy.Environment, pcie_gen: int):
        self.env = env
        # PCIe Gen 5 bandwidth (approx 32 GB/s for x16 slot)
        self.bandwidth_gb_s = 32.0 if pcie_gen == 5 else 16.0 # Simplified
        self.resource = simpy.Resource(env, capacity=1) # Simplified for now
        print(f"[{env.now:.2f}] PCIeInterface (Gen {pcie_gen}) initialized with {self.bandwidth_gb_s} GB/s BW")

    def transfer_data(self, size_gb: float):
        transfer_time = size_gb / self.bandwidth_gb_s
        with self.resource.request() as req:
            yield req
            yield self.env.timeout(transfer_time)
        print(f"[{self.env.now:.2f}] PCIeInterface: Transfer of {size_gb} GB completed.")

class MemorySystem:
    def __init__(self, env: simpy.Environment, gpu_params: GpuParameters):
        self.env = env
        self.params = gpu_params
        self.bandwidth_resource = simpy.Resource(env, capacity=1) # Simplified for now
        print(f"[{env.now:.2f}] MemorySystem (HBM) initialized with {self.params.memory_bandwidth_tb_s} TB/s BW")

    def access_data(self, size_gb: float):
        # Simulate HBM bandwidth limited access
        transfer_time = size_gb / (self.params.memory_bandwidth_tb_s * 1024) # Convert TB/s to GB/s
        with self.bandwidth_resource.request() as req:
            yield req
            yield self.env.timeout(transfer_time)
        # print(f"[{self.env.now:.2f}] MemorySystem: Accessed {size_gb} GB.")

class Warp:
    def __init__(self, env: simpy.Environment, warp_id: int, sm_id: int, gpu_memory_system: MemorySystem):
        self.env = env
        self.warp_id = warp_id
        self.sm_id = sm_id
        self.gpu_memory_system = gpu_memory_system
        self.action = None # To hold the process

    def execute_compute(self, cycles: int):
        # Simulate compute cycles
        yield self.env.timeout(cycles * 1e-6) # Assuming 1 cycle = 1us for better visibility
        # print(f"[{self.env.now:.2f}] Warp {self.warp_id} on SM {self.sm_id}: Computed for {cycles} cycles.")

    def request_memory(self, size_gb: float):
        # Simulate memory access from HBM
        yield self.env.process(self.gpu_memory_system.access_data(size_gb))
        # print(f"[{self.env.now:.2f}] Warp {self.warp_id} on SM {self.sm_id}: Requested {size_gb} GB from HBM.")

class WarpScheduler:
    def __init__(self, env: simpy.Environment, sm_id: int, scheduler_id: int, gpu_memory_system: MemorySystem):
        self.env = env
        self.sm_id = sm_id
        self.scheduler_id = scheduler_id
        self.gpu_memory_system = gpu_memory_system
        self.ready_warps = simpy.Store(env) # Warps ready to be scheduled
        self.execution_unit = simpy.Resource(env, capacity=1) # Simplified: one execution unit per scheduler
        self.action = env.process(self.run())
        print(f"[{env.now:.2f}] WarpScheduler {scheduler_id} on SM {sm_id} initialized.")

    def run(self):
        while True:
            warp = yield self.ready_warps.get() # Wait for a warp to be ready
            # print(f"[{self.env.now:.2f}] WarpScheduler {self.scheduler_id} on SM {self.sm_id}: Scheduling Warp {warp.warp_id}")
            with self.execution_unit.request() as req:
                yield req # Acquire execution unit
                # Here, the warp would execute its next instruction/block of instructions
                # For simplicity, we'll just yield control back to the warp's process
                # In a real model, this would involve more detailed instruction execution
                yield self.env.timeout(0.000001) # Small scheduling overhead
                # print(f"[{self.env.now:.2f}] WarpScheduler {self.scheduler_id} on SM {self.sm_id}: Finished scheduling Warp {warp.warp_id}")

    def schedule_warp(self, warp: Warp):
        self.ready_warps.put(warp)

class StreamingMultiprocessor:
    def __init__(self, env: simpy.Environment, sm_id: int, gpu_params: GpuParameters, gpu_memory_system: MemorySystem):
        self.env = env
        self.sm_id = sm_id
        self.gpu_params = gpu_params
        self.gpu_memory_system = gpu_memory_system
        self.warp_schedulers = []
        # Assuming 4 warp schedulers per SM for simplicity, typical for Hopper/Blackwell
        for i in range(4):
            self.warp_schedulers.append(WarpScheduler(env, sm_id, i, gpu_memory_system))
        
        self.active_warps = simpy.Resource(env, capacity=gpu_params.max_warps_per_sm)
        print(f"[{env.now:.2f}] SM {sm_id} initialized with {gpu_params.max_warps_per_sm} max active warps.")

    def process_warp(self, warp: Warp, compute_cycles: int, memory_access_gb: float):
        with self.active_warps.request() as req:
            yield req # Acquire slot for active warp
            # Assign warp to a scheduler (simplified: just pick first one)
            # In a real model, this would involve load balancing
            self.warp_schedulers[0].schedule_warp(warp)
            
            # Simulate warp execution
            yield self.env.process(warp.execute_compute(compute_cycles))
            if memory_access_gb > 0:
                yield self.env.process(warp.request_memory(memory_access_gb))
            
            # print(f"[{self.env.now:.2f}] SM {self.sm_id}: Warp {warp.warp_id} finished processing.")

class GPU:
    def __init__(self, env: simpy.Environment, params: GpuParameters):
        self.env = env
        self.params = params
        self.memory_system = MemorySystem(env, params)
        self.sms = []
        for i in range(params.num_sms):
            self.sms.append(StreamingMultiprocessor(env, i, params, self.memory_system))
        print(f"[{env.now:.2f}] GPU ({params.architecture}) initialized with {params.num_sms} SMs.")

    def launch_kernel(self, num_warps: int, compute_cycles_per_warp: int, memory_access_gb_per_warp: float):
        print(f"[{self.env.now:.2f}] GPU: Launching kernel with {num_warps} warps.")
        warp_processes = []
        for i in range(num_warps):
            sm_id = i % self.params.num_sms # Simple round-robin assignment to SMs
            warp = Warp(self.env, i, sm_id, self.memory_system)
            warp_processes.append(self.env.process(
                self.sms[sm_id].process_warp(warp, compute_cycles_per_warp, memory_access_gb_per_warp)
            ))
        yield simpy.AllOf(self.env, warp_processes) # Wait for all warps to complete
        print(f"[{self.env.now:.2f}] GPU: Kernel execution completed.")

class System:
    def __init__(self, env: simpy.Environment, gpu_params: GpuParameters, storage_params: StorageParameters):
        self.env = env
        self.storage = StorageSystem(env, storage_params)
        self.host_memory = HostMemory(env)
        self.pcie = PCIeInterface(env, gpu_params.pcie_gen)
        self.gpu = GPU(env, gpu_params)
        print(f"[{self.env.now:.2f}] System initialized.")

    def run_workload(self, workload_scenario: str, **kwargs):
        print(f"\n[{self.env.now:.2f}] Running Workload: {workload_scenario}")
        if workload_scenario == "VectorDB_Search":
            yield self.env.process(self._run_vector_db_search(**kwargs))
        elif workload_scenario == "LLM_KV_Cache_Offloading":
            yield self.env.process(self._run_llm_kv_cache_offloading(**kwargs))
        elif workload_scenario == "GNN_Training":
            yield self.env.process(self._run_gnn_training(**kwargs))
        else:
            print(f"Unknown workload scenario: {workload_scenario}")

    def _run_vector_db_search(self, index_size_gb: float, query_batch_size: int, num_warps_per_query: int, compute_cycles_per_warp: int, memory_access_gb_per_warp: float, candidate_data_gb: float):
        # Index loading
        start_time = self.env.now
        print(f"[{start_time:.2f}] VectorDB Search: Index loading (Storage -> Host -> PCIe -> GPU HBM)...")
        yield self.env.process(self.storage.read_data(index_size_gb))
        yield self.env.process(self.host_memory.transfer_data(index_size_gb))
        yield self.env.process(self.pcie.transfer_data(index_size_gb + (query_batch_size * 0.000001))) # Add small query data
        end_time = self.env.now
        print(f"[{end_time:.2f}] VectorDB Search: Index loading completed. Duration: {end_time - start_time:.2f}s")
        
        # GPU computation (Coarse-grained search)
        start_time = self.env.now
        print(f"[{start_time:.2f}] VectorDB Search: GPU computation (Coarse-grained search)...")
        yield self.env.process(self.gpu.launch_kernel(query_batch_size * num_warps_per_query, compute_cycles_per_warp, memory_access_gb_per_warp))
        end_time = self.env.now
        print(f"[{end_time:.2f}] VectorDB Search: GPU computation (Coarse-grained search) completed. Duration: {end_time - start_time:.2f}s")

        # Candidate data loading
        start_time = self.env.now
        print(f"[{start_time:.2f}] VectorDB Search: Candidate data loading (Storage -> Host -> PCIe -> GPU HBM)...")
        yield self.env.process(self.storage.read_data(candidate_data_gb))
        yield self.env.process(self.host_memory.transfer_data(candidate_data_gb))
        yield self.env.process(self.pcie.transfer_data(candidate_data_gb))
        end_time = self.env.now
        print(f"[{end_time:.2f}] VectorDB Search: Candidate data loading completed. Duration: {end_time - start_time:.2f}s")

        # GPU computation (Fine-grained search)
        start_time = self.env.now
        print(f"[{start_time:.2f}] VectorDB Search: GPU computation (Fine-grained search)...")
        yield self.env.process(self.gpu.launch_kernel(query_batch_size * num_warps_per_query, compute_cycles_per_warp * 2, memory_access_gb_per_warp * 0.5)) # More compute, less memory
        end_time = self.env.now
        print(f"[{end_time:.2f}] VectorDB Search: GPU computation (Fine-grained search) completed. Duration: {end_time - start_time:.2f}s")
        
        print(f"[{self.env.now:.2f}] VectorDB Search: All steps completed.")

    def _run_llm_kv_cache_offloading(self, cache_page_size_gb: float, num_pages_to_swap: int, compute_cycles_per_token: int):
        total_swap_gb = cache_page_size_gb * num_pages_to_swap
        
        # Swapping out
        start_time = self.env.now
        print(f"[{start_time:.2f}] LLM KV Cache Offloading: Swapping out {total_swap_gb} GB (GPU -> Host -> Storage)...")
        yield self.env.process(self.pcie.transfer_data(total_swap_gb)) # GPU to Host
        yield self.env.process(self.host_memory.transfer_data(total_swap_gb))
        yield self.env.process(self.storage.write_data(total_swap_gb))
        end_time = self.env.now
        print(f"[{end_time:.2f}] LLM KV Cache Offloading: Swapping out completed. Duration: {end_time - start_time:.2f}s")

        # Swapping in
        start_time = self.env.now
        print(f"[{start_time:.2f}] LLM KV Cache Offloading: Swapping in {total_swap_gb} GB (Storage -> Host -> GPU)...")
        yield self.env.process(self.storage.read_data(total_swap_gb))
        yield self.env.process(self.host_memory.transfer_data(total_swap_gb))
        yield self.env.process(self.pcie.transfer_data(total_swap_gb)) # Host to GPU
        end_time = self.env.now
        print(f"[{end_time:.2f}] LLM KV Cache Offloading: Swapping in completed. Duration: {end_time - start_time:.2f}s")

        # LLM computation
        start_time = self.env.now
        print(f"[{start_time:.2f}] LLM KV Cache Offloading: LLM computation...")
        yield self.env.process(self.gpu.launch_kernel(100, compute_cycles_per_token, 0.001)) # 100 warps, small memory access
        end_time = self.env.now
        print(f"[{end_time:.2f}] LLM KV Cache Offloading: LLM computation completed. Duration: {end_time - start_time:.2f}s")
        print(f"[{self.env.now:.2f}] LLM KV Cache Offloading: All steps completed.")

    def _run_gnn_training(self, graph_data_gb: float, feature_data_gb: float, num_mini_batches: int, compute_cycles_per_batch: int, memory_access_gb_per_batch: float):
        print(f"[{self.env.now:.2f}] GNN Training: Initial graph data loading...")
        # Simulate loading graph structure (random access)
        yield self.env.process(self.storage.read_data(graph_data_gb * 0.1)) # 10% for graph structure
        yield self.env.process(self.host_memory.transfer_data(graph_data_gb * 0.1))
        yield self.env.process(self.pcie.transfer_data(graph_data_gb * 0.1))

        for i in range(num_mini_batches):
            print(f"[{self.env.now:.2f}] GNN Training: Mini-batch {i+1}/{num_mini_batches} - Feature loading...")
            # Simulate loading features for current mini-batch (sequential access)
            yield self.env.process(self.storage.read_data(feature_data_gb))
            yield self.env.process(self.host_memory.transfer_data(feature_data_gb))
            yield self.env.process(self.pcie.transfer_data(feature_data_gb))

            print(f"[{self.env.now:.2f}] GNN Training: Mini-batch {i+1}/{num_mini_batches} - GPU computation...")
            yield self.env.process(self.gpu.launch_kernel(100, compute_cycles_per_batch, memory_access_gb_per_batch))
        print(f"[{self.env.now:.2f}] GNN Training: Completed.")

