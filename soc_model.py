
import simpy
import random
from dataclasses import dataclass

# 1. Packet Definition
@dataclass
class Packet:
    """SoC 통신에 사용되는 패킷을 정의하는 데이터 클래스입니다."""
    addr: int
    cmd: str
    databyte: int
    creation_time: float

# 2. Traffic Generator (TG)
class TrafficGenerator:
    """
    주기적으로 패킷을 생성하여 메모리 컨트롤러로 전송하는 트래픽 생성기(TG)입니다.
    """
    def __init__(self, env: simpy.Environment, name: str, memory_pipe: simpy.Store):
        self.env = env
        self.name = name
        self.memory_pipe = memory_pipe
        self.packets_sent = 0
        # SimPy 프로세스를 시작합니다.
        self.action = env.process(self.run())

    def run(self):
        """패킷을 생성하고 전송하는 메인 로직입니다."""
        print(f'{self.env.now:.2f}: [{self.name}] 트래픽 생성 시작')
        while True:
            # 1~5 사이의 랜덤한 시간 간격으로 패킷 생성
            yield self.env.timeout(random.randint(1, 5))

            self.packets_sent += 1
            packet = Packet(
                addr=random.randint(0, 2**16),
                cmd=random.choice(['READ', 'WRITE']),
                databyte=random.randint(1, 64),
                creation_time=self.env.now
            )
            print(f'{self.env.now:.2f}: [{self.name}] 패킷 생성 및 전송 -> {packet}')
            self.memory_pipe.put(packet)

# 3. Memory
class Memory:
    """
    패킷을 수신하고, 명령어(READ/WRITE)에 따라 다른 처리 시간을 시뮬레이션하는 메모리입니다.
    """
    def __init__(self, env: simpy.Environment, name: str):
        self.env = env
        self.name = name
        # TG로부터 패킷을 받을 파이프(버퍼)
        self.pipe = simpy.Store(env)
        # SimPy 프로세스를 시작합니다.
        self.action = env.process(self.run())

    def run(self):
        """패킷을 수신하고 처리하는 메인 로직입니다."""
        print(f'{self.env.now:.2f}: [{self.name}] 메모리 동작 시작')
        while True:
            # 파이프에 패킷이 들어올 때까지 기다립니다.
            packet: Packet = yield self.pipe.get()
            
            latency = self.env.now - packet.creation_time
            print(f'{self.env.now:.2f}: [{self.name}] 패킷 수신 <- {packet} (Latency: {latency:.2f})')

            # 명령어에 따라 처리 시간 시뮬레이션
            if packet.cmd == 'READ':
                yield self.env.timeout(2) # READ는 2 사이클 소요
            elif packet.cmd == 'WRITE':
                yield self.env.timeout(4) # WRITE는 4 사이클 소요
            
            print(f'{self.env.now:.2f}: [{self.name}] 패킷 처리 완료')

# 4. Test (Simulation)
if __name__ == '__main__':
    print('--- SoC 성능 모델 시뮬레이션 시작 ---')
    
    # 시뮬레이션 환경 설정
    env = simpy.Environment()
    
    # 컴포넌트 생성
    memory = Memory(env, 'DRAM_CTRL')
    # TG가 메모리의 파이프를 알 수 있도록 전달
    tg1 = TrafficGenerator(env, 'CPU_CORE_1', memory.pipe)
    
    # 시뮬레이션 실행 (30 time unit 동안)
    simulation_time = 30
    env.run(until=simulation_time)
    
    print(f'--- {simulation_time} time unit 후 시뮬레이션 종료 ---')

