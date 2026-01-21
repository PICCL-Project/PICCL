from __future__ import annotations

import argparse
from pathlib import Path
from itertools import permutations

PROJECT_HOME = Path(__file__).resolve().parent.parent.parent

# Stores global state info
class SimState():
    def __init__(self) -> None:
        self.time: float = 0.0
        self.next_event_id: int = 0
        self.scheduler: Scheduler = None
        self.n_nodes: int = 0
        self.profiles: dict[tuple[int, int], dict[int: dict[str, float]]] = {} # {(src, dst): {size: {param: value}}}
        self.stats: dict = {}

        # Device-specific constants
        self.max_device_radix = 4
        self.pipelinable: bool = False
        self.group_mapping: dict[int, int] = {}                 # {node: group}
        self.group_queues: dict[int, list[list[Event]]] = {}          # {group_id: [tasks]}

        # These are modified by the scheduler
        self.tasks: dict[int: Task] = {}                        # {task_id: Task}
        self.eventq: EventQueue = EventQueue()
        self.completed_tasks: set[int] = set()
        self.started_tasks: set[int] = set()
        self.device_ports: dict[int, list[int]] = {}            # {node: [ports]}
        self.wait_queue: list[Task] = []
        self.last_send: dict[int, list[float]] = {}             # {node: [last_send_time]}
        self.last_recv: dict[int, list[float]] = {}             # {node: [last_send_time]}
        


class Simulator():
    def __init__(self, scheduler: type[Scheduler], args) -> None:
        self.args = args

        # print("Initializing simulator...")
        self.state = SimState()
        self.state.scheduler = scheduler(self.state)

        # Set state with config
        self.state.max_device_radix = args.radix
        # print(f"Device radix: {self.state.max_device_radix}")

        # Load the schedule and profile
        self.state.n_nodes, self.state.tasks = parse_schedule_to_tasks(args.schedule)
        # print(f"Number of nodes: {self.state.n_nodes}")
        # print(f"Number of tasks: {len(self.state.tasks)}")

        # Project home
        topology, param_files, self.state.group_mapping = parse_topology(get_path(args.topo))
        params = parse_loggp_param(param_files)
        interp_params = interpolate_loggp_params(params)

        # Initialize the scheduler
        self.__initialize_state()
        self.__initialize_eventq()
        self.__initialize_profile(topology, interp_params)
        self.__initialize_groupq()
        
        # print("Scheduler initialized.")
        # print(f"Number of nodes: {self.state.n_nodes}")
        # print(f"Number of tasks: {len(self.state.tasks)}")

    def __initialize_state(self):
        # print("Initializing state...")
        # Initialize device radix and last send time
        for node in range(self.state.n_nodes):
            self.state.device_ports[node] = {"send":list(range(self.state.max_device_radix)), 
                                             "recv":list(range(self.state.max_device_radix))}
            self.state.last_send[node] = [-float("inf")] * self.state.max_device_radix
            self.state.last_recv[node] = [-float("inf")] * self.state.max_device_radix

    def __initialize_eventq(self):
        # print("Initializing event queue...")
        start_tasks = [task for task in self.get_tasks() if not task.deps]
        for task in start_tasks:
            event = Event(self.state, task, timestamp=0.0, type="start")
            self.state.eventq.post_event(event)

    def __initialize_profile(self, topology, params):
        # print("Initializing profiles...")
        for src, dst in permutations(range(self.state.n_nodes), 2):
            groups = topology[(src, dst)]

            if src == 0 and dst == 1:
                pass
            smallest_level_file = min(groups, key=lambda x: x[1])[2]
            self.state.profiles[(src, dst)] = params[smallest_level_file] 

    def __initialize_groupq(self):
        # print("Initializing group queues...")
        for group in self.state.group_mapping.values():
            group_id = group["group_id"]
            ports = group["ports"]

            if group_id not in self.state.group_queues.items():
                self.state.group_queues[group_id] = {"send":[], "recv":[], "availSend": ports, "availRecv": ports}


    def get_tasks(self) -> list[Task]:
        return list(self.state.tasks.values())

    def print_queue(self):
        self.state.eventq.print_events()

    def run(self):
        # try:
        while True:
            event = self.state.eventq.get_event()
            if event is None:
                break
            self.state.time = event.timestamp
            event.dispatch()
        # except Exception as e:
        #     pass
            # print(e)
            # print("Event queue is empty. Simulation finished.")

    def output_stats(self):
        output_file = self.args.output

        # output states to file
        with open(output_file, "w") as f:
            f.write("LogGP Simulation Output\n")
            f.write("\n====================================\n\n")
            f.write("Simulation args:\n")
            f.write("  Schedule: {}\n".format(self.args.schedule))
            f.write("  Topology: {}\n".format(self.args.topo))
            f.write("  Device radix: {}\n".format(self.args.radix))
            f.write("  Output file: {}\n".format(self.args.output))
            f.write("\n====================================\n\n")
            f.write("Simulation Overview:\n")
            f.write("  Simulation finished at time {} us\n".format(self.state.time * 1e6))
            f.write("  Number of tasks: {}\n".format(len(self.state.tasks)))
            f.write("  Number of completed tasks: {}\n".format(len(self.state.completed_tasks)))
            f.write("\n====================================\n\n")
            f.write("Scheduler Statistics:\n")
            f.write("  Number of events: {}\n".format(self.state.eventq.counter))
            f.write("  Send Contention: \n")
            for group_id, stat in self.state.stats["send_contention"].items():
                f.write("    Group {}: {} / {} = {} attempts successful\n".format(group_id, stat["success"], stat["attempts"],  stat["success"] / stat["attempts"]))
                f.write("    Max Queue Size: {}\n".format(stat["max_queue_size"]))
            f.write("  Recv Contention: \n")
            for group_id, stat in self.state.stats["recv_contention"].items():
                f.write("    Group {}: {} / {} = {} attempts successful\n".format(group_id, stat["success"], stat["attempts"],  stat["success"] / stat["attempts"]))
                f.write("    Max Queue Size: {}\n".format(stat["max_queue_size"]))

        return


    def summarize(self):
        # print()
        # print("======================")
        # print(f"Simulation finished at time {self.state.time * 1e6} us")
        if len(self.state.completed_tasks) != len(self.state.tasks):
            print("0")
            print(self.state.completed_tasks)
            raise ValueError("Not all tasks completed {}/{}".format(len(self.state.completed_tasks), len(self.state.tasks)))
        else: 
            print(f"{self.state.stats['finished_tasks'][-1][0] * 1e6}")
        
    

from event_queue import EventQueue
from schedulers.scheduler import Scheduler
from schedulers.loggp_scheduler import LogGPScheduler
from schedulers.loggpc_scheduler import LogGPCScheduler
from event import Event
from parser import parse_schedule_to_tasks, parse_loggp_param, parse_topology
from task import Task
from utils import get_path, interpolate_loggp_params



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the LogGP simulation.")
    parser.add_argument("--schedule", 
                        type=str, 
                        default="./schedules/8_ring_allgather.xml", 
                        help="Path to the schedule XML file.")
    parser.add_argument("--topo",
                        type=str, 
                        default="./topo/2_4_1_fc.xml", 
                        help="Path to the topology XML file.")
    parser.add_argument("--radix", 
                        type=int, 
                        default=4, 
                        help="Maximum number of channels per device (default: 1).")
    parser.add_argument("--output",
                        type=str, 
                        default="./output/sim_output.txt", 
                        help="Path to the output file.")

    # Parse arguments
    args = parser.parse_args()

    # Initialize the simulator with the mock scheduler
    simulator = Simulator(LogGPCScheduler, args)
    # simulator = Simulator(LogGPScheduler, args)

    # Run the simulation
    simulator.run()

    # Output the simulation statistics
    # simulator.output_stats()

    # Finish the simulation
    simulator.summarize()

