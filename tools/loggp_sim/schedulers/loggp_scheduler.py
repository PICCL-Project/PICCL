from __future__ import annotations

from sim import SimState
from task import Task
from event import Event
from schedulers.scheduler import Scheduler

# Scheduler for LogGP model
class LogGPScheduler(Scheduler):
    def __init__(self, state: SimState, config: dict = {}) -> None:
        self.state = state
        self.config = config
        self.stats = {}
        self.events = {
            "start": self.task_start,
            "done": self.task_done,
            "execute_or": self.task_execute_or,
            "execute_os": self.task_execute_os,
            "execute_L": self.task_execute_L,
            "execute_g": self.task_execute_g,
            "execute_G": self.task_execute_G,
        }

        return


    def task_start(self, event: Event, task: Task) -> None:
        print(f"Task {task.id} start at time {self.state.time}.")

        # Check if task can be started, if it can be started, assign ports
        if self.state.device_ports[task.s_node] and self.state.device_ports[task.d_node]:
            task.s_port = self.state.device_ports[task.s_node].pop(0)
            task.d_port = self.state.device_ports[task.d_node].pop(0)

            # Post event g at current time
            self.state.eventq.post_event(
                Event(self.state, task, timestamp=self.state.time, type="execute_g")
            )

        # Otherwise, add back into wait queue
        else:
            self.state.wait_queue.append(task)

        return


    def task_done(self, event: Event, task: Task)-> None:
        print(f"Task {task.id} done at time {self.state.time}.")

        self.state.completed_tasks.add(task.id)

        self.state.device_ports[task.s_node].append(task.s_port)
        self.state.device_ports[task.d_node].append(task.d_port)

        # Get list of tasks to start (those in the wait queue and those that depend on this task)
        start_tasks = []

        start_tasks += self.state.wait_queue
        self.state.wait_queue = []

        for task_id in task.children:
            child_task: Task = self.state.tasks[task_id]
            if not child_task.deps - self.state.completed_tasks:
                start_tasks.append(child_task) 
        
        for task in start_tasks:
            self.state.eventq.post_event(
                Event(self.state, task, timestamp=self.state.time, type="start")
            )
        
        return


    def task_execute_or(self, event: Event, task: Task) -> None:
        print(f"Task {task.id} execute or at time {self.state.time}.")

        # Compute elapsed time
        elapse_time = self.state.profiles[(task.s_node, task.d_node)]["o_r"](task.size)

        # Assign task done at the end of the event
        self.state.eventq.post_event(
            Event(self.state, task, timestamp=self.state.time + elapse_time, type="done")
        )

        return


    def task_execute_os(self, event: Event, task: Task) -> None:
        print(f"Task {task.id} execute os at time {self.state.time}.")

        # Compute elapsed time
        elapse_time = self.state.profiles[(task.s_node, task.d_node)]["o_s"](task.size)

        # Assign task L after the event
        self.state.eventq.post_event(
            Event(self.state, task, timestamp=self.state.time + elapse_time, type="execute_L")
        )

        return

    

    def task_execute_L(self, event: Event, task: Task) -> None:
        print(f"Task {task.id} execute L at time {self.state.time}.")

        task.send_left -= 1
        self.state.last_send[task.s_node][task.s_port] = self.state.time

        if (task.send_left == 0):
            # Post event or at elapsed time
            elapse_time = self.state.profiles[(task.s_node, task.d_node)]["L"](task.size)
            
            self.state.eventq.post_event(
                Event(self.state, task, timestamp=self.state.time + elapse_time, type="execute_or")
            )
        else:
            # Post event G at current time
            self.state.eventq.post_event(
                Event(self.state, task, timestamp=self.state.time, type="execute_G")
            )

        return

    
    def task_execute_g(self, event: Event, task: Task) -> None:
        print(f"Task {task.id} execute g at time {self.state.time}.")

        # Compute elapsed time
        elapse_time = max(0.0,
                          self.state.profiles[(task.s_node, task.d_node)]["g"](task.size) \
                            - (self.state.time - self.state.last_send[task.s_node][task.s_port]) \
                                - self.state.profiles[(task.s_node, task.d_node)]["o_s"](task.size))
        
        # Assign task o_s at the end of the event
        self.state.eventq.post_event(
            Event(self.state, task, timestamp=self.state.time + elapse_time, type="execute_os")
        )

        return
    

    def task_execute_G(self, event: Event, task: Task) -> None:
        print(f"Task {task.id} execute G at time {self.state.time}.")

        # Compute elapsed time
        elapse_time = self.state.profiles[(task.s_node, task.d_node)]["G"](task.size)

        # Assign task L at the end of the event
        self.state.eventq.post_event(
            Event(self.state, task, timestamp=self.state.time + elapse_time, type="execute_L")
        )

        return
