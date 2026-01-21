from __future__ import annotations

from sim import SimState
from task import Task
from event import Event

# Scheduler for LogGP model
class Scheduler():
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
        # Unimplemented error
        raise NotImplementedError("task_start is not implemented.")


    def task_done(self, event: Event, task: Task)-> None:
        # Unimplemented error
        raise NotImplementedError("task_done is not implemented.")


    def task_execute_or(self, event: Event, task: Task) -> None:
        # Unimplemented error
        raise NotImplementedError("task_execute_or is not implemented.")


    def task_execute_os(self, event: Event, task: Task) -> None:
        # Unimplemented error
        raise NotImplementedError("task_execute_os is not implemented.")

    

    def task_execute_L(self, event: Event, task: Task) -> None:
        # Unimplemented error
        raise NotImplementedError("task_execute_L is not implemented.")

    
    def task_execute_g(self, event: Event, task: Task) -> None:
        # Unimplemented error
        raise NotImplementedError("task_execute_g is not implemented.")
    

    def task_execute_G(self, event: Event, task: Task) -> None:
        # Unimplemented error
        raise NotImplementedError("task_execute_G is not implemented.")
    
    # The send event should add the event to the group's NIC queue
    def task_execute_Send(self, event: Event, task: Task) -> None:
        # Unimplemented error
        raise NotImplementedError("task_execute_Send is not implemented.")
        

    def task_execute_Recv(self, event: Event, task: Task) -> None:
        # Unimplemented error
        raise NotImplementedError("task_execute_Recv is not implemented.")
