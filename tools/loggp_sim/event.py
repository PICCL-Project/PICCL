from __future__ import annotations

# Simulator event
class Event():
    def __init__(self, state: SimState, task: Task, timestamp: float, type: str, parent: int = -1) -> None:
        self.state = state
        self.task = task

        self.id = state.next_event_id
        self.timestamp = timestamp
        self.type = type
        self.parent = parent

        state.next_event_id += 1

    def __lt__(self, other):
        return self.timestamp < other.timestamp
    
    def __str__(self) -> str:
        return f"Event(id={self.id}, type={self.type}, timestamp={self.timestamp}, task_id={self.task.id})"


    def dispatch(self):
        # print(f"Event {self.id}:")
        if self.type in self.state.scheduler.events.keys():
            # print(f"Dispatching event {self.id} of type {self.type} at time {self.timestamp}.")
            self.state.scheduler.events[self.type](self, self.task)
        else:
            raise ValueError(f"Unknown event type: {self.type}")
    
    def update_time(self, new_timestamp: float):
        self.timestamp = new_timestamp

    def complete(self):
        pass

from sim import SimState
from task import Task