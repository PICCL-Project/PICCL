from __future__ import annotations

import heapq
import itertools

class EventQueue:
    def __init__(self):
        self.pq = []  # Priority queue
        self.entry_finder = {}  # Map of events to their entries
        self.counter = itertools.count()  # Unique sequence count
        self.REMOVED = '<removed-event>'  # Placeholder for a removed event

    def post_event(self, event: Event):
        count = next(self.counter)
        entry = [event.timestamp, count, event]
        self.entry_finder[event.id] = entry
        heapq.heappush(self.pq, entry)

    def update_event(self, event: Event, new_timestamp: float):
        if event.id in self.entry_finder:
            self.delete_event(event)
            event.update_time(new_timestamp)
            self.post_event(event)

    def delete_event(self, event: Event):
        entry = self.entry_finder.pop(event.id, None)
        if entry is not None:
            entry[-1] = self.REMOVED

    def get_event(self) -> Event:
        while self.pq:
            timestamp, count, event = heapq.heappop(self.pq)
            if event != self.REMOVED:
                del self.entry_finder[event.id]
                return event
        return None

    def print_events(self):
        events = [entry for entry in self.pq if entry[-1] != self.REMOVED]
        events.sort()
        print("")
        print("### CURRENT EVENTS IN QUEUE:")
        for timestamp, count, event in events:
            print(event)
        print("")


from event import Event
from task import Task
