from __future__ import annotations

from sim import SimState
from task import Task
from event import Event
from schedulers.scheduler import Scheduler

QUANTUM = 1e-12

# Scheduler for LogGP model
class LogGPCScheduler(Scheduler):
    def __init__(self, state: SimState, config: dict = {}) -> None:
        self.state = state
        self.config = config

        self.state.stats["finished_tasks"] = []
        self.state.stats["tasks_start"] = {}
        self.state.stats["tasks_done"] = {}
        self.state.stats["send_contention"] = {}
        self.state.stats["recv_contention"] = {}

        self.state.mtu = 4096  # Maximum transmission unit
        self.state.send_wait_queue = []
        self.state.recv_buffer = {}
        self.state.augmented_task_latency = {}

        self.events = {
            "start": self.task_start,
            "done": self.task_done,
            "execute_or": self.task_execute_or,
            "execute_os": self.task_execute_os,
            "execute_Send": self.task_execute_Send,
            "execute_Recv": self.task_execute_Recv,
            "free_SendNIC": self.task_free_SendNIC,
            "free_RecvNIC": self.task_free_RecvNIC,
            "free_SendPort": self.task_free_SendPort,
            "free_RecvPort": self.task_free_RecvPort,
            "execute_StartSend": self.task_start_Send,
            "execute_send_g": self.task_execute_send_g,
            "execute_recv_g": self.task_execute_recv_g,
            "execute_OccupySend": self.task_occupy_Send,
            "execute_OccupyRecv": self.task_occupy_Recv,
            "execute_StartRecv": self.task_start_Recv,
        }

        return


    def __try_start_children_tasks(self, task: Task) -> None:
        start_tasks = []

        for task_id in task.children:
            child_task: Task = self.state.tasks[task_id]
            if child_task.id not in self.state.started_tasks:
                if not (child_task.deps - self.state.completed_tasks) :
                    start_tasks.append(child_task) 
            
        for s_task in start_tasks:
            self.state.eventq.post_event(
                Event(self.state, s_task, timestamp=self.state.time, type="start")
            )
        return
    

    def task_start(self, event: Event, task: Task) -> None:
        # print(f"Task {task.id} start at time {self.state.time}.")

        self.state.eventq.post_event(
            Event(self.state, task, timestamp=self.state.time, type="execute_os")
        )

        return
    
    
    def task_execute_os(self, event: Event, task: Task) -> None:
        # print(f"Task {task.id} execute os at time {self.state.time}.")

        # Compute elapsed time
        elapse_time = self.state.profiles[(task.s_node, task.d_node)]["o_s"](task.size)

        # Assign task L after the event
        self.state.eventq.post_event(
            Event(self.state, task, timestamp=self.state.time + elapse_time, type="execute_StartSend")
        )

        return
    

    # ================================================== SEND LOGIC

    def task_start_Send(self, event: Event, task: Task) -> None:
        # Add task to the send wait queue
        self.state.send_wait_queue.append(task)
        self.task_occupy_Send(event, task)
        return
    
    def task_occupy_Send(self, event: Event, task: Task) -> None:
        # Find available send port
        if self.state.device_ports[task.s_node]["send"]:
            task.s_port = self.state.device_ports[task.s_node]["send"].pop(0)
            self.state.send_wait_queue.remove(task)
            self.state.eventq.post_event(
                Event(self.state, task, timestamp=self.state.time, type="execute_send_g")
            )


    def __try_start_send(self, time) -> None:
        # Check if there is a task in the send wait queue
        if not self.state.send_wait_queue:
            return
        
        tasks = self.state.send_wait_queue.copy()
        self.state.send_wait_queue = []
        for task in tasks:
            # Check if there is a send port available

            self.state.eventq.post_event(
                Event(self.state, task, timestamp=time, type="execute_StartSend")
            )
        return
    
    def task_free_SendPort(self, event: Event, task: Task) -> None:
        self.state.last_send[task.s_node][task.s_port] = self.state.time    
        self.state.device_ports[task.s_node]["send"].append(task.s_port)

        # Get list of tasks to start (those in the wait queue and those that depend on this task)
        self.__try_start_send(self.state.time + QUANTUM)

        return
    

    def task_execute_send_g(self, event: Event, task: Task) -> None:
        # print(f"Task {task.id} execute g at time {self.state.time}.")

        # Compute elapsed time
        elapse_time = max(0.0,
                          self.state.profiles[(task.s_node, task.d_node)]["g"](task.size) \
                            - (self.state.time - self.state.last_send[task.s_node][task.s_port]))
        
        # Assign task o_s at the end of the event
        self.state.eventq.post_event(
            Event(self.state, task, timestamp=self.state.time + elapse_time, type="execute_Send")
        )

        return
    

    # ================================================== RECV LOGIC

    def task_start_Recv(self, event: Event, task: Task) -> None:
        if task.d_port != -1:
            # If the task has a specific port, we can start it immediately
            self.state.eventq.post_event(
                Event(self.state, task, timestamp=self.state.time + self.state.augmented_task_latency[task.id], type="execute_recv_g")
            )
            return
        
        if task not in self.state.recv_buffer:
            # Initialize the recv buffer for this task
            self.state.recv_buffer[task] = []
            task.first_recv_time = self.state.time

        # Add task to the receive wait queue
        self.state.recv_buffer[task].append(self.state.time)

        self.task_occupy_Recv(event, task)
        
        return
    

    def task_occupy_Recv(self, event: Event, task: Task) -> None:
        # Find available send port
        if self.state.device_ports[task.d_node]["recv"]:
            # Update port information
            task.d_port = self.state.device_ports[task.d_node]["recv"].pop(0)

            # Update augmented task latency
            self.state.augmented_task_latency[task.id] = self.state.time - task.first_recv_time

            for t in self.state.recv_buffer[task]:
                # Post the event for each time in the buffer
                self.state.eventq.post_event(
                    Event(self.state, task, timestamp=t + self.state.augmented_task_latency[task.id], type="execute_recv_g")
                )

            self.state.recv_buffer[task] = []
        return
    

    def __try_start_recv(self, time) -> None:
        for task, times in self.state.recv_buffer.items():
            # Check if there is a send port available
            if times:
                self.state.eventq.post_event(
                    Event(self.state, task, timestamp=time, type="execute_OccupyRecv")
                )
        return

    def task_free_RecvPort(self, event: Event, task: Task) -> None:
        self.state.last_recv[task.d_node][task.d_port] = self.state.time    
        self.state.device_ports[task.d_node]["recv"].append(task.d_port)

        # Get list of tasks to start (those in the wait queue and those that depend on this task)
        self.__try_start_recv(self.state.time + QUANTUM)

        return

    def task_execute_recv_g(self, event: Event, task: Task) -> None:
        # print(f"Task {task.id} execute g at time {self.state.time}.")

        if task.recv_g == -1:
            # Compute elapsed time
            elapse_time = max(0.0,
                            self.state.profiles[(task.s_node, task.d_node)]["g"](task.size) \
                                - (self.state.time - self.state.last_recv[task.d_node][task.d_port]))
            task.recv_g = elapse_time
        else:
            elapse_time = task.recv_g
        
        # Assign task o_s at the end of the event
        self.state.eventq.post_event(
            Event(self.state, task, timestamp=self.state.time + elapse_time, type="execute_Recv")
        )

        return

    
    # ===================================================

    def task_done(self, event: Event, task: Task)-> None:
        # print(f"Task {task.id} done at time {self.state.time}.")
        # Record task stats
        self.state.stats["finished_tasks"].append((self.state.time, task.id))
        self.state.stats["tasks_done"][task.id] = self.state.time

        self.state.completed_tasks.add(task.id)

        self.__try_start_children_tasks(task)
        
        return
    

    # ======================================================= Send / Recv Logic

    def task_free_SendNIC(self, event: Event, task: Task) -> None:
        # print(f"Task {task.id} free Send NIC {self.state.time}.")

        group_id = self.state.group_mapping[task.s_node]["group_id"]
        self.state.group_queues[group_id]["availSend"] += 1
        return

    # The send event should add the event to the group's NIC queue
    def task_execute_Send(self, event: Event, task: Task) -> None:
        # print(f"Task {task.id} execute Send at time {self.state.time}.")

        send_group_id = self.state.group_mapping[task.s_node]["group_id"]

        # Check if task is an empty task
        if task.id == -1:
            # check if there are any tasks in the queue. If not, simply return
            groupq = self.state.group_queues[send_group_id]
            sendq = groupq["send"]
            if not sendq:
                return
            
            task = sendq.pop(0)

        send_group_id = self.state.group_mapping[task.s_node]["group_id"]
        recv_group_id = self.state.group_mapping[task.d_node]["group_id"]

        # Check if send group has a dict in contention
        if send_group_id not in self.state.stats["send_contention"].keys():
            self.state.stats["send_contention"][send_group_id] = {
                "success": 0,
                "attempts": 0,
                "max_queue_size": 0,
            }
        self.state.stats["send_contention"][send_group_id]["max_queue_size"] = max(self.state.stats["send_contention"][send_group_id]["max_queue_size"], 
                                                                                   len(self.state.group_queues[send_group_id]["send"]))


        # Check if src and dest are in the same L0 group. If they are,
        # we can avoid adding to the NIC queue
        if send_group_id == recv_group_id:
            # Compute the total amount of data to send
            bytes_to_send = min(self.state.mtu, task.send_left)

            self.state.stats["send_contention"][send_group_id]["attempts"] += 1
            
            # Modify contention stats
            self.state.stats["send_contention"][send_group_id]["success"] += 1

            # Post execute recv event at time + latency

            G = self.state.profiles[(task.s_node, task.d_node)]["G"](task.size)

            elapse_time = G * (bytes_to_send - 1) \
                + self.state.profiles[(task.s_node, task.d_node)]["L"](task.size)
            self.state.eventq.post_event(
                Event(self.state, task, timestamp=self.state.time + elapse_time, type="execute_StartRecv"))

            # Decrement size left
            task.send_left -= bytes_to_send

            # Post free send ports event at time + g
            if task.send_left == 0:
                # elapse_time += self.state.profiles[(task.s_node, task.d_node)]["g"](task.size)
                self.state.eventq.post_event(
                    Event(self.state, task, timestamp=self.state.time + G * (bytes_to_send - 1) - QUANTUM, type="free_SendPort"))

            # If size left is not 0, post event Send at time + G
            if task.send_left > 0:
                elapse_time = G * bytes_to_send
                self.state.eventq.post_event(
                    Event(self.state, task, timestamp=self.state.time + elapse_time, type="execute_Send"))

            return

        # Otherwise, we need to add the task to the NIC queue
        groupq = self.state.group_queues[send_group_id]

        # Add the task to the NIC queue
        sendq = groupq["send"]
        sendq.append(task)

        self.state.stats["send_contention"][send_group_id]["attempts"] += len(self.state.group_queues[send_group_id]["send"])

        # Check if there is a NIC port available
        # print(f"Group {send_group_id} has {groupq['availSend']} available Send NICs.")
        if groupq["availSend"] > 0: 
            groupq["availSend"] -= 1
            send_task = sendq.pop(0)

            # Modify contention stats
            self.state.stats["send_contention"][send_group_id]["success"] += 1

            # Compute the total amount of data to send
            bytes_to_send = min(self.state.mtu, send_task.send_left)

            G = self.state.profiles[(send_task.s_node, send_task.d_node)]["G"](send_task.size)

            # Post event Recv at time + latency
            elapse_time = G * (bytes_to_send - 1) \
                + self.state.profiles[(send_task.s_node, send_task.d_node)]["L"](send_task.size)
            e = Event(self.state, send_task, timestamp=self.state.time + elapse_time, type="execute_StartRecv")
            self.state.eventq.post_event(e)


            # Decrement size left
            send_task.send_left -= bytes_to_send

            # Post event free_SendNIC at time + G - QUANTUM
            elapse_time = G * bytes_to_send
            self.state.eventq.post_event(
                Event(self.state, send_task, timestamp=self.state.time + elapse_time - QUANTUM, type="free_SendNIC"))
            
            # Post free send ports event at time + g
            if send_task.send_left == 0:
                # elapse_time += self.state.profiles[(send_task.s_node, send_task.d_node)]["g"](send_task.size)
                self.state.eventq.post_event(
                    Event(self.state, send_task, timestamp=self.state.time + G * (bytes_to_send - 1) - QUANTUM, type="free_SendPort"))


            # If size left is not 0, post event Send at time + G
            if send_task.send_left > 0:
                self.state.eventq.post_event(
                    Event(self.state, send_task, timestamp=self.state.time + elapse_time, type="execute_Send"))
            elif sendq:
                next_task = sendq.pop(0)
                self.state.eventq.post_event(
                    Event(self.state, next_task, timestamp=self.state.time + elapse_time, type="execute_Send"))
            else:
                emtpy_task = Task(-1, task.s_node, task.d_node, -1, -1, -1)
                self.state.eventq.post_event(
                    Event(self.state, emtpy_task, timestamp=self.state.time + elapse_time, type="execute_Send"))
                
        # Otherwise, there are no NIC ports available, so we do nothing

        return

    # ================================================== 


    def task_free_RecvNIC(self, event: Event, task: Task) -> None:
        # print(f"Task {task.id} free Recv NIC {self.state.time}.")

        group_id = self.state.group_mapping[task.d_node]["group_id"]
        self.state.group_queues[group_id]["availRecv"] += 1
        return


    def task_execute_Recv(self, event: Event, task: Task) -> None:
        # print(f"Task {task.id} execute Recv at time {self.state.time}.")

        recv_group_id = self.state.group_mapping[task.d_node]["group_id"]
        
        # Check if task is an empty task
        if task.id == -1:
            # check if there are any tasks in the queue. If not, simply return
            groupq = self.state.group_queues[recv_group_id]
            recvq = groupq["recv"]
            if not recvq:
                return
            
            task = recvq.pop(0)

        send_group_id = self.state.group_mapping[task.s_node]["group_id"]
        recv_group_id = self.state.group_mapping[task.d_node]["group_id"]

        # Check if send group has a dict in contention
        if send_group_id not in self.state.stats["recv_contention"].keys():
            self.state.stats["recv_contention"][send_group_id] = {
                "success": 0,
                "attempts": 0,
                "max_queue_size": 0,
            }

        self.state.stats["recv_contention"][send_group_id]["max_queue_size"] = max(self.state.stats["recv_contention"][send_group_id]["max_queue_size"], 
                                                                                   len(self.state.group_queues[send_group_id]["recv"]))

        if send_group_id == recv_group_id:
            # Compute the total amount of data to send
            bytes_to_recv = min(self.state.mtu, task.recv_left)

            self.state.stats["recv_contention"][send_group_id]["attempts"] += 1

            # Modify contention stats
            self.state.stats["recv_contention"][send_group_id]["success"] += 1

            # if this is the lalf.state.hast event
            task.recv_left -= bytes_to_recv
            if task.recv_left == 0:
                elapse_time = self.state.profiles[(task.s_node, task.d_node)]["G"](task.size) * (bytes_to_recv - 1)
                self.state.eventq.post_event(
                    Event(self.state, task, timestamp=self.state.time + elapse_time, type="execute_or")
                )
                
                # Post free Recv ports event at time + g
                self.task_free_RecvPort(event, task)
            return
        
        groupq = self.state.group_queues[recv_group_id]

        # Add the task to the NIC queue
        recvq = groupq["recv"]
        recvq.append(task)

        self.state.stats["recv_contention"][send_group_id]["attempts"] += len(self.state.group_queues[send_group_id]["recv"])

        # Check if there is a NIC port available
        # print(f"Group {send_group_id} has {groupq['availRecv']} available Recv NICs.")
        if groupq["availRecv"] > 0: 
            groupq["availRecv"] -= 1
            recv_task = recvq.pop(0)

            # Modify contention stats
            self.state.stats["recv_contention"][send_group_id]["success"] += 1

            bytes_to_recv = min(self.state.mtu, task.recv_left)

            recv_task.recv_left -= bytes_to_recv
            # if this is the last event
            if recv_task.recv_left == 0:
                elapse_time = self.state.profiles[(recv_task.s_node, recv_task.d_node)]["G"](recv_task.size) * (bytes_to_recv - 1)
                self.state.eventq.post_event(
                    Event(self.state, recv_task, timestamp=self.state.time + elapse_time, type="execute_or")
                )

                # Post free Recv ports event at time
                self.task_free_RecvPort(event, recv_task)
                
            # We don't handle the else case, since the execute_Send event will do it for us
            
            # Post event free_RecvNIC at time + G - QUANTUM
            elapse_time = self.state.profiles[(recv_task.s_node, recv_task.d_node)]["G"](recv_task.size) * bytes_to_recv
            self.state.eventq.post_event(
                Event(self.state, recv_task, timestamp=self.state.time + elapse_time - QUANTUM, type="free_RecvNIC"))

            if recvq:
                next_task = recvq.pop(0)
                self.state.eventq.post_event(
                    Event(self.state, next_task, timestamp=self.state.time + elapse_time, type="execute_Recv"))
            else:
                emtpy_task = Task(-1, task.s_node, task.d_node, -1, -1, -1)
                self.state.eventq.post_event(
                    Event(self.state, emtpy_task, timestamp=self.state.time + elapse_time, type="execute_Recv"))
        

        return



    # ===================================================

    def task_execute_or(self, event: Event, task: Task) -> None:
        # print(f"Task {task.id} execute or at time {self.state.time}.")

        o_r = self.state.profiles[(task.s_node, task.d_node)]["o_r"](task.size)

        # Compute elapsed time
        # elapse_time = o_r + max(0.0, g - o_r)
        elapse_time = o_r

        # Assign task done at the end of the event
        self.state.eventq.post_event(
            Event(self.state, task, timestamp=self.state.time + elapse_time, type="done")
        )

        return


    
    
        