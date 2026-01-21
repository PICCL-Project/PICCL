from __future__ import annotations

# Stores task-specific info
class Task():
    def __init__(self, id: int = -1, s_node: int = -1, d_node: int = -1, size: int = 0, 
                 deps: set[int] = set(), children: set[int] = set()) -> None:
        self.id = id
        self.s_node = s_node
        self.d_node = d_node
        self.size = size
        self.deps = deps
        self.children = children
        self.send_left = size
        self.recv_left = size
        self.s_port = -1
        self.d_port = -1
        self.recv_g = -1

    def __repr__(self) -> str:
        return f"Task(id={self.id}, s_node={self.s_node}, d_node={self.d_node}, size={self.size}, task_dep={self.deps})"


