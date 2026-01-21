class AllGatherGenerator:
    def __init__(self, num_nodes: int, buffer_size: int, algorithm: str = "ring", config: dict = None):
        self.algorithm = algorithm
        self.num_nodes = num_nodes
        self.buffer_size = buffer_size
        self.transfer_size = buffer_size // num_nodes


    def generate(self, node):
        if self.algorithm == "ring":
            return self.__ring(node)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")


    def __ring(self, rank: int) -> tuple[list[dict], list[dict]]:
        """
        Generate the ring schedule for allgather operation.
        
        Returns:
            list[dict]: The generated ring schedule as a list of dictionaries.
        """
        sends = []
        recvs = []

        recv_from = (rank - 1 + self.num_nodes) % self.num_nodes
        send_to = (rank + 1) % self.num_nodes

        send_offset = rank
        recv_offset = recv_from

        for i in range(self.num_nodes - 1):

            transfer_size = min(self.transfer_size, self.buffer_size - (recv_offset * self.transfer_size))
            recv = {
                "from": recv_from,
                "to": rank,
                "offset": recv_offset * self.transfer_size,
                "size": transfer_size,
                "tag": i,
                "type": "recv"
            }
            
            transfer_size = min(self.transfer_size, self.buffer_size - (send_offset * self.transfer_size))
            send = {
                "from": rank,
                "to": send_to,
                "deps": [(recv_from, send_offset * self.transfer_size)],
                "size": transfer_size,
                "tag": i,
                "type": "send"
            }

            recvs.append(recv)
            sends.append(send)

            send_offset = recv_offset
            recv_offset = (recv_from - 1 + self.num_nodes) % self.num_nodes

        return (sends, recvs)

