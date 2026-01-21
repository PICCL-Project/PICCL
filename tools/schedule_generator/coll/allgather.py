import math

from coll.collective import Collective

class AllGatherGenerator(Collective):
    def __init__(self, num_nodes: int, buffer_size: int, algorithm: str = "ring", config: dict = None):
        """
        Initialize the AllGatherGenerator class.

        Args:
            num_nodes (int): Number of nodes in the collective.
            buffer_size (int): Buffer size for each transfer.
            algorithm (str): Algorithm to use for scheduling. Default is "ring".
            config (dict): Configuration dictionary. Default is None.
        """
        super().__init__(num_nodes, buffer_size, algorithm, config)
        self.collective_type = "allgather"
        self.algorithms = {
            "ring": self.__ring,
            "k_ring": self.__k_ring,
            "RD": self.__recursive_doubling,
            "RM": self.__recursive_multiplying,
            "PRD": self.__permuted_recursive_doubling,
            "PRM": self.__permuted_recursive_multiplying
        }


    def __ring(self) -> str:
        """
        Generate the ring schedule for allgather operation.
        
        Returns:
            str: The generated ring schedule as an XML string.
        """
        xml_lines = []
        xml_lines.append(f'<collective name="{self.collective_type}" algorithm="{self.algorithm}" nodes="{self.num_nodes}" buffer_size="{self.buffer_size}">')

        transfer_id = 0
        # Calculate the size of each transfer
        transfer_size = self.buffer_size // self.num_nodes
        st_dep = {i:[-1] for i in range(self.num_nodes)}

        for r in range(self.num_nodes - 1): 
            xml_lines.append(f'\n    # (round {r})')

            new_st_dep = {i:[-1] for i in range(self.num_nodes)}
            for src in range(self.num_nodes):
                dst = (src + 1) % self.num_nodes
                deps = st_dep[src]

                xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{",".join([str(n) for n in deps])}"/>')

                if new_st_dep[dst] == [-1]:
                    new_st_dep[dst] = [transfer_id]
                else:
                    new_st_dep[dst].append(transfer_id)
                
                if new_st_dep[src] == [-1]:
                    new_st_dep[src] = [transfer_id]
                else:
                    new_st_dep[src].append(transfer_id)

                transfer_id += 1
            st_dep = new_st_dep

            
        xml_lines.append('</collective>')
        return "\n".join(xml_lines)
    


    def __k_ring(self) -> str:
        """
        Generate the k-ring schedule for allgather operation.
        
        Returns:
            str: The generated k-ring schedule in XML format.
        """
        k = self.config.get("k", 1)

        if self.num_nodes % k != 0:
            raise ValueError("Number of nodes must be divisible by k.")

        xml_lines = []
        xml_lines.append(f'<collective name="allgather" algorithm="k-ring" k="{k}" nodes="{self.num_nodes}" buffer_size="{self.buffer_size}">')

        transfer_id = 0
        # Calculate the size of each transfer
        transfer_size = self.buffer_size // self.num_nodes
        # Generate intra-group rounds
        num_groups = self.num_nodes // k
        # Source-transfer dependencies
        st_dep = {i:[-1] for i in range(self.num_nodes)}

        # Generate inter-group rounds
        for group in range(num_groups - 1):
            xml_lines.append(f'\n    # (inter-group {group})')

            new_st_dep = {i:[-1] for i in range(self.num_nodes)}
            for src in range(self.num_nodes): 
                dst = (src + k) % self.num_nodes
                deps = st_dep[src]
                xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{",".join([str(n) for n in deps])}"/>')
                
                if new_st_dep[dst] == [-1]:
                    new_st_dep[dst] = [transfer_id]
                else:
                    new_st_dep[dst].append(transfer_id)
                
                if new_st_dep[src] == [-1]:
                    new_st_dep[src] = [transfer_id]
                else:
                    new_st_dep[src].append(transfer_id)

                transfer_id += 1
            st_dep = new_st_dep

        # Generate intra-group rounds
        for g in range(num_groups):

            for r in range(k - 1): 
                xml_lines.append(f'\n    # (intra-group {g} round {r})')

                new_st_dep = {i:[-1] for i in range(self.num_nodes)}
                for src in range(self.num_nodes):
                    group = src // k
                    group_offset = group * k
                    group_id = src % k

                    dst = group_offset + (group_id + 1) % k
                    deps = st_dep[src]

                    xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{",".join([str(n) for n in deps])}"/>')

                    if new_st_dep[dst] == [-1]:
                        new_st_dep[dst] = [transfer_id]
                    else:
                        new_st_dep[dst].append(transfer_id)

                    if new_st_dep[src] == [-1]:
                        new_st_dep[src] = [transfer_id]
                    else:
                        new_st_dep[src].append(transfer_id)

                    transfer_id += 1

                st_dep = new_st_dep


        xml_lines.append('</collective>')
        return "\n".join(xml_lines)
    
    
    def __recursive_doubling(self) -> str:
        xml_lines = []
        xml_lines.append(f'<collective name="allgather" algorithm="recursive-double" nodes="{self.num_nodes}" buffer_size="{self.buffer_size}">')

        # Check num nodes is a power of 2
        if self.num_nodes & (self.num_nodes - 1) != 0:
            raise ValueError("Number of nodes must be a power of 2.")

        # Check buffer size divides evenly across nodes
        if self.buffer_size % self.num_nodes:
            raise ValueError("Buffer size must divide evenly over the number of nodes.")

        message_size = int(self.buffer_size / self.num_nodes)
        mask = 1
        transfer_id = 0
        deps = {}

        while message_size < self.buffer_size:
            
            xml_lines.append(f'\n    # (round size: {message_size})')

            new_deps = {}
            for rank in range(self.num_nodes):
                dst = rank ^ mask
                deps.setdefault(rank, [-1])
                xml_lines.append(f'    <transfer id="{transfer_id}" size="{message_size}" src="{int(rank)}" dst="{int(dst)}" deps="{deps[rank][-1]}"/>')
                new_deps.setdefault(dst, [-1])
                new_deps[dst].append(transfer_id)
                transfer_id += 1

            deps = new_deps
            message_size *= 2
            mask *= 2

        xml_lines.append('</collective>')
        return "\n".join(xml_lines)


    def __recursive_multiplying(self) -> str:

        k = self.config.get("k", 2)

        xml_lines = []
        xml_lines.append(f'<collective name="allgather" algorithm="recursive-multiply" nodes="{self.num_nodes}" buffer_size="{self.buffer_size}">')

        # Check num nodes is a power of k
        res = math.log(self.num_nodes) / math.log(k)
        if not res == math.floor(res):
            raise ValueError("Number of nodes must be a power of {}.".format(k))

        # Check buffer size divides evenly across nodes
        if self.buffer_size % self.num_nodes:
            raise ValueError("Buffer size must divide evenly over the number of nodes.")

        message_size = int(self.buffer_size / self.num_nodes)
        transfer_id = 0
        deps = {}
        distance = 1
        next_distance = k

        while distance < self.num_nodes:
            
            xml_lines.append(f'\n    # (round size: {message_size})')

            new_deps = {}
            for rank in range(self.num_nodes):
                deps.setdefault(rank, [-1 for _ in range(k-1)])
                current_deps = str(deps[rank][-1])
                if current_deps != "-1":
                    for k_val in range(2,k):
                        current_deps = current_deps + "," + str(deps[rank][-k_val])
                starting_rank = int(rank / next_distance) * next_distance
                rank_offset = starting_rank + rank % distance
                for dst in range(rank_offset, starting_rank + next_distance, distance):
                    if(dst != rank):
                        xml_lines.append(f'    <transfer id="{transfer_id}" size="{message_size}" src="{int(rank)}" dst="{int(dst)}" deps="{current_deps}"/>')
                        new_deps.setdefault(dst, [-1])
                        new_deps[dst].append(transfer_id)
                        transfer_id += 1

            deps = new_deps
            message_size *= k
            distance = next_distance
            next_distance *= k

        xml_lines.append('</collective>')
        return "\n".join(xml_lines)


    def __permuted_recursive_doubling(self) -> str:
        xml_lines = []
        xml_lines.append(f'<collective name="allgather" algorithm="permuted-recursive-double" nodes="{self.num_nodes}" buffer_size="{self.buffer_size}">')

        # Check num nodes is a power of 2
        if self.num_nodes & (self.num_nodes - 1) != 0:
            raise ValueError("Number of nodes must be a power of 2.")

        transfer_id = 0
        # Source-transfer dependencies
        st_dep = {i:[-1] for i in range(self.num_nodes)}

        transfer_size = self.buffer_size // self.num_nodes
        size = self.num_nodes
        while size > 1: 
            new_st_dep = {i:[-1] for i in range(self.num_nodes)}

            xml_lines.append(f'\n    # (round size: {size})')

            for rank in range(self.num_nodes):
                relRank = rank % size
                nextSize = size / 2
                root = rank // size * size

                transfer_rank = root + (relRank + nextSize) % size

                deps = st_dep[rank]
                xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{int(rank)}" dst="{int(transfer_rank)}" deps="{",".join([str(n) for n in deps])}"/>')
                new_st_dep[transfer_rank] = ([transfer_id])
                transfer_id += 1

            st_dep = new_st_dep
            size = size / 2
            transfer_size *= 2

        xml_lines.append('</collective>')
        return "\n".join(xml_lines)

    def __permuted_recursive_multiplying(self) -> str:
        k = self.config.get("k", 2)

        xml_lines = []
        xml_lines.append(f'<collective name="allgather" algorithm="permuted-recursive-multiply" nodes="{self.num_nodes}" buffer_size="{self.buffer_size}">')

        # Check num nodes is a power of k
        res = math.log(self.num_nodes) / math.log(k)
        if not res == math.floor(res):
            raise ValueError("Number of nodes must be a power of {}.".format(k))

        # Check buffer size divides evenly across nodes
        if self.buffer_size % self.num_nodes:
            raise ValueError("Buffer size must divide evenly over the number of nodes.")

        message_size = int(self.buffer_size / self.num_nodes)
        transfer_id = 0
        deps = {}
        distance = self.num_nodes
        next_distance = self.num_nodes // k

        while distance > 1:
            
            xml_lines.append(f'\n    # (round size: {message_size})')

            new_deps = {}
            for rank in range(self.num_nodes):
                deps.setdefault(rank, [-1 for _ in range(k-1)])
                current_deps = str(deps[rank][-1])
                if current_deps != "-1":
                    for k_val in range(2,k):
                        current_deps = current_deps + "," + str(deps[rank][-k_val])
                root = int(rank / distance) * distance
                relRank = rank % distance
                for i in range(1,k):
                    dst = root + (relRank + i * next_distance) % distance
                    if(dst != rank):
                        xml_lines.append(f'    <transfer id="{transfer_id}" size="{message_size}" src="{int(rank)}" dst="{int(dst)}" deps="{current_deps}"/>')
                        new_deps.setdefault(dst, [-1])
                        new_deps[dst].append(transfer_id)
                        transfer_id += 1

            deps = new_deps
            message_size *= k
            distance = next_distance
            next_distance = next_distance // k

        xml_lines.append('</collective>')
        return "\n".join(xml_lines)
    