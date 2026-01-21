import math

from coll.collective import Collective

class AllReduceGenerator(Collective):
    def __init__(self, num_nodes: int, buffer_size: int, algorithm: str = "ring", config: dict = None):
        """
        Initialize the AllReduceGenerator class.

        Args:
            num_nodes (int): Number of nodes in the collective.
            buffer_size (int): Buffer size for each transfer.
            algorithm (str): Algorithm to use for scheduling. Default is "ring".
            config (dict): Configuration dictionary. Default is None.
        """
        super().__init__(num_nodes, buffer_size, algorithm, config)
        self.collective_type = "allreduce"
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
        Generate the ring schedule for ReduceScatterGenerator operation.
        
        Returns:
            str: The generated ring schedule as an XML string.
        """
        num_nodes = self.num_nodes
        buffer_size = self.buffer_size


        xml_lines = []
        xml_lines.append(f'<collective name="{self.collective_type}" algorithm="{self.algorithm}" nodes="{num_nodes}" buffer_size="{buffer_size}">')

        transfer_id = 0
        # Calculate the size of each transfer
        transfer_size = buffer_size / num_nodes
        # Generate intra-group rounds
        # Source-transfer dependencies
        deps = [-1] * num_nodes
        
        for i in range(num_nodes - 1):
            xml_lines.append(f'\n    # (round {i})')

            new_deps = [-1] * num_nodes
            for src in range(num_nodes):
                dst = (src + 1) % num_nodes
                d = deps[src]
                
                xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{d}"/>')
                
                new_deps[dst] = transfer_id
                transfer_id += 1
            
            deps = new_deps

        xml_lines.append('</collective>')
        return "\n".join(xml_lines)
    


    def __k_ring(self) -> str:
        """
        Generate the k-ring schedule for AllReduceGenerator operation.
        
        Args:
            num_nodes (int): Number of nodes in the k-ring.
            k (int): Number of nodes in each group (k-ring parameter).
            buffer_size (int): Buffer size for each transfer.

        Returns:
            str: The generated k-ring schedule in XML format.
        """
        k = self.config.get("k", 1)
        num_nodes = self.num_nodes
        buffer_size = self.buffer_size

        if num_nodes % k != 0:
            raise ValueError("Number of nodes must be divisible by k.")

        xml_lines = []
        xml_lines.append(f'<collective name="{self.collective_type}" algorithm="k-ring" k="{k}" nodes="{num_nodes}" buffer_size="{buffer_size}">')

        transfer_id = 0
        # Calculate the size of each transfer
        message_size = int(buffer_size / num_nodes)
        # Generate intra-group rounds
        num_groups = int(num_nodes // k)

        transfer_id = 0

        deps = {}

        # Initial intra-group
        for round_num in range(1, k):
            xml_lines.append(f'\n    # (Initial intra-group round: {round_num})\n')

            new_deps = {}
            for group_num in range(0, num_groups):
                for rel_rank in range(0, k):
                    rank = group_num*k + rel_rank
                    dst = rank + 1
                    # Wrap-around
                    if dst >= (group_num + 1) * k:
                        dst -= k
                    deps.setdefault(rank, [-1])
                    xml_lines.append(f'    <transfer id="{transfer_id}" size="{message_size}" src="{int(rank)}" dst="{int(dst)}" deps="{deps[rank][-1]}"/>')
                    new_deps.setdefault(dst, [-1])
                    new_deps[dst].append(transfer_id)
                    transfer_id += 1

            deps = new_deps

        #Inter-Group
        for round_num in range(1, num_groups):
            xml_lines.append(f'\n    # (Inter-group round: {round_num})\n')

            new_deps = {}
            for group_num in range(0, num_groups):
                for rel_rank in range(0, k):
                    rank = group_num*k + rel_rank
                    dst = rank + k

                    # Wrap-around
                    if dst >= num_nodes:
                        dst -= num_nodes

                    deps.setdefault(rank, [-1])
                    xml_lines.append(f'    <transfer id="{transfer_id}" size="{message_size}" src="{int(rank)}" dst="{int(dst)}" deps="{deps[rank][-1]}"/>')
                    new_deps.setdefault(dst, [-1])
                    new_deps[dst].append(transfer_id)
                    transfer_id += 1
            
            deps = new_deps
            
            # Mid-inter round-intra rounds
            for intra_round_num in range(1, k):
                xml_lines.append(f'\n        # (Round {round_num}: Intra-group round: {intra_round_num})\n')

                new_deps = {}
                for group_num in range(0, num_groups):
                    for rel_rank in range(0, k):
                        rank = group_num*k + rel_rank
                        dst = rank + 1
                        # Wrap-around
                        if dst >= (group_num + 1) * k:
                            dst -= k
                        deps.setdefault(rank, [-1])
                        xml_lines.append(f'    <transfer id="{transfer_id}" size="{message_size}" src="{int(rank)}" dst="{int(dst)}" deps="{deps[rank][-1]}"/>')
                        new_deps.setdefault(dst, [-1])
                        new_deps[dst].append(transfer_id)
                        transfer_id += 1

                deps = new_deps

        # Allgather ring, copied from allgather.py
        transfer_size = self.buffer_size // self.num_nodes
        st_dep = deps

        for r in range(self.num_nodes - 1): 
            xml_lines.append(f'\n    # (round {r})')

            new_st_dep = {i:[-1] for i in range(self.num_nodes)}
            for src in range(self.num_nodes):
                dst = (src + 1) % self.num_nodes
                deps = st_dep[src]

                # filter out any -1 dependencies
                deps = [d for d in deps if d != -1]

                xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{",".join([str(n) for n in deps])}"/>')

                new_st_dep[dst] = ([transfer_id])
                transfer_id += 1

            st_dep = new_st_dep


        xml_lines.append('</collective>')
        return "\n".join(xml_lines)
    

    def __recursive_doubling(self) -> str:
        xml_lines = []
        xml_lines.append(f'<collective name="{self.collective_type}" algorithm="recursive-double" nodes="{self.num_nodes}" buffer_size="{self.buffer_size}">')

        # Check num nodes is a power of 2
        if self.num_nodes & (self.num_nodes - 1) != 0:
            raise ValueError("Number of nodes must be a power of 2.")

        message_size = self.buffer_size
        mask = 1
        transfer_id = 0
        deps = {}

        while mask < self.num_nodes:
            
            xml_lines.append(f'\n    # (round size: {mask})')

            new_deps = {}
            for rank in range(self.num_nodes):
                dst = rank ^ mask
                deps.setdefault(rank, [-1])
                xml_lines.append(f'    <transfer id="{transfer_id}" size="{message_size}" src="{int(rank)}" dst="{int(dst)}" deps="{deps[rank][-1]}"/>')
                new_deps.setdefault(dst, [-1])
                new_deps[dst].append(transfer_id)
                transfer_id += 1

            deps = new_deps
            mask *= 2

        xml_lines.append('</collective>')
        return "\n".join(xml_lines)
    

    def __recursive_multiplying(self) -> str:
        k = self.config.get("k", 2)

        xml_lines = []
        xml_lines.append(f'<collective name="{self.collective_type}" algorithm="recursive-multiply" nodes="{self.num_nodes}" buffer_size="{self.buffer_size}">')

        # Check num nodes is a power of k
        res = math.log(self.num_nodes) / math.log(k)
        if not res == math.floor(res):
            raise ValueError("Number of nodes must be a power of {}.".format(k))

        message_size = self.buffer_size
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
            distance = next_distance
            next_distance *= k

        xml_lines.append('</collective>')
        return "\n".join(xml_lines)


    def __permuted_recursive_doubling(self) -> str:
        xml_lines = []
        xml_lines.append(f'<collective name="{self.collective_type}" algorithm="permuted-recursive-double" nodes="{self.num_nodes}" buffer_size="{self.buffer_size}">')

        # Check num nodes is a power of 2
        if self.num_nodes & (self.num_nodes - 1) != 0:
            raise ValueError("Number of nodes must be a power of 2.")

        transfer_id = 0
        # Source-transfer dependencies
        st_dep = {i:[-1] for i in range(self.num_nodes)}

        transfer_size = self.buffer_size
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

        xml_lines.append('</collective>')
        return "\n".join(xml_lines)
    


    def __permuted_recursive_multiplying(self) -> str:
        k = self.config.get("k", 2)

        xml_lines = []
        xml_lines.append(f'<collective name="{self.collective_type}" algorithm="permuted-recursive-multiply" nodes="{self.num_nodes}" buffer_size="{self.buffer_size}">')

        # Check num nodes is a power of k
        res = math.log(self.num_nodes) / math.log(k)
        if not res == math.floor(res):
            raise ValueError("Number of nodes must be a power of {}.".format(k))

        message_size = self.buffer_size
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
            distance = next_distance
            next_distance = next_distance // k

        xml_lines.append('</collective>')
        return "\n".join(xml_lines)
    