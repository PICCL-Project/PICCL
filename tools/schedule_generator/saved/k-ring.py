

class KRingScheduleGenerator:
    """
    A class to generate a ring schedule for collective communication algorithms.
    """
    def __init__(self, num_nodes: int, buffer_size: int, k: int, collective: str):
        """
        Initialize the RingScheduleGenerator. 
        Args:
            num_nodes (int): Number of nodes in the ring.
            buffer_size (int): Buffer size for each transfer.
            collective (str): Type of collective operation (e.g., "allgather").
        """
        self.num_nodes = num_nodes
        self.buffer_size = buffer_size
        self.collective = collective
        self.k = k
        

    def generate_schedule(self) -> str:
        """
        Generate the ring schedule in XML format.

        Returns:
            str: The generated ring schedule in XML format.
        """
        if self.collective == "allgather":
            return self.__allgather(self.num_nodes, self.k, self.buffer_size)
        elif self.collective == "allreduce":
            return self.__allreduce(self.num_nodes, self.k, self.buffer_size)
        elif self.collective == "broadcast":
            return self.__broadcast(self.num_nodes, self.k, self.buffer_size)
        elif self.collective == "reduce":
            return self.__reduce(self.num_nodes, self.k, self.buffer_size)
        
        else:
            raise ValueError(f"Unsupported collective operation: {self.collective}")

    def __allgather(self, num_nodes: int, k: int, buffer_size: int) -> str:
        """
        Generate the k-ring schedule for allgather operation.
        
        Args:
            num_nodes (int): Number of nodes in the k-ring.
            k (int): Number of nodes in each group (k-ring parameter).
            buffer_size (int): Buffer size for each transfer.

        Returns:
            str: The generated k-ring schedule in XML format.
        """
        if num_nodes % k != 0:
            raise ValueError("Number of nodes must be divisible by k.")

        xml_lines = []
        xml_lines.append(f'<collective name="allgather" algorithm="k-ring" k="{k}" nodes="{num_nodes}" buffer_size="{buffer_size}">')

        transfer_id = 0
        # Calculate the size of each transfer
        transfer_size = buffer_size // num_nodes
        # Generate intra-group rounds
        num_groups = num_nodes // k
        # Source-transfer dependencies
        st_dep = {i:[-1] for i in range(num_nodes)}

        # Generate inter-group rounds
        for group in range(num_groups - 1):
            xml_lines.append(f'\n    # (inter-group {group})')

            new_st_dep = {i:[-1] for i in range(num_nodes)}
            for src in range(num_nodes): 
                dst = (src + k) % num_nodes
                deps = st_dep[src]
                xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{",".join([str(n) for n in deps])}"/>')
                new_st_dep[dst] = ([transfer_id])
                transfer_id += 1
            st_dep = new_st_dep

        # Generate intra-group rounds
        for g in range(num_groups):

            for r in range(k - 1): 
                xml_lines.append(f'\n    # (intra-group {g} round {r})')

                new_st_dep = {i:[-1] for i in range(num_nodes)}
                for src in range(num_nodes):
                    group = src // k
                    group_offset = group * k
                    group_id = src % k

                    dst = group_offset + (group_id + 1) % k
                    deps = st_dep[src]

                    xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{",".join([str(n) for n in deps])}"/>')

                    new_st_dep[dst] = ([transfer_id])
                    transfer_id += 1

                st_dep = new_st_dep


        xml_lines.append('</collective>')
        return "\n".join(xml_lines)
    

    def __reduce(self, num_nodes: int, k: int, buffer_size: int) -> str:
        """
        Generate the k-ring schedule for reduce operation.
        
        Args:
            num_nodes (int): Number of nodes in the k-ring.
            k (int): Number of nodes in each group (k-ring parameter).
            buffer_size (int): Buffer size for each transfer.

        Returns:
            str: The generated k-ring schedule in XML format.
        """

        if num_nodes % k != 0:
            raise ValueError("Number of nodes must be divisible by k.")

        xml_lines = []
        xml_lines.append(f'<collective name="reduce" algorithm="k-ring" k="{k}" nodes="{num_nodes}" buffer_size="{buffer_size}">')

        transfer_id = 0
        # Calculate the size of each transfer
        transfer_size = buffer_size
        # Generate intra-group rounds
        num_groups = num_nodes // k
        # Source-transfer dependencies
        st_dep = {i:[-1] for i in range(num_nodes)}
        
        # Generate intra-group rounds
        for r in range(k - 1): 

            new_st_dep = {i:[-1] for i in range(num_nodes)}
            for g in range(num_groups):
                xml_lines.append(f'\n    # (intra-group {g} round {r})')

                rel_src = r + 1
                rel_dst = (rel_src + 1) % k

                src = g * k + rel_src
                dst = g * k + rel_dst

                deps = st_dep[src]
                st_dep[src] = [-1]
                xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{",".join(str(n) for n in deps)}"/>')

                new_st_dep[dst] = [transfer_id]
                transfer_id += 1

            for key, val in new_st_dep.items():
                if val != [-1]:
                    if st_dep[key] == [-1]:
                        st_dep[key] = val
                    else:
                        st_dep[key] += val
        
        # Generate inter-group rounds
        for g in range(1, num_groups):
            new_st_dep = {i:[-1] for i in range(num_nodes)}
            
            xml_lines.append(f'\n    # (inter-group {g})')

            src = g * k
            dst = (src + k) % num_nodes

            deps = st_dep[src]
            st_dep[src] = ([-1])
            xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{",".join([str(n) for n in deps])}"/>')

            new_st_dep[dst] = [transfer_id]
            transfer_id += 1

            for key, val in new_st_dep.items():
                if val != [-1]:
                    if st_dep[key] == [-1]:
                        st_dep[key] = val
                    else:
                        st_dep[key] += val


        xml_lines.append('</collective>')
        return "\n".join(xml_lines)


    def __broadcast(self, num_nodes: int, k: int, buffer_size: int) -> str:
        """
        Generate the k-ring schedule for broadcast operation.
        
        Args:
            num_nodes (int): Number of nodes in the k-ring.
            k (int): Number of nodes in each group (k-ring parameter).
            buffer_size (int): Buffer size for each transfer.

        Returns:
            str: The generated k-ring schedule in XML format.
        """

        if num_nodes % k != 0:
            raise ValueError("Number of nodes must be divisible by k.")

        xml_lines = []
        xml_lines.append(f'<collective name="broadcast" algorithm="k-ring" k="{k}" nodes="{num_nodes}" buffer_size="{buffer_size}">')

        transfer_id = 0
        # Calculate the size of each transfer
        transfer_size = buffer_size
        # Generate intra-group rounds
        num_groups = num_nodes // k
        # Source-transfer dependencies
        st_dep = {i:[-1] for i in range(num_nodes)}
        
        # Generate inter-group rounds
        for g in range(0, num_groups-1):
            new_st_dep = {i:[-1] for i in range(num_nodes)}
            
            xml_lines.append(f'\n    # (inter-group {g})')

            src = g * k
            dst = (src + k) % num_nodes

            deps = st_dep[src]
            st_dep[src] = ([-1])
            xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{",".join([str(n) for n in deps])}"/>')

            new_st_dep[dst] = [transfer_id]
            transfer_id += 1

            for key, val in new_st_dep.items():
                if val != [-1]:
                    if st_dep[key] == [-1]:
                        st_dep[key] = val
                    else:
                        st_dep[key] += val

        # Generate intra-group rounds
        for r in range(k - 1): 

            new_st_dep = {i:[-1] for i in range(num_nodes)}
            for g in range(num_groups):
                xml_lines.append(f'\n    # (intra-group {g} round {r})')

                rel_src = r
                rel_dst = (rel_src + 1) % k

                src = g * k + rel_src
                dst = g * k + rel_dst

                deps = st_dep[src]
                st_dep[src] = [-1]
                xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{",".join(str(n) for n in deps)}"/>')

                new_st_dep[dst] = [transfer_id]
                transfer_id += 1

            for key, val in new_st_dep.items():
                if val != [-1]:
                    if st_dep[key] == [-1]:
                        st_dep[key] = val
                    else:
                        st_dep[key] += val
        
        


        xml_lines.append('</collective>')
        return "\n".join(xml_lines)


    def __allreduce(self, num_nodes: int, k: int, buffer_size: int) -> str:
        pass


if __name__ == "__main__":
    # Example usage
    num_nodes = 8
    buffer_size = 4097
    k = 2
    collective = "broadcast"

    generator = KRingScheduleGenerator(num_nodes, buffer_size, k, collective)
    schedule = generator.generate_schedule()
    print(schedule)