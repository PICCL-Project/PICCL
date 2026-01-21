from coll.collective import Collective

class ReduceScatterGenerator(Collective):
    def __init__(self, num_nodes: int, buffer_size: int, algorithm: str = "ring", config: dict = None):
        """
        Initialize the ReduceScatterGenerator class.

        Args:
            num_nodes (int): Number of nodes in the collective.
            buffer_size (int): Buffer size for each transfer.
            algorithm (str): Algorithm to use for scheduling. Default is "ring".
            config (dict): Configuration dictionary. Default is None.
        """
        super().__init__(num_nodes, buffer_size, algorithm, config)
        self.collective_type = "reduce_scatter"
        self.algorithms = {
            "ring": self.__ring,
            "k_ring": self.__k_ring
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

        xml_lines.append('</collective>')
        return "\n".join(xml_lines)
    


    def __k_ring(self) -> str:
        """
        Generate the k-ring schedule for ReduceScatterGenerator operation.
        
        Args:   
            num_nodes (int): Number of nodes in the k-ring.
            k (int): Number of nodes in each group (k-ring parameter).
            buffer_size (int): Buffer size for each transfer.

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
                new_st_dep[dst] = ([transfer_id])
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

                    new_st_dep[dst] = ([transfer_id])
                    transfer_id += 1

                st_dep = new_st_dep


        xml_lines.append('</collective>')
        return "\n".join(xml_lines)