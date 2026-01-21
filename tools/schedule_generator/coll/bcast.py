from coll.collective import Collective

class BroadcastGenerator(Collective):
    def __init__(self, num_nodes: int, buffer_size: int, algorithm: str = "ring", config: dict = None):
        """
        Initialize the BroadcastGenerator class.

        Args:
            num_nodes (int): Number of nodes in the collective.
            buffer_size (int): Buffer size for each transfer.
            algorithm (str): Algorithm to use for scheduling. Default is "ring".
            config (dict): Configuration dictionary. Default is None.
        """
        super().__init__(num_nodes, buffer_size, algorithm, config)
        self.collective_type = "broadcast"
        self.algorithms = {
            "ring": self.__ring,
            "k_ring": self.__k_ring
        }


    def __ring(self) -> str:
        """
        Generate the ring schedule for bcast operation.
        
        Returns:
            str: The generated ring schedule as an XML string.
        """
        num_nodes = self.num_nodes
        buffer_size = self.buffer_size

        xml_lines = []
        xml_lines.append(f'<collective name="{self.collective_type}" algorithm="{self.algorithm}" nodes="{num_nodes}" buffer_size="{buffer_size}">')

        transfer_id = 0
        # Calculate the size of each transfer
        transfer_size = buffer_size
        # Generate intra-group rounds
        # Source-transfer dependencies
        st_dep = {i:[-1] for i in range(num_nodes)}

        # Generate inter-group rounds
        for r in range(0, num_nodes-1):
            new_st_dep = {i:[-1] for i in range(num_nodes)}
            
            xml_lines.append(f'\n    # (inter-group {r})')

            src = r
            dst = (src + 1) % num_nodes

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
    


    def __k_ring(self) -> str:
        """
        Generate the k-ring schedule for bcast operation.
        
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
        xml_lines.append(f'<collective name="reduce" algorithm="k-ring" k="{k}" nodes="{num_nodes}" buffer_size="{buffer_size}">')

        transfer_id = 0
        # Calculate the size of each transfer
        transfer_size = buffer_size
        # Generate intra-group rounds
        num_groups = num_nodes // k
        
        deps = [-1 for _ in range(num_nodes)]
        for group in range(num_groups - 1):
            src = group * k
            dst = (src + k) % num_nodes
            xml_lines.append(f'\n    # (intra-group {group})')
            xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{deps[src]}"/>')

            deps[dst] = transfer_id
    
            transfer_id += 1
        
        inter_deps = [-1 for _ in range(num_nodes)]
        for group in range(num_groups):
            xml_lines.append(f'\n    # (intra-group {group})')
            for i in range(0, k-1):
                src = group * k + i
                dst = group * k + (i + 1) % k

                if i == 0 and deps[src] == -1:
                    xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="-1"/>')
                elif i == 0:
                    xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{deps[src]}"/>')
                else:
                    xml_lines.append(f'    <transfer id="{transfer_id}" size="{transfer_size}" src="{src}" dst="{dst}" deps="{inter_deps[src]}"/>')

                inter_deps[dst] = transfer_id
                transfer_id += 1

        xml_lines.append('</collective>')
        return "\n".join(xml_lines)