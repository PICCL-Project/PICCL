from coll.allgather import AllGatherGenerator

def schedule_generator(num_nodes: int, buffer_size: int, collective: str = "allgather", algorithm: str = "ring", config: dict = None) -> str:
    """
    Generate the schedule for allgather operation.
    
    Args:
        num_nodes (int): Number of nodes in the k-ring.
        buffer_size (int): Buffer size for each transfer.
        algorithm (str): Algorithm to use for scheduling. Default is "ring".
        config (dict): Configuration dictionary. Default is None.
    
    Returns:
        str: The generated schedule as an XML string.
    """
    if collective == "allgather":
        generator = AllGatherGenerator(num_nodes, buffer_size, algorithm, config)
    else:
        raise ValueError(f"Unsupported collective operation: {collective}")
    
    
    # Generate the schedule for each node
    nodes_schedule: list[tuple[list, list]] = []
    for node in range(num_nodes):
        nodes_schedule.append(generator.generate(node))


    # Take each node's send and recv schedule and generate an xml to denote the schedule
    xml_lines = []
    xml_lines.append(f'<collective name="{collective}" algorithm="{algorithm}" nodes="{num_nodes}" buffer_size="{buffer_size}">')

    send_waiting = set()
    recv_waiting = set()
    send_blocked = {node : False for node in range(num_nodes)}
    recv_blocked = {node : False for node in range(num_nodes)}

    existing_data = {}

    transfer_id = 0

    while True:
        for node in range(num_nodes):
            send_schedule, recv_schedule = nodes_schedule[node]
            
            if not send_blocked[node]:
                
                send = send_schedule[0]
                deps = send["deps"]


                # if there isn't a corresponding recv, post it to send_waiting 
                key = (send["from"], send["to"], send["size"], "recv")
                
                if key in recv_waiting:
                    recv_waiting.remove(key)
                    xml_lines.append(f'<transfer id="{transfer_id}" size="{send["size"]}" src="{send["from"]}" dst="{send["to"]}" deps="2,4"/>')


                else:
                    # Otherwise post it to send_waiting
                    send_waiting.add((send["from"], send["to"], send["size"], "send"))
                

            if not recv_blocked[node]:
                pass
 
    for node in 
    
    return generator.generate()