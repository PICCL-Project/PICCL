import xml.etree.ElementTree as ET
from collections import defaultdict

from task import Task
from utils import get_path

def get_children_tasks(task_id: int, tasks: dict[int, Task]) -> list[Task]:
    """
    Get the children tasks of a specific task.

    Args:
        task_id (int): The ID of the task whose children are to be found.
        tasks (dict[int, Task]): Dictionary of all tasks.

    Returns:
        list[Task]: List of children tasks.
    """
    children = []
    for task in tasks.values():
        if task_id in task.deps:
            children.append(task.id)
    return children


def parse_schedule_to_tasks(xml_file: str) -> tuple[int, dict[int:Task]]:
    """
    Parse the XML file and convert <transfer> elements into Task objects.

    Args:
        xml_file (str): Path to the XML file.

    Returns:
        list[Task]: List of Task objects.
    """
    tasks = {}
    tree = ET.parse(xml_file)
    root = tree.getroot()

    num_nodes = int(root.get("nodes"))

    # Iterate over all <transfer> elements
    for transfer in root.findall("transfer"):
        task_id = int(transfer.get("id"))
        s_node = int(transfer.get("src"))
        d_node = int(transfer.get("dst"))
        size = int(transfer.get("size"))
        deps = transfer.get("deps")

        # Parse dependencies into a set of integers
        dep_set = set()
        if deps and deps != "-1":
            dep_set = {int(dep) for dep in deps.split(",")}

        # Create a Task object
        task = Task(id=task_id, s_node=s_node, d_node=d_node, size=size, deps=dep_set)
        tasks[task_id] = task

    # Set the children tasks
    for task in tasks.values():
        task.children = get_children_tasks(task.id, tasks)

    return num_nodes, tasks


def parse_loggp_param(csv_paths) -> dict[int: dict[str, float]]:
    """
    Parse the list of CSV file and convert it into a dictionary of parameters.

    Args:
        csv_paths (list[str]): Path to the CSV file.

    Returns:
        dict[str:dict[int: dict[str, float]]]: Dictionary of parameters.
    """
    profiles = {}
    for csv_path in csv_paths:
        profile = {}
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(',')
                size = int(parts[0])
                
                profile[size] = dict(zip(["L","o_s","o_r","g","G"], [float(s) for s in parts[1:]]))

        profiles[csv_path] = profile

    return profiles


def parse_topology(xml_file: str) -> dict[tuple[int, int], list[tuple[int]]]:
    """
    Parse the topology XML file and determine which groups each pair of nodes belongs to.

    Args:
        xml_file (str): Path to the topology XML file.

    Returns:
        1. node_pairs_to_groups (dict[tuple[int, int], list[tuple[int]]]): A
            dictionary where the keys are pairs of nodes (tuples), and the values 
            lists of tuples of group IDs, level, and parameter file they belong to.
        2. files (set): A set of parameter files used in the topology.
        3. Node to level-0 group mapping (dict[int, int]): A dictionary mapping each node to its level-0 group.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Dictionary to store group memberships for each pair of nodes
    node_pairs_to_groups = defaultdict(list)
    files = set()
    group_mapping = {}

    def parse_group(group_element, current_group_id, current_level, current_param):
        """
        Recursively parse a group element and its subgroups.

        Args:
            group_element (Element): The current group element.
            current_group_id (int): The ID of the current group.
            current_level (int): The level of the current group.
            current_param (str): The parameter file associated with the current group.
        """
        # Get all nodes in this group
        nodes = [int(node.get("id")) for node in group_element.findall("node")]

        # if current_level == 0:
        if current_level == 0:
            for node in nodes:
                group_mapping[node] = {
                    "group_id": current_group_id, 
                    "ports": int(group_element.get("ports", 1))
                    }

        # Recursively parse subgroups
        for subgroup in group_element.findall("group"):
            subgroup_id = int(subgroup.get("id"))
            subgroup_level = int(subgroup.get("level"))
            subgroup_param = get_path(subgroup.get("param_file"))
            files.add(subgroup_param)
            nodes += parse_group(subgroup, subgroup_id, subgroup_level, subgroup_param)

        # Add all pairs of nodes in this group to the dictionary
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_pairs_to_groups[(nodes[i], nodes[j])].append((current_group_id, current_level, current_param))
                node_pairs_to_groups[(nodes[j], nodes[i])].append((current_group_id, current_level, current_param))

        return nodes

    # Start parsing from the top-level groups
    for group in root.findall("group"):
        group_id = int(group.get("id"))
        group_level = int(group.get("level"))
        group_param = get_path(group.get("param_file"))
        files.add(group_param)
        parse_group(group, group_id, group_level, group_param)

    return node_pairs_to_groups, files, group_mapping