import sys
import argparse

from coll.allgather import AllGatherGenerator
from coll.reduce import ReduceGenerator
from coll.bcast import BroadcastGenerator
from coll.reduce_scatter import ReduceScatterGenerator
from coll.allreduce import AllReduceGenerator





def main():
    parser = argparse.ArgumentParser(description="Generate collective communication schedules.")
    parser.add_argument("--collective", type=str, required=True, help="Collective operation (e.g., allgather, bcast).")
    parser.add_argument("--num_nodes", type=int, required=True, help="Number of nodes in the collective.")
    parser.add_argument("--buffer_size", type=int, required=True, help="Buffer size for each transfer.")
    parser.add_argument("--algorithm", type=str, default="ring", help="Algorithm to use for scheduling (default: ring).")
    parser.add_argument("--k", type=int, default=1, help="K value for k-ring algorithm (default: 1).")
    parser.add_argument("--output", type=str, required=True, help="Output file to save the generated schedule.")
    
    args = parser.parse_args()
    config = {"k": args.k}
    
    if args.collective == "allgather":
        generator = AllGatherGenerator(args.num_nodes, args.buffer_size, args.algorithm, config)
    elif args.collective == "allreduce":
        generator = AllReduceGenerator(args.num_nodes, args.buffer_size, args.algorithm, config)
    elif args.collective == "reduce":
        generator = ReduceGenerator(args.num_nodes, args.buffer_size, args.algorithm, config)
    elif args.collective == "bcast":
        generator = BroadcastGenerator(args.num_nodes, args.buffer_size, args.algorithm, config)
    elif args.collective == "reduce_scatter":
        generator = ReduceScatterGenerator(args.num_nodes, args.buffer_size, args.algorithm, config)
    else:
        raise ValueError(f"Unsupported collective operation: {args.collective}")
    
    schedule = generator.generate()
    
    with open(args.output, "w") as f:
        f.write(schedule)
    
    print(f"Schedule generated and saved to {args.output}")





if __name__ == "__main__":
    sys.exit(main())