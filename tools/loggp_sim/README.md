## LogGP Simulator
A discrete event simulation of network communication, modeled using the LogGP performance model.

To use this simulator, the user must specify 1. A parameter CSV file generated from the `loggp_param` tool and 2. A schedule XML file, an example parameter and scheudle file can be found in `profiles/osu_loggp_intra.csv` and `schedules/8_ring_allgather.xml` respectively. 

To run the simulation, run `sim.py` with the following arguments
```
python ./sim.py --schedule=<PATH_TO_SCHEDULE> \
                --topo=<PATH_TO_TOPOLOGY> \
                --radix=<NUMBER_OF_PORTS>
```

An example broadcast collective schedule would resemble the following. Note that the `deps` field reference a prior transfer `id`.
```
<collective name="broadcast" algorithm="direct" k="2" nodes="8" buffer_size="4097">

    <transfer id="0" size="4097" src="0" dst="1" deps="-1"/>
    <transfer id="1" size="4097" src="0" dst="2" deps="-1"/>
    <transfer id="2" size="4097" src="0" dst="3" deps="-1"/>
    <transfer id="3" size="4097" src="0" dst="4" deps="-1"/>
    <transfer id="4" size="4097" src="0" dst="5" deps="-1"/>
    <transfer id="5" size="4097" src="0" dst="6" deps="-1"/>
    <transfer id="6" size="4097" src="0" dst="7" deps="-1"/>

</collective>
```

An example topology groups subsets of nodes into different groups at different levels. Each group would have a set of LogGP parameters assocaited with it. The following is a topology of 8 nodes separated into 4 groups of 2 nodes each.
```
<topo num_nodes="8" num_levels="2">
    <group level="1" id="0" param_file="./profiles/nccl_l1_simple_loggp_mock.csv">

        <group level="0" id="0" param_file="./profiles/nccl_l0_simple_loggp.csv">
            <node id="0"/>
            <node id="1"/>
        </group>

        <group level="0" id="1" param_file="./profiles/nccl_l0_simple_loggp.csv">
            <node id="2"/>
            <node id="3"/>
        </group>

        <group level="0" id="2" param_file="./profiles/nccl_l0_simple_loggp.csv">
            <node id="4"/>
            <node id="5"/>
        </group>

        <group level="0" id="3" param_file="./profiles/nccl_l0_simple_loggp.csv">
            <node id="6"/>
            <node id="7"/>
        </group>
        
    </group>
</topo>
```