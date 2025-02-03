import time
import json
from docplex.mp.model import Model
import os
import pandas
import sys
from read_topology import Topology

# Class for Paths, each path has an ID, a source, a target, a sequence of nodes, and the subpaths for backhaul (p1), midhaul (p2) and fronthaul (p3)
class Path:
    def __init__(self, id, source, target, seq, p1, p2, p3, delay_p1, delay_p2, delay_p3):
        self.id = id
        self.source = source
        self.target = target
        self.seq = seq
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.delay_p1 = delay_p1
        self.delay_p2 = delay_p2
        self.delay_p3 = delay_p3

    def __str__(self):
        return "ID: {}\tSEQ: {}\t P1: {}\t P2: {}\t P3: {}\t dP1: {}\t dP2: {}\t dP3: {}".format(self.id, self.seq, self.p1, self.p2, self.p3, self.delay_p1, self.delay_p2, self.delay_p3)


# Class for Computing Nodes (CNs), each CN has an ID, a CPU and the number of BS attached to it
class CN:
    def __init__(self, id, cpu, num_BS):
        self.id = id
        self.cpu = cpu
        self.num_BS = num_BS

    def __str__(self):
        return "ID: {}\tCPU: {}".format(self.id, self.cpu)


# Class for the functional splits defined as Viable NG-RAN Configurations (VNCs), each VNC has an ID, CPU demand, RAM demand, a list of functions running at CU (Fs_CU), DU (Fs_DU) and RU (Fs_RU), and a cost
class VNC:
    def __init__(self, id, cpu_CU, cpu_DU, cpu_RU, ram_CU, ram_DU, ram_RU, Fs_CU, Fs_DU, Fs_RU, delay_BH, delay_MH,
                 delay_FH, bw_BH, bw_MH, bw_FH, cost):
        self.id = id

        self.cpu_CU = cpu_CU
        self.ram_CU = ram_CU
        self.Fs_CU = Fs_CU

        self.cpu_DU = cpu_DU
        self.ram_DU = ram_DU
        self.Fs_DU = Fs_DU

        self.cpu_RU = cpu_RU
        self.ram_RU = ram_RU
        self.Fs_RU = Fs_RU

        self.delay_BH = delay_BH
        self.delay_MH = delay_MH
        self.delay_FH = delay_FH

        self.bw_BH = bw_BH
        self.bw_MH = bw_MH
        self.bw_FH = bw_FH

        self.cost = cost


# Class for Radio Units (RUs), each RU has an ID and the CN to which it is attached
class RU:
    def __init__(self, id, CN):
        self.id = id
        self.CN = CN

    def __str__(self):
        return "RU: {}\tCR: {}".format(self.id, self.CN)

# list of links in the topology
links = []
# capacity of each link in the topologyy
capacity = {}
# delay of each link in the topology
delay = {}
# set of CNs int he topology
cns = {}
# set of paths in the topology
paths = {}


# Method to read topology file
def read_topology(number_of_CUs, topology_type):
    # reading file with the topology links
    with open("../topologies/{}_CNs_links.json".format(number_of_CUs), 'r') as json_file:
        data = json.load(json_file)

        # create a set of links with delay and capacity read by the json file, stored in a global list "links"
        json_links = data["links"]
        for item in json_links:
            link = json_links[item]
            source = link["source"]
            destination = link["destination"]

            split = str(source["node"]).rsplit('N', 1)
            if source["node"] != "CN":
                source_node = int(split[1])
            else:
                source_node = 0

            split = str(destination["node"]).rsplit('N', 1)
            if destination["node"] != "CN":
                destination_node = int(split[1])
            else:
                destination_node = 0

            # create links full duplex for each link readed by the json file
            capacity[(source_node, destination_node)] = int(str(link["linkCapacity"]))
            delay[(source_node, destination_node)] = float(str(link["LinkDelay"]).replace(',', '.'))
            if (source_node, destination_node) != '':
                links.append((source_node, destination_node))

            # creating links full duplex for each link readed by the json file
            capacity[(destination_node, source_node)] = int(str(link["linkCapacity"]))
            delay[(destination_node, source_node)] = float(str(link["LinkDelay"]).replace(',', '.'))
            if (destination_node, source_node) != '':
                links.append((destination_node, source_node))

        # create and store the set of CNs with RAM and CPU in a global list
        with open("../topologies/{}_CNs_nodes_{}.json".format(number_of_CUs, topology_type), 'r') as json_file:
            data = json.load(json_file)
            json_nodes = data["nodes"]
            for item in json_nodes:
                split = str(item).rsplit('-', 1)
                CR_id = split[1]
                node = json_nodes[item]
                # for this work we are not using RAM since we don't have any data about it
                # node_RAM = node["RAM"]
                node_CPU = node["CPU"]
                cn = CN(int(CR_id), node_CPU, 0)
                cns[int(CR_id)] = cn
        cns[0] = CN(0, 0, 0)

        # create a set of paths that are calculated previously
        with open("../topologies/{}_CNs_paths.json".format(number_of_CUs), 'r') as json_paths_file:
            #read the json file that contain the set of paths
            json_paths_f = json.load(json_paths_file)
            json_paths = json_paths_f["paths"]

            # for each path calculate the id, source (always the core node) and destination
            for item in json_paths:
                path = json_paths[item]
                path_id = path["id"]
                path_source = path["source"]

                if path_source == "CN":
                    path_source = 0

                path_target = path["target"]
                path_seq = path["seq"]

                # calculate the intermediate paths p1, p2 and p3 (that path's correspond to BH, MH and FH respectively)
                paths_p = [path["p1"], path["p2"], path["p3"]]

                list_p1 = []
                list_p2 = []
                list_p3 = []

                for path_p in paths_p:
                    aux = ""
                    sum_delay = 0

                    for tup in path_p:
                        aux += tup
                        tup_aux = tup
                        tup_aux = tup_aux.replace('(', '')
                        tup_aux = tup_aux.replace(')', '')
                        tup_aux = tuple(map(int, tup_aux.split(', ')))
                        if path_p == path["p1"]:
                            list_p1.append(tup_aux)
                        elif path_p == path["p2"]:
                            list_p2.append(tup_aux)
                        elif path_p == path["p3"]:
                            list_p3.append(tup_aux)
                        sum_delay += float(str(delay[tup_aux]).replace(',', '.'))

                    if path_p == path["p1"]:
                        delay_p1 = sum_delay
                    elif path_p == path["p2"]:
                        delay_p2 = sum_delay
                    elif path_p == path["p3"]:
                        delay_p3 = sum_delay

                    if path_seq[0] == 0:
                        delay_p1 = 0

                    if path_seq[1] == 0:
                        delay_p2 = 0

                # create the path and store at the global dict "paths"
                p = Path(path_id, path_source, path_target, path_seq, list_p1, list_p2, list_p3, delay_p1, delay_p2, delay_p3)
                paths[path_id] = p


# Method tha defines VNCs structure, the cost for each VNC is from https://arxiv.org/abs/2305.17321
# id, cpu_CU, cpu_DU, cpu_RU, ram_CU, ram_DU, ram_RU, Fs_CU, Fs_DU, Fs_RU, delay_BH, delay_MH, delay_FH, bw_BH, bw_MH, bw_FH, cost):
def VNC_structure():
    VNC0 = VNC(0, 0, 0, 0, 0, 0, 0.01, [0], [0], ['f8', 'f7', 'f6', 'f5', 'f4', 'f3', 'f2', 'f1', 'f0'], 0, 0, 10, 0, 0, 9.9, 7)
    VNC1 = VNC(1, 0, 0.98, 0, 0, 0.01, 0.01, [0], ['f8', 'f7'], ['f6', 'f5', 'f4', 'f3', 'f2', 'f1', 'f0'], 0, 10, 10, 0, 9.9, 13.2, 5.034)
    VNC2 = VNC(2, 0, 2.54, 0, 0, 0.01, 0.01, [0], ['f8', 'f7', 'f6', 'f5', 'f4', 'f3', 'f2'], ['f1', 'f0'], 0, 10, 0.25, 0, 9.9, 42.6, 0.119)
    VNCs = {0: VNC0, 1: VNC1, 2: VNC2}
    return VNCs


# Method that defines the RUs location in the topology
def RU_location(number_of_CUs, topology_type):
    rus = {}
    count = 1
    with open("../topologies/{}_CNs_nodes_{}.json".format(number_of_CUs, topology_type), 'r') as json_file:
        data = json.load(json_file)

        json_rcs = data["nodes"]

        for item in json_rcs:
            node = json_rcs[item]
            num_rus = node["RU"]
            num_rc = str(item).split('-', 1)[1]

            for i in range(0, num_rus):
                rus[count] = RU(count, int(num_rc))
                count += 1

    return rus


# Method that defines the optimization model
def run_model(number_of_CUs, RUs_demand, topology_type):
    print("-----------------------------------------------------------------------------------------------------------")
    print("Running Optimal Model")
    print("-----------------------------------------------------------------------------------------------------------")

    read_topology(number_of_CUs, topology_type)
    VNCs = VNC_structure()
    rus = RU_location(number_of_CUs, topology_type)
    
    del_p = []
    for p in paths:
        # this is to remove paths that do not place the CU at the nodes from 0 to 4. This is specific to the topology used in this work
        if paths[p].seq[1] not in [0, 1, 2]:
            del_p.append(p)
    for p in del_p:
        del paths[p]
    
    mdl = Model(name='PlaceRAN Problem', log_output=False)

    for p in paths:
        print(paths[p].seq, paths[p].delay_p1, paths[p].delay_p2, paths[p].delay_p3)


    # Create the decision variable index lists
    i = [(p, d, b) for p in paths for d in VNCs for b in rus if (paths[p].seq[2] == rus[b].CN) and
         ((paths[p].seq[0] != 0) or
          (paths[p].seq[0] == 0 and paths[p].seq[1] != 0 and d in [1, 2]) or
          (paths[p].seq[1] == 0 and d in [0])) and
         (paths[p].delay_p1 <= VNCs[d].delay_BH) and
         (paths[p].delay_p2 <= VNCs[d].delay_MH) and
         (paths[p].delay_p3 <= VNCs[d].delay_FH)]
    j = [(c, f) for f in ["f8", "f7", "f6", "f5", "f4", "f3", "f2"] for c in cns.keys()]  
    l = [c for c in cns.keys() if c != 0]

    # creating the decision variables
    mdl.x = mdl.binary_var_dict(keys=i, name='x')
    mdl.w = mdl.binary_var_dict(keys=l, name='w')
    mdl.z = mdl.binary_var_dict(keys=j, name='z')

    # This is just to linearize the multiplication of binary variables
    for c in cns:
        if (cns[c].id == 0):
            continue

        max_value = sum(1 for it in i if c in paths[it[0]].seq)
        max_value = max_value + 1

        mdl.add_constraint(mdl.w[cns[c].id] <= mdl.sum(mdl.x[it] for it in i if c in paths[it[0]].seq) / max_value + 0.99999999999)
        mdl.add_constraint(mdl.w[cns[c].id] >= mdl.sum(mdl.x[it] for it in i if c in paths[it[0]].seq) / max_value)
    
    # Calculte the cost of the solution based on the selected VNCs
    proc_cost = (mdl.sum(mdl.x[it] * VNCs[it[1]].cost for it in i))

    # define the objective function as minimize VNCs cost
    mdl.minimize(proc_cost)

    # Constraint ensuring that each RU has its VNC selected
    for b in rus:
        mdl.add_constraint(mdl.sum(mdl.x[it] for it in i if it[2] == b) == 1, 'unicity')

    # # Constraint ensuring that each link has its capacity respected
    # capacity_expressions = {}
    # for it in i:
    #     for link in paths[it[0]].p1:
    #         capacity_expressions.setdefault(link, mdl.linear_expr()).add_term(mdl.x[it], VNCs[it[1]].bw_BH)

    #     for link in paths[it[0]].p2:
    #         capacity_expressions.setdefault(link, mdl.linear_expr()).add_term(mdl.x[it], VNCs[it[1]].bw_MH)

    #     for link in paths[it[0]].p3:
    #         capacity_expressions.setdefault(link, mdl.linear_expr()).add_term(mdl.x[it], VNCs[it[1]].bw_FH)

    # # Set the capacity constraints
    # for l in links:
    #     if l in capacity_expressions.keys():
    #         mdl.add_constraint(capacity_expressions[l] <= capacity[l], 'links_bw')

    # # Set delay constraints for VNCs based on the path selected
    # for it in i:
    #     mdl.add_constraint((mdl.x[it] * paths[it[0]].delay_p1) <= VNCs[it[1]].delay_BH, 'delay_req_p1')
    # for it in i:
    #     mdl.add_constraint((mdl.x[it] * paths[it[0]].delay_p2) <= VNCs[it[1]].delay_MH, 'delay_req_p2')
    # for it in i:
    #     mdl.add_constraint((mdl.x[it] * paths[it[0]].delay_p3 <= VNCs[it[1]].delay_FH), 'delay_req_p3')

    # # Set the CPU constraints for each CN - In this work we used only CPU in the experiments since data of RAM usage was not available
    # for c in cns:
    #     mdl.add_constraint(mdl.sum(mdl.x[it] * VNCs[it[1]].cpu_CU * RUs_demand[it[2]] for it in i if c == paths[it[0]].seq[0]) +
    #         mdl.sum(mdl.x[it] * VNCs[it[1]].cpu_DU * RUs_demand[it[2]] for it in i if c == paths[it[0]].seq[1]) +
    #         mdl.sum(mdl.x[it] * VNCs[it[1]].cpu_RU * RUs_demand[it[2]] for it in i if c == paths[it[0]].seq[2]) <= cns[c].cpu, 'crs_cpu_usage')
    
    mdl.solve()

    FO = mdl.solution.get_objective_value()

    return FO


if __name__ == '__main__':
    number_of_CUs = sys.argv[1]
    topology_type = sys.argv[2]

    number_of_RUs = 47

    data = {"episodes": [], "reward": []}

    topology_json = json.load(open("../DRL_agent/topologies/topology_{}_RUs_{}_CUs_{}.json".format(number_of_RUs, number_of_CUs, topology_type), 'r'))

    topo = Topology(n_CUs=len(topology_json["CUs"]), 
                    n_DUs=len(topology_json["DUs"]), 
                    n_RUs=len(topology_json["RUs"]), 
                    CUs=topology_json["CUs"], 
                    DUs=topology_json["DUs"],
                    RUs=topology_json["DUs"],
                    paths=topology_json["paths"])
    for week in range(51, 52):
        count_episode = 0
        for day in range(0, 7):
            for t in range(0, 24):
                RUs_demand = [topo.DemandDU("../DRL_agent/demand/new_demand_51.csv", DU)[t]/100 for DU in range(0, len(topo.ListDUs())+1)]
                start_all = time.time()
                print("Running instance {}".format(count_episode))
                count_episode += 1
                output = run_model(number_of_CUs, RUs_demand, topology_type)
                print("SOLUTION OPTIMAL COST", output)
                data["episodes"].append(count_episode)
                data["reward"].append(output)
                end_all = time.time()
                print("TOTAL TIME: {}".format(end_all - start_all))
                print("-----------------------------------------------------------------------------------------------------------")
                print("-----------------------------------------------------------------------------------------------------------")
    df = pandas.DataFrame.from_dict(data)
    df.to_csv("../DRL_agent/evaluation_data/Optimal_47_RUs_{}_CUs_{}.csv".format(number_of_CUs, topology_type))