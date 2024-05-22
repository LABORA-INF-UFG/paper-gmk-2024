import time
import json
from docplex.mp.model import Model
import os
from read_topology import Topology
import sys
import pandas


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


class CR:
    def __init__(self, id, cpu, num_BS):
        self.id = id
        self.cpu = cpu
        self.num_BS = num_BS

    def __str__(self):
        return "ID: {}\tCPU: {}".format(self.id, self.cpu)


class DRC:
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


class FS:
    def __init__(self, id, f_cpu, f_ram):
        self.id = id
        self.f_cpu = f_cpu
        self.f_ram = f_ram


class RU:
    def __init__(self, id, CR):
        self.id = id
        self.CR = CR

    def __str__(self):
        return "RU: {}\tCR: {}".format(self.id, self.CR)


links = []
capacity = {}
delay = {}
crs = {}
paths = {}
conj_Fs = {}
bw_factor = 1
CPU_factor = 1


def read_topology(q_CRs):
    with open("../topologies/{}_CNs_links.json".format(q_CRs), 'r') as json_file:
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

        # create and store the set of RC's with RAM and CPU in a global list "rcs"-rc[0] is the core network without CR
        with open("../topologies/{}_CNs_nodes.json".format(q_CRs), 'r') as json_file:
            data = json.load(json_file)
            json_nodes = data["nodes"]
            for item in json_nodes:
                split = str(item).rsplit('-', 1)
                CR_id = split[1]
                node = json_nodes[item]
                #node_RAM = node["RAM"]
                node_CPU = node["CPU"]
                cr = CR(int(CR_id), node_CPU, 0)
                crs[int(CR_id)] = cr
        crs[0] = CR(0, 0, 0)

        # create a set of paths that are calculated previously by the algorithm implemented in "path_gen.py"
        with open("../topologies/{}_CNs_paths.json".format(q_CRs), 'r') as json_paths_file:
            #read the json file that contain the set of paths
            json_paths_f = json.load(json_paths_file)
            json_paths = json_paths_f["paths"]

            #for each path calculate the id, source (always the core node) and destination
            for item in json_paths:
                path = json_paths[item]
                path_id = path["id"]
                path_source = path["source"]

                if path_source == "CN":
                    path_source = 0

                path_target = path["target"]
                path_seq = path["seq"]

                #calculate the intermediate paths p1, p2 and p3 (that path's correspond to BH, MH and FH respectively)
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

                #create the path and store at the global dict "paths"
                p = Path(path_id, path_source, path_target, path_seq, list_p1, list_p2, list_p3, delay_p1, delay_p2, delay_p3)
                paths[path_id] = p


def DRC_structure():
    DRC0 = DRC(0, 0, 0, 4.9, 0, 0, 0.01, [0], [0], ['f8', 'f7', 'f6', 'f5', 'f4', 'f3', 'f2', 'f1', 'f0'], 10, 10, 10, 0, 0, 9.9, 7)
    DRC1 = DRC(1, 0, 0.98, 3.92, 0, 0.01, 0.01, [0], ['f8', 'f7'], ['f6', 'f5', 'f4', 'f3', 'f2', 'f1', 'f0'], 10, 10, 10, 0, 9.9, 13.2, 5.034)
    DRC2 = DRC(2, 0, 2.54, 2.354, 0, 0.01, 0.01, [0], ['f8', 'f7', 'f6', 'f5', 'f4', 'f3', 'f2'], ['f1', 'f0'], 10, 10, 0.25, 0, 9.9, 42.6, 0.119)
    DRCs = {0: DRC0, 1: DRC1, 2: DRC2}
    return DRCs

# self, id, cpu_CU, cpu_DU, cpu_RU, ram_CU, ram_DU, ram_RU, Fs_CU, Fs_DU, Fs_RU, delay_BH, delay_MH, delay_FH, bw_BH, bw_MH, bw_FH, cost

def RU_location(q_CRs):
    """
    Read TIM topology files
    :return:
    """
    rus = {}
    count = 1
    with open("../topologies/{}_CNs_nodes.json".format(q_CRs, q_CRs), 'r') as json_file:
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


DRC_f1 = 0
f1_vars = []
f2_vars = []


def run_stage_1(q_CRs, RUs_demand, factor):
    # print("Running Stage - 1")
    # print("-----------------------------------------------------------------------------------------------------------")
    alocation_time_start = time.time()
    # read_topology_T1()
    read_topology(q_CRs)
    DRCs = DRC_structure()
    rus = RU_location(q_CRs)

    for ru in rus:
        found = False
        for p in paths:
            if rus[ru].CR == paths[p].seq[2]:
                found = True
        if not found:
            print('No path to ', rus[ru].CR)
    
    del_p = []
    for p in paths:
        if paths[p].seq[1] not in [0, 1, 2, 3, 4]:
            del_p.append(p)
    for p in del_p:
        del paths[p]

    read_topology_end = time.time()
    F1 = FS('f8', 2, 2)
    F2 = FS('f7', 2, 2)
    F3 = FS('f6', 2, 2)
    F4 = FS('f5', 2, 2)
    F5 = FS('f4', 2, 2)
    F6 = FS('f3', 2, 2)
    F7 = FS('f2', 2, 2)
    conj_Fs = {'f8': F1, 'f7': F2, 'f6': F3, 'f5': F4, 'f4': F5, 'f3': F6, 'f2': F7}
    mdl = Model(name='PlaceRAN Problem', log_output=False)
    i = [(p, d, b) for p in paths for d in DRCs for b in rus if (paths[p].seq[2] == rus[b].CR) and
         ((paths[p].seq[0] == 0 and paths[p].seq[1] == 0 and d in [0]) or (paths[p].seq[0] == 0 and paths[p].seq[1] != 0 and d in [1, 2, 3, 4]))]

    p_i = [b for b in rus]
    c_i = [c for c in crs]

    mdl.x = mdl.binary_var_dict(keys=i, name='x')
    mdl.p3 = mdl.binary_var_dict(keys=p_i, name='p3')
    mdl.c = mdl.binary_var_dict(keys=c_i, name='c')

    
    proc_cost = (mdl.sum(mdl.x[it] * DRCs[it[1]].cost for it in i))

    declined_cost = mdl.sum(mdl.p3[it] for it in p_i)

    second_declined_cost = mdl.sum(mdl.c[it] for it in c_i)

    mdl.minimize(proc_cost + 4.915 * (declined_cost + second_declined_cost))

    for b in rus:
        mdl.add_constraint(mdl.sum(mdl.x[it] for it in i if it[2] == b) == 1, 'unicity')
    
    
    for ru in rus:
        mdl.add_constraint(mdl.sum(mdl.x[it] * paths[it[0]].delay_p3 for it in i if it[2] == ru) <= mdl.sum(mdl.x[it] * DRCs[it[1]].delay_FH for it in i if it[2] == ru) + mdl.p3[ru] * 999)
        mdl.add_constraint(mdl.sum(mdl.x[it] * paths[it[0]].delay_p3 for it in i if it[2] == ru) >= mdl.sum(mdl.x[it] * DRCs[it[1]].delay_FH for it in i if it[2] == ru) - (1 - mdl.p3[ru]) * 999)
    
    mdl.add_constraint(mdl.sum(mdl.x[it] for it in i if paths[it[0]].seq[1] == 1 and it[2] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 43, 44, 45, 46] and it[1] in [2]) == 0)

    mdl.add_constraint(mdl.sum(mdl.x[it] for it in i if paths[it[0]].seq[1] == 2 and it[2] not in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 43, 44, 45, 46] and it[1] in [2]) == 0)

    mdl.add_constraint(mdl.sum(mdl.x[it] for it in i if paths[it[0]].seq[1] == 3 and it[2] not in [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] and it[1] in [2]) == 0)

    mdl.add_constraint(mdl.sum(mdl.x[it] for it in i if paths[it[0]].seq[1] == 4 and it[2] not in [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42] and it[1] in [2]) == 0)
    
    for cn in crs:
        mdl.add_constraint(mdl.sum(mdl.x[it] * DRCs[it[1]].cpu_DU * RUs_demand[it[2]] for it in i if paths[it[0]].seq[1] == cn) <= (crs[cn].cpu * factor) + mdl.c[cn] * 999)
        mdl.add_constraint(mdl.sum(mdl.x[it] * DRCs[it[1]].cpu_DU * RUs_demand[it[2]] for it in i if paths[it[0]].seq[1] == cn) >= (crs[cn].cpu * factor) - (1 - mdl.c[cn]) * 999)
        # mdl.add_constraint(mdl.sum(mdl.x[it] * paths[it[0]].delay_p3 - DRCs[it[1]].delay_FH for it in i if it[2] == ru) <= mdl.p3[ru] * 9999.9)
        # mdl.add_constraint(mdl.sum(mdl.x[it] * paths[it[0]].delay_p3 - DRCs[it[1]].delay_FH for it in i if it[2] == ru) >= -9999.9 * (1 - mdl.p3[ru]))

    # for c in crs:
    #     mdl.add_constraint(
    #         mdl.sum(mdl.x[it] * DRCs[it[1]].cpu_CU for it in i if c == paths[it[0]].seq[0]) +
    #         mdl.sum(mdl.x[it] * DRCs[it[1]].cpu_DU for it in i if c == paths[it[0]].seq[1]) +
    #         mdl.sum(mdl.x[it] * DRCs[it[1]].cpu_RU for it in i if c == paths[it[0]].seq[2]) <= crs[c].cpu, 'crs_cpu_usage')
    mdl.solve()

    disp_Fs = {}

    for cr in crs:
        disp_Fs[cr] = {'f8': 0, 'f7': 0, 'f6': 0, 'f5': 0, 'f4': 0, 'f3': 0, 'f2': 0, 'f1': 0, 'f0': 0}

    for it in i:
        for cr in crs:
            if mdl.x[it].solution_value > 0:
                if cr in paths[it[0]].seq:
                    seq = paths[it[0]].seq
                    if cr == seq[0]:
                        Fs = DRCs[it[1]].Fs_CU
                        for o in Fs:
                            if o != 0:
                                dct = disp_Fs[cr]
                                dct["{}".format(o)] += 1
                                disp_Fs[cr] = dct

                    if cr == seq[1]:
                        Fs = DRCs[it[1]].Fs_DU
                        for o in Fs:
                            if o != 0:
                                dct = disp_Fs[cr]
                                dct["{}".format(o)] += 1
                                disp_Fs[cr] = dct

                    if cr == seq[2]:
                        Fs = DRCs[it[1]].Fs_RU
                        for o in Fs:
                            if o != 0:
                                dct = disp_Fs[cr]
                                dct["{}".format(o)] += 1
                                disp_Fs[cr] = dct

    # for cr in disp_Fs:
    #     print(str(cr) + str(disp_Fs[cr]))

    for it in i:
        if mdl.x[it].solution_value > 0.8:
            pass
            # print("x{} -> {}".format(it, mdl.x[it].solution_value))
            # print(paths[it[0]])
        
    count_constraints = 0
    flag = False
    for it in p_i:
        if mdl.p3[it].solution_value > 0.8:
            count_constraints += 1
            # print("p3 {} -> {}".format(it, mdl.p3[it].solution_value))
            flag = True
    
    for it in c_i:
        if mdl.c[it].solution_value > 0.8:
            count_constraints += 1
            # print("C {} -> {}".format(it, mdl.c[it].solution_value))
            flag = True

    FO = mdl.solution.get_objective_value()
    
    # print("FO: {}".format(1 - FO/(10.5 * 47)))#1 - FO/(47*7)))

    global f1_vars
    for it in i:
        if mdl.x[it].solution_value > 0:
            f1_vars.append(it)

    return {"FO": FO, "flag": flag, "count": count_constraints}


if __name__ == '__main__':
    number_of_CUs = sys.argv[1]
    number_of_RUs = sys.argv[2]
    topology_type = sys.argv[3]

    data = {"episodes": [], "reward": [], "constraints": [], "flag": []}

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
                RUs_demand = [topo.DemandDU("../DRL_agent/demand/new_demand_{}.csv".format(week), DU)[t]/100 for DU in range(0, len(topo.ListDUs())+1)]
                bw_factor = 1
                CPU_factor = 1
                start_all = time.time()
                print("Running episode {}".format(count_episode))
                count_episode += 1
                output = run_stage_1(q_CRs=4, RUs_demand=RUs_demand, factor=1)
                print(output)
                data["episodes"].append(count_episode)
                data["reward"].append(output["FO"])
                data["constraints"].append(output["count"])
                if output["flag"]:
                    data["flag"].append(1)
                else:
                    data["flag"].append(0)
                end_all = time.time()
                print("TOTAL TIME: {}".format(end_all - start_all))

    df = pandas.DataFrame.from_dict(data)
    df.to_csv("../DRL_agent/evaluation_data/Murti_47_RUs_4_CUs_{}.csv".format(topology_type))
