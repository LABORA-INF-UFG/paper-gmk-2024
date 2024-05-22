import time

def select_DUs(topo, demand, current_split, CUs_current_capacity):

    midhaul_capacity = 30000

    DUs_dict = {}
    DU_count = 0
    CU_current_capacity = [topo.CUs[0]["capacity"] * CUs_current_capacity for cu in range(0, topo.n_CUs)]
    midhaul_links = {}
    paths_latency = {}
    DRAN_RUs = []
    O2_RUs = []
    for du in range(0, topo.n_DUs):
        for p in topo.paths:
            if p["DU_ID"] == du:
                paths_latency[du] = p["latency"]
                break

    for cu in range(0, topo.n_CUs):
        midhaul_links[cu] = [midhaul_capacity, midhaul_capacity]

    DU_assignment = []

    for i in demand:
        DUs_dict[DU_count] = i
        DU_count += 1
    
    sorted_DUs = {k: v for k, v in sorted(DUs_dict.items(), key=lambda item: item[1])}

    for du in range(0, topo.n_DUs):
        DU_assignment.append(-1)
    
    not_allocated = 0

    for du in sorted_DUs:
        assigned = False
        for CU_ID in range(0, topo.n_CUs):
            if current_split[du] == 0:
                if CU_current_capacity[CU_ID] - demand[du]/100 >= 0:
                    if midhaul_links[CU_ID][0] - demand[du]/100 * 1 >= 0 and paths_latency[du][CU_ID] <= 1000:
                        CU_current_capacity[CU_ID] -= demand[du]/100
                        midhaul_links[CU_ID][0] -= demand[du]/100 * 1
                        DU_assignment[du] = CU_ID
                        assigned = True
                        break
                    elif midhaul_links[CU_ID][1] - demand[du]/100 * 1 >= 0 and paths_latency[du][CU_ID] <= 1000:
                        CU_current_capacity[CU_ID] -= demand[du]/100
                        midhaul_links[CU_ID][1] -= demand[du]/100 * 1
                        DU_assignment[du] = CU_ID
                        assigned = True
                        break
            elif current_split[du] == 1:
                if CU_current_capacity[CU_ID] - (0.98 * demand[du]/100) >= 0:
                    if midhaul_links[CU_ID][0] - demand[du]/100 * 4 >= 0 and paths_latency[du][CU_ID] <= 10:
                        CU_current_capacity[CU_ID] -= 0.98 * demand[du]/100
                        midhaul_links[CU_ID][0] -= demand[du]/100 * 4
                        DU_assignment[du] = CU_ID
                        assigned = True
                        break
                    elif midhaul_links[CU_ID][1] - demand[du]/100 * 4 >= 0 and paths_latency[du][CU_ID] <= 10:
                        CU_current_capacity[CU_ID] -= 0.98 * demand[du]/100
                        midhaul_links[CU_ID][1] -= demand[du]/100 * 4
                        DU_assignment[du] = CU_ID
                        assigned = True
                        break
            elif current_split[du] == 2:
                if CU_current_capacity[CU_ID] - (2.548 * demand[du]/100) >= 0:
                    if midhaul_links[CU_ID][0] - demand[du]/100 * 86.1 >= 0 and paths_latency[du][CU_ID] <= 0.25:
                        CU_current_capacity[CU_ID] -= 2.548 * demand[du]/100
                        midhaul_links[CU_ID][0] -= demand[du]/100 * 86.1
                        DU_assignment[du] = CU_ID
                        assigned = True
                        break
                    elif midhaul_links[CU_ID][1] - demand[du]/100 * 86.1 >= 0 and paths_latency[du][CU_ID] <= 0.25:
                        CU_current_capacity[CU_ID] -= 2.548 * demand[du]/100
                        midhaul_links[CU_ID][1] -= demand[du]/100 * 86.1
                        DU_assignment[du] = CU_ID
                        assigned = True
                        break
        if not assigned:
            for CU_ID in range(0, topo.n_CUs):
                if CU_current_capacity[CU_ID] - (0.98 * demand[du]/100) >= 0:
                    if midhaul_links[CU_ID][0] - demand[du]/100 * 4 >= 0 and paths_latency[du][CU_ID] <= 10:
                        CU_current_capacity[CU_ID] -= 0.98 * demand[du]/100
                        midhaul_links[CU_ID][0] -= demand[du]/100 * 4
                        DU_assignment[du] = CU_ID
                        assigned = True
                        O2_RUs.append(du)
                        break
                    elif midhaul_links[CU_ID][1] - demand[du]/100 * 4 >= 0 and paths_latency[du][CU_ID] <= 10:
                        CU_current_capacity[CU_ID] -= 0.98 * demand[du]/100
                        midhaul_links[CU_ID][1] -= demand[du]/100 * 4
                        DU_assignment[du] = CU_ID
                        assigned = True
                        O2_RUs.append(du)
                        break
        if not assigned:
            DRAN_RUs.append(du)
            not_allocated += 1
    
    # DU_assignment = [random.randint(0, 1) for i in range(0, 4)]

    # print(DU_assignment)

    
    # print(DU_assignment)
    return {"DU_assignment": DU_assignment, "reward": len(set(DU_assignment)), "not allocated": not_allocated, "D-RAN RUs": DRAN_RUs, "O2 RUs":O2_RUs}