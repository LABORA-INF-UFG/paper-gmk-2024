import json
import random
import sys

def CUs_generator(n_CUs):
    CUs = []
    for id in range(0, n_CUs):
        CUs.append({
            "ID": id,
            "node": id + 1,
            "capacity": 50
        })
    
    return CUs

def DUs_generator(n_CUs, n_RUs):
    DUs = []
    for id in range(0, n_RUs):
        DUs.append({
            "ID": id,
            "node": n_CUs + 1 + id,
            "capacity": 10
        })
    
    return DUs


def RUs_generator(n_CUs, n_RUs):
    RUs = []
    for id in range(0, n_RUs):
        RUs.append({
            "ID": id,
            "DU_ID": id,
            "node": n_CUs + 1 + id,
            "capacity": 10
        })
    
    return RUs


def paths_generator(n_CUs, n_RUs):
    paths = []
    count_paths = {}
    CUs_links = {}
    CUs_links[0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 43, 44, 45, 46]
    CUs_links[1] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 43, 44, 45, 46]
    CUs_links[2] = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    CUs_links[3] = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]

    for du_id in range(0, n_RUs):
        count_paths[du_id] = 0
        latency = []
        for cu in range(n_CUs):
            if du_id in CUs_links[cu]:
                latency.append(0.25)
            else:
                latency.append(1)
        paths.append({
            "DU_ID": du_id,
            "latency": latency
        })
    
    return paths


if __name__ == '__main__':
    n_CUs = int(sys.argv[1])
    n_RUs = int(sys.argv[2])
    CUs = CUs_generator(n_CUs)
    DUs = DUs_generator(n_CUs, n_RUs)
    RUs = RUs_generator(n_CUs, n_RUs)
    paths = paths_generator(n_CUs, n_RUs)

    json.dump({"CUs": CUs, "DUs": DUs, "RUs": RUs, "paths": paths}, open("topology_{}_RUs_{}_CUs.json".format(n_RUs, n_CUs), 'w'))

