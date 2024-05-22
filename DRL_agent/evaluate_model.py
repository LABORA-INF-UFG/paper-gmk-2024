import logging
logging.basicConfig(filename="logs/training.log", level=logging.INFO)
import json
import sys
from read_topology import Topology
from vRAN_env import CustomEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
import pandas
import matplotlib.pyplot as plt

if __name__ == '__main__':
    number_of_CUs = sys.argv[1]
    number_of_RUs = sys.argv[2]
    topology_type = sys.argv[3]
    training_instance = sys.argv[4]
    
    topology_json = json.load(open("topologies/topology_{}_RUs_{}_CUs_{}.json".format(number_of_RUs, number_of_CUs, topology_type), 'r'))

    topo = Topology(n_CUs=len(topology_json["CUs"]), 
                    n_DUs=len(topology_json["DUs"]), 
                    n_RUs=len(topology_json["RUs"]), 
                    CUs=topology_json["CUs"], 
                    DUs=topology_json["DUs"],
                    RUs=topology_json["DUs"],
                    paths=topology_json["paths"])
    
    env = CustomEnv(topo, training_instance)
    # check_env(env)
    # Define and Train the agent
    # model = A2C.load("trained_models/{}".format(training_instance))
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)
    model = A2C.load("trained_models/{}".format(training_instance))

    time = 0
    week = 51
    day = 0
    current_split = [0 for i in range(0, topo.n_DUs)]
    DU_assignment = [0 for i in range(0, topo.n_DUs)]
    CUs_current_capacity = 1

    obs = env.evaluate([topo.DemandDU("demand/new_demand_51.csv", DU)[time] for DU in range(0, len(topo.ListDUs()))] + \
        current_split + DU_assignment + [CUs_current_capacity], week)

    print([topo.DemandDU("demand/new_demand_51.csv", DU)[time] for DU in range(0, len(topo.ListDUs()))])

    history = {"episodes": [], "reward": []}
    count_episode = 0

    count = 0

    while day < 8:
        print(count)
        count += 1
        end_ep = False
        if count_episode > 0:
            history["episodes"].append(count_episode)
        while not end_ep:
            # print(obs)
            ac, _ = model.predict(obs, deterministic=True)
            obs, rw, end_ep, _ = env.step(ac)
            if count_episode > 0:
                history["reward"].append((1 - rw) * 10.5 * int(number_of_RUs))
            # print("Action {} - rw = {}".format(ac, rw))
        FO1 = rw
        print("FO = {}".format(FO1))
        count_episode += 1

        if time < 22:
            time += 1
        else:
            time = 0
            day += 1
    df = pandas.DataFrame.from_dict(history)
    
    print(df.head)
    df.to_csv("evaluation_data/{}_evaluation_costs.csv".format(training_instance), index=False)
