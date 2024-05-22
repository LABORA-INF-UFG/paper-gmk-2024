import logging
logging.basicConfig(filename="logs/training.log", level=logging.INFO)
import json
import sys
from read_topology import Topology
from vRAN_env import CustomEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
import time

if __name__ == '__main__':
    number_of_CUs = sys.argv[1]
    number_of_RUs = sys.argv[2]
    topology_type = sys.argv[3]
    training_instance = sys.argv[4]

    logging.info("Start trainning") 
    
    topology_json = json.load(open("topologies/topology_{}_RUs_{}_CUs_{}.json".format(number_of_RUs, number_of_CUs, topology_type), 'r'))

    topo = Topology(n_CUs=len(topology_json["CUs"]), 
                    n_DUs=len(topology_json["DUs"]), 
                    n_RUs=len(topology_json["RUs"]), 
                    CUs=topology_json["CUs"], 
                    DUs=topology_json["DUs"],
                    RUs=topology_json["DUs"],
                    paths=topology_json["paths"])

    logging.info("Creating actions based on # RUs")
    
    env = CustomEnv(topo, training_instance)
    # check_env(env)
    # Define and Train the agent
    start_time = time.time()
    model = A2C("MlpPolicy", env, verbose=1).learn(total_timesteps=8740)
    end_time = time.time()

    # json.dump({"elapsed_time": end_time-start_time}, open("trained_instances/{}/time.json".format(training_instance), 'w'))

    model.save("trained_models/{}".format(training_instance))
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)