import gym
import numpy as np
from gym import spaces
from read_topology import Topology
import json
from stable_baselines3 import A2C
from heuristic import select_DUs
import time

class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self, topo, training_instance):
        self.n_splits = 3
        self.action_space = spaces.MultiDiscrete(np.array([self.n_splits for i in range(0, topo.n_RUs)]))
        self.observation_space = spaces.Box(low=0, high=100, shape=(topo.n_DUs + topo.n_DUs + topo.n_DUs + 1,), dtype=np.int64)
        self.topo = topo
        self.day = 0
        self.time = 0
        self.reward_history = {}
        self.elapsed_time = []
        self.start_time = time.time()
        self.reward_history_second_agent = {}
        self.reward_proc = {}
        self.reward_migr = {}
        self.split_0 = {}
        self.split_1 = {}
        self.split_2 = {}
        self.count_episodes = -1
        self.training_instance = training_instance
        self.start = True
        self.week = 0
        self.t = 0
        self.current_split = [0 for i in range(0, self.topo.n_DUs)]
        self.previous_split =[0 for i in range(0, self.topo.n_DUs)]
        self.DU_assignment = [0 for i in range(0, self.topo.n_DUs)]

    def step(self, action):
        self.start_time = time.time()
        self.t = time.time()
        cost = 0
        CUs_usage = [0 for i in range(0, self.topo.n_CUs)]
        self.DU_assignment = [self.observation[len(self.observation) - i] for i in range(2, self.topo.n_DUs + 2)]
        CUs_current_capacity = self.observation[len(self.observation) - 1]
        
        # print(proc_cost/(7 * self.topo.n_DUs), (cost - proc_cost)/(7*47*0.05))

        info = {}

        demand_heuristic = [self.observation[vDU] for vDU in range(self.topo.n_DUs)]
        json_second_agent = select_DUs(self.topo, demand_heuristic, self.current_split, CUs_current_capacity)

        self.DU_assignment = json_second_agent["DU_assignment"]

        not_allocated = json_second_agent["not allocated"]
        infeasible_cost = 0
        if not_allocated:
            infeasible_cost = not_allocated * 7
            print("not_allocated")
        
        self.current_split = [action[i] for i in range(0, self.topo.n_DUs)]

        # calculating processing cost and CU processing constraint
        DU_ID = 0
        for a in action:
            if DU_ID in json_second_agent["D-RAN RUs"]:
                self.current_split[DU_ID] = 0
                cost += 7
            elif DU_ID in json_second_agent["O2 RUs"]:
                self.current_split[DU_ID] = 1
                cost += 2 * 0.017 + 5
            elif not a:
                CUs_usage[self.DU_assignment[DU_ID]] += self.observation[DU_ID]/100 * 0
                cost += 7
            elif a == 1:
                CUs_usage[self.DU_assignment[DU_ID]] += self.observation[DU_ID]/100 * 0.98
                cost += 2 * 0.017 + 5
            elif a == 2:
                CUs_usage[self.DU_assignment[DU_ID]] += self.observation[DU_ID]/100 * 2.548
                cost += 7 * 0.017
            DU_ID += 1

        proc_cost = cost
        self.reward_proc[self.day][int(self.count_episodes)].append(proc_cost)
        
        # calculating the deployment (migration) cost
        self.previous_split = [self.observation[self.topo.n_DUs + i] for i in range(0, self.topo.n_DUs)]

        for du in range(0, self.topo.n_DUs):
            if self.previous_split[du] == 0:
                if self.current_split[du] == 1:
                    cost += 2 * 0.5
                elif self.current_split[du] == 2:
                    cost += 7 * 0.5
            if self.previous_split[du] == 1:
                if self.current_split[du] == 0:
                    cost += 2 * 0.5
                elif self.current_split[du] == 2:
                    cost += 5 * 0.5
            if self.previous_split[du] == 2:
                if self.current_split[du] == 0:
                    cost += 7 * 0.5
                elif self.current_split[du] == 1:
                    cost += 5 * 0.5

        self.reward_migr[self.day][int(self.count_episodes)].append(cost - proc_cost)

        migr_cost = cost - proc_cost
        
        # calculating used splits
        DU_ID = 0
        for a in self.current_split:
            if a == 0:
                self.split_0[self.day][int(self.count_episodes)] += 1
            elif a == 1:
                self.split_1[self.day][int(self.count_episodes)] += 1
            elif a == 2:
                self.split_2[self.day][int(self.count_episodes)] += 1
            DU_ID += 1
        
        # Feasible = 7 * self.topo.n_DUs
        Feasible = 1
        
        # for cu in CUs_usage:
        #     if cu > self.topo.CUs[0]["capacity"] * CUs_current_capacity:
        #         Feasible = 0
        #         print("Infeasible")
        #         break

        cost_ratio = (proc_cost + infeasible_cost + migr_cost)/(10.5 * self.topo.n_DUs)

        self.reward = (Feasible - (1 * cost_ratio))

        self.reward_history[self.day][int(self.count_episodes)].append(self.reward)

        self.reward_history_second_agent[self.day][int(self.count_episodes)].append(json_second_agent["reward"])

        self.time += 1

        self.terminated = True

        if False:# self.time == (24 * (self.day + 1)):
            self.terminated = True
            self.day += 1
        else:
            self.observation = [self.topo.DemandDU("demand/new_demand_{}.csv".format(self.week), DU)[self.time] 
                            for DU in range(0, len(self.topo.ListDUs()))] + self.current_split + self.DU_assignment + [CUs_current_capacity]

        self.elapsed_time.append(time.time() - self.start_time)

        return self.observation, self.reward, self.terminated, info

    def reset(self):
        self.terminated = False
        self.count_episodes += 1
        self.t = time.time()

        if self.time == 22:
            print("h:", self.time, "d", self.day, "w", self.week)
        
        if self.time > 23:
            self.day += 1
            self.time = 0
        
        if self.day > 6:
            self.week += 1
            self.day = 0
            # self.time = 0
        
        if self.week in [4 * i + j for i in range(1, 12, 2) for j in range(1, 5)]:
            CUs_current_capacity = 0.5
        else:
            CUs_current_capacity = 1
        
        if self.week > 50:
            print("STARTING AGAIN", self.week)
            self.week = 0
        
        self.observation = [self.topo.DemandDU("demand/new_demand_{}.csv".format(self.week), DU)[self.time] for DU in range(0, len(self.topo.ListDUs()))] + \
        self.current_split + self.DU_assignment + [CUs_current_capacity]

        if self.day not in self.reward_history.keys():
            self.reward_history[self.day] = {}
            self.reward_history_second_agent[self.day] = {}
            self.reward_proc[self.day] = {}
            self.reward_migr[self.day] = {}
            self.split_0[self.day] = {}
            self.split_1[self.day] = {}
            self.split_2[self.day] = {}
        if self.time > 22:
            json.dump(self.reward_history, open("trained_instances/{}/all_reward_history.json".format(self.training_instance), 'w'))
            json.dump(self.reward_history_second_agent, open("trained_instances/{}/second_agent.json".format(self.training_instance), 'w'))
            json.dump(self.reward_proc, open("trained_instances/{}/proc_cost.json".format(self.training_instance), 'w'))
            json.dump(self.reward_migr, open("trained_instances/{}/migr_cost.json".format(self.training_instance), 'w'))
            json.dump(self.split_0, open("trained_instances/{}/split_0.json".format(self.training_instance), 'w'))
            json.dump(self.split_1, open("trained_instances/{}/split_1.json".format(self.training_instance), 'w'))
            json.dump(self.split_2, open("trained_instances/{}/split_2.json".format(self.training_instance), 'w'))
            json.dump({"elapsed_time": self.elapsed_time}, open("trained_instances/{}/time.json".format(self.training_instance), 'w'))
        
        self.reward_history[self.day][int(self.count_episodes)] = []
        self.reward_history_second_agent[self.day][int(self.count_episodes)] = []
        self.reward_proc[self.day][int(self.count_episodes)] = []
        self.reward_migr[self.day][int(self.count_episodes)] = []
        self.split_0[self.day][int(self.count_episodes)] = 0
        self.split_1[self.day][int(self.count_episodes)] = 0
        self.split_2[self.day][int(self.count_episodes)] = 0

        return self.observation


    def evaluate(self, obs, week):
        self.terminated = False
        self.count_episodes += 1
        self.t = time.time()
        self.observation = obs
        self.week = week

        if self.day not in self.reward_history.keys():
            self.reward_history[self.day] = {}
            self.reward_history_second_agent[self.day] = {}
            self.reward_proc[self.day] = {}
            self.reward_migr[self.day] = {}
            self.split_0[self.day] = {}
            self.split_1[self.day] = {}
            self.split_2[self.day] = {}
        
        self.reward_history[self.day][int(self.count_episodes)] = []
        self.reward_history_second_agent[self.day][int(self.count_episodes)] = []
        self.reward_proc[self.day][int(self.count_episodes)] = []
        self.reward_migr[self.day][int(self.count_episodes)] = []
        self.split_0[self.day][int(self.count_episodes)] = 0
        self.split_1[self.day][int(self.count_episodes)] = 0
        self.split_2[self.day][int(self.count_episodes)] = 0

        return self.observation
