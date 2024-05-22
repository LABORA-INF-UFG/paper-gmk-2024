import json
import pandas as pd

class Topology():
    def __init__(self, n_CUs, n_DUs, n_RUs, paths, CUs, DUs, RUs):
        self.n_CUs = n_CUs
        self.n_DUs = n_DUs
        self.n_RUs = n_RUs
        self.CUs = CUs
        self.DUs = DUs
        self.RUs = RUs
        self.paths = paths
    
    def ListPaths(self):
        return self.paths
    
    def ListCUs(self):
        return self.CUs
    
    def ListDUs(self):
        return self.DUs
    
    def LinkCapacity(self, l):
        return self.links[l]
    
    def CapacityCU(self, c):
        return self.CUs[c]["capacity"]

    def CapacityDU(self, d):
        return self.DUs[d]["capacity"]
    
    def DemandDU(self, file, DU):
        self.file = file
        self.df = pd.read_csv(file)
        
        return self.df[self.df['bs']==DU]['users'].tolist()