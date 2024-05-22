import pandas
import matplotlib.pyplot as plt

CU1_avg_demand = {}
CU2_avg_demand = {}
CU3_avg_demand = {}
CU4_avg_demand = {}

CU1_data = {"time_hour": [], "demand": []}
CU2_data = {"time_hour": [], "demand": []}
CU3_data = {"time_hour": [], "demand": []}
CU4_data = {"time_hour": [], "demand": []}

for i in range(0, 51):
    df = pandas.read_csv("demand/new_demand_{}.csv".format(i))

    CU1_flag = False
    CU2_flag = False
    CU3_flag = False
    CU4_flag = False

    CU1_BSs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 43, 44, 45, 46]
    CU2_BSs = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 43, 44, 45, 46]
    CU3_BSs = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    CU4_BSs = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]

    for bs in CU1_BSs:
        tmp_df = df.loc[df["bs"] == bs]
        if not CU1_flag:
            new_df_CU1 = tmp_df
            CU1_flag = True
        else:
            new_df_CU1 = pandas.concat([new_df_CU1, tmp_df])
    
    for bs in CU2_BSs:
        tmp_df = df.loc[df["bs"] == bs]
        if not CU2_flag:
            new_df_CU2 = tmp_df
            CU2_flag = True
        else:
            new_df_CU2 = pandas.concat([new_df_CU2, tmp_df])
    
    for bs in CU3_BSs:
        tmp_df = df.loc[df["bs"] == bs]
        if not CU3_flag:
            new_df_CU3 = tmp_df
            CU3_flag = True
        else:
            new_df_CU3 = pandas.concat([new_df_CU3, tmp_df])
    
    for bs in CU4_BSs:
        tmp_df = df.loc[df["bs"] == bs]
        if not CU4_flag:
            new_df_CU4 = tmp_df
            CU4_flag = True
        else:
            new_df_CU4 = pandas.concat([new_df_CU4, tmp_df])

    for time in new_df_CU1["time_hour"].unique():
        tmp_df = new_df_CU1.loc[new_df_CU1["time_hour"] == time]
        if time not in CU1_avg_demand.keys():
            CU1_avg_demand[time] = []
        CU1_avg_demand[time].append(tmp_df["users"].sum())
    
    for time in new_df_CU2["time_hour"].unique():
        tmp_df = new_df_CU2.loc[new_df_CU2["time_hour"] == time]
        if time not in CU2_avg_demand.keys():
            CU2_avg_demand[time] = []
        CU2_avg_demand[time].append(tmp_df["users"].sum())
    
    for time in new_df_CU3["time_hour"].unique():
        tmp_df = new_df_CU3.loc[new_df_CU3["time_hour"] == time]
        if time not in CU3_avg_demand.keys():
            CU3_avg_demand[time] = []
        CU3_avg_demand[time].append(tmp_df["users"].sum())
    
    for time in new_df_CU4["time_hour"].unique():
        tmp_df = new_df_CU4.loc[new_df_CU4["time_hour"] == time]
        if time not in CU4_avg_demand.keys():
            CU4_avg_demand[time] = []
        CU4_avg_demand[time].append(tmp_df["users"].sum())

for time in CU1_avg_demand.keys():
    CU1_data["time_hour"].append(time)
    CU1_data["demand"].append(max(CU1_avg_demand[time]))#sum(CU1_avg_demand[time])/len(CU1_avg_demand[time]))

for time in CU2_avg_demand.keys():
    CU2_data["time_hour"].append(time)
    CU2_data["demand"].append(max(CU2_avg_demand[time]))#sum(CU1_avg_demand[time])/len(CU1_avg_demand[time]))

for time in CU3_avg_demand.keys():
    CU3_data["time_hour"].append(time)
    CU3_data["demand"].append(max(CU3_avg_demand[time]))#sum(CU1_avg_demand[time])/len(CU1_avg_demand[time]))

for time in CU4_avg_demand.keys():
    CU4_data["time_hour"].append(time)
    CU4_data["demand"].append(max(CU4_avg_demand[time]))#sum(CU1_avg_demand[time])/len(CU1_avg_demand[time]))

df = pandas.DataFrame.from_dict(CU1_data)
print(df.head())
df["demand"].plot(y="demand")
plt.show()

df = pandas.DataFrame.from_dict(CU2_data)
print(df.head())
df["demand"].plot(y="demand")
plt.show()

df = pandas.DataFrame.from_dict(CU3_data)
print(df.head())
df["demand"].plot(y="demand")
plt.show()

df = pandas.DataFrame.from_dict(CU4_data)
print(df.head())
df["demand"].plot(y="demand")
plt.savefig("analisys.pdf")
plt.show()