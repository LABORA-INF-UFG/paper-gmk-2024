import pandas
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

matplotlib.rcParams.update({'font.size': 20})

plt.rcParams["figure.figsize"] = (8, 4)

# plt.rcParams["font.family"] = "Times New Roman"

def fmt(x):
    return '{:.1f}%'.format(x)

def pie_plot():

    df = pandas.read_csv("../DRL_agent/evaluation_data/Murti_47_RUs_4_CUs_AC.csv")

    plt.pie(df["flag"].value_counts().values, labels=["Infeasible", "Feasible"], autopct=fmt, colors=["gray", "royalblue"])

    plt.savefig("pie_plot.pdf", bbox_inches='tight')


def lines_plot():
    df_murti = pandas.read_csv("../DRL_agent/evaluation_data/Murti_47_RUs_4_CUs_AC.csv")
    ax = df_murti["constraints"].plot(figsize=(8, 4), color="black", linewidth=1.3)
    ax.hlines(y=0, xmin=0, xmax=len(df_murti["constraints"]), linewidth=3, color="royalblue", linestyle="-")
    ax.fill_between(x=[i for i in range(0, len(df_murti["constraints"]))], y1 = df_murti["constraints"].to_list(), y2 = 0, color="gray", alpha=0.3)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_xlim(0, 167)
    ax.set_ylim(-0.2, 26)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
    ax.grid()
    ax.set_ylabel("Broken constraints (#)")
    ax.set_xlabel("Instances")

    # ax1 = ax.twinx()
    # df_DRL["reward"] = df_DRL["reward"].apply(lambda x: (- 1) * (x - 1) * 10.5 * 47)
    # ax = df_DRL["reward"].plot(figsize=(8, 4), color="royalblue", linewidth=2)
    # ax.set_ylabel("Total cost (#)")
    # ax.set_xlabel("Instance")
    # df_murti["reward"] = df_murti["reward"].apply(lambda x: (- 1) * (x - 1) * 10.5 * 47)
    # df_optimal["reward"] = df_optimal["reward"].apply(lambda x: (- 1) * (x - 1) * 10.5 * 47)
    # df_optimal["reward"].plot(ax=ax, figsize=(8, 4), color="lightgray", linewidth=2)
    # df_murti["reward"].plot(ax=ax, figsize=(8, 4), color="firebrick", linewidth=2)
    # ax.set_xlim(0, 166)

    # # ax.set_ylim(0, 1)
    # # ax.set_ylim(0, 1)
    # plt.ylim(0, 280)
    # plt.xlim(1, 168)
    # plt.grid(axis="y", linestyle='--')

    plt.rcParams.update({'font.size': 18})

    legend_elements = [Line2D([0], [0], color='royalblue', lw=3, label='DRL agent'),
                   Line2D([0], [0], color='black', lw=3, label='vRAN-UDP')]
    plt.legend(handles=legend_elements, loc="upper right", ncol=3)

    plt.savefig("50%_comparison.pdf", bbox_inches='tight')

# # plt.ylim(0, 1)



# pie_plot()

lines_plot()