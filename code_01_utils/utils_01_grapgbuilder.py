import networkx as nx
import pandas as pd
import numpy as np
import typing as tp
import os
import heapq
import matplotlib
import matplotlib.pyplot as plt

from rich.progress import track
from pprint import pp


class GraphBuilder:
    def __init__(self, graph_type: tp.Literal["directed", "undirected"]):
        self.graph = nx.DiGraph() if graph_type == "directed" else nx.Graph()

    def from_csv(
            self,
            csv_path: str,
            remove_loops: bool = False,
            user_id_col: str = "uid",
            node_id_col: str = "bsid",
    ):
        df = pd.read_csv(csv_path)
        if "trjid" not in df.columns:
            if "d" in df.columns:
                df["trjid"] = df.apply(lambda row: f"{row[user_id_col]}trj{row.d:03d}", axis=1)
            else:
                df["trjid"] = df.apply(lambda row: f"{row[user_id_col]}trj000", axis=1)

        # ADD NODE
        for node_name, group in track(df.groupby(node_id_col), description="[+] Adding nodes"):
            group = group.reset_index(drop=True)
            self.graph.add_node(
                node_for_adding=node_name,
                # node_for_adding=group.at[0, node_id_col],
                x=group.at[0, "x"],
                y=group.at[0, "y"],
                popularity=group["trjid"].nunique(),
            )
        print(f"[=] {self.graph.number_of_nodes()} nodes added.")

        # ADD EDGE
        for trj_name, group in track(df.groupby("trjid"), description="[+] Adding edges"):
            node_a = None
            node_b = None
            for index, node_name in enumerate(group[node_id_col]):
                if index == 0:
                    node_a = node_name
                    continue
                else:
                    assert node_a is not None
                    node_b = node_name
                    if node_a == node_b and remove_loops:
                        continue
                    if self.graph.has_edge(node_a, node_b):
                        self.graph[node_a][node_b]["weight"] += 1
                    else:
                        self.graph.add_edge(node_a, node_b, weight=1)
                    node_a = node_b
        print(f"[=] {self.graph.number_of_edges()} edges added.")
        return self

    def plot_ext(
            self,
            node_size: float | None = None,
            # arrow_size: int = 1,
            alpha: float = 0.8,
            fig_size: int | tp.Tuple[int, int] = 8,
            xlim: tp.Tuple[float, float] | None = None,
            ylim: tp.Tuple[float, float] | None = None,
            cmap_name: str = "YlOrRd",
            node_color: str = "blue",
            custom_scaler_max_ratio: float | None = None,
    ):
        layout = {
            node_data[0]: (node_data[1]["x"], node_data[1]["y"])
            for node_data in self.graph.nodes.data()
        }
        fig_size = (fig_size, fig_size) if isinstance(fig_size, int) else fig_size
        plt.figure(figsize=fig_size, tight_layout=True)
        if xlim is not None and ylim is not None:
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xticks(np.arange(int(xlim[0]), int(xlim[1]) + 1, 0.5))
            plt.yticks(np.arange(int(ylim[0]), int(ylim[1]) + 1, 0.5))
            plt.grid()
        else:
            plt.axis("off")

        node_popularities = [node_data["popularity"] for node_name, node_data in self.graph.nodes(data=True)]
        nx.draw_networkx_nodes(
            self.graph,
            # node_size=node_size,
            node_size=[node_size * popu for popu in node_popularities] if node_size is not None else 1,
            pos=layout,
            node_color=node_color,
        )

        edge_weights: list = [self.graph[node][edge]["weight"] for node, edge in self.graph.edges()]
        if custom_scaler_max_ratio is not None:
            num_edge_weights = len(edge_weights)
            scale_min = 1
            scale_max = heapq.nlargest(int((1 - custom_scaler_max_ratio) * num_edge_weights), edge_weights)[-1]
            edge_weights = [(x - scale_min) / (scale_max - scale_min) if x < scale_max else 1.0 for x in edge_weights]

        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        nx.draw_networkx_edges(
            self.graph,
            pos=layout,
            # arrowstyle="->",
            # arrowsize=arrow_size,
            alpha=alpha,
            edge_cmap=cmap,
            width=3,
            edge_color=edge_weights,
        )

        plt.margins(0, 0)

        plt.show()
        plt.close()


if __name__ == "__main__":
    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v3")

    builder = GraphBuilder("undirected").from_csv(
        # csv_path="data_02_preprocessed_data/YJMob100K/p1_filtered_strict/DURATION_20_DAY_07_26_UID_20.csv",
        csv_path="data_02_preprocessed_data/YJMob100K/p1_filtered/DURATION_40_DAY_14_53_UID_80.csv",
        remove_loops=True,
        node_id_col="bsid_50",
    )

    builder.plot_ext(
        node_size=1.2,
        alpha=0.25,
        fig_size=(16, 16),
        custom_scaler_max_ratio=0.9,
        node_color="blue",
        # cmap_name="cool",
    )
