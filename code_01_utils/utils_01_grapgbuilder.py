import networkx as nx
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import rich.progress as richprogress
import typing as tp


class GraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph_from_csv(
            self,
            from_csv: str,
            split_by_day: bool = True,
            remove_self_loops: bool = True,
            debug_mode: bool = False,
            node_id_col: str = "caid",
    ):
        df = pd.read_csv(from_csv)
        df["trjid"] = df.apply(lambda row: f"u{row.uid}_d{row.d}", axis=1) if split_by_day else df["uid"]

        # 添加节点信息
        for _, group in richprogress.track(df.groupby([node_id_col]), description="[+] adding nodes"):
            group = group.reset_index(drop=True)
            self.graph.add_node(
                node_for_adding=group.at[0, node_id_col],
                # 由于存在长时间滞留记录，因此以经过此节点的轨迹数量作为度量
                trj=group["trjid"].unique(),
                popu=group["trjid"].nunique(),
                # 相同caid的位置记录应该是相同的，因此只取第0个记录的经纬度
                x=group.at[0, "x"],
                y=group.at[0, "y"],
            )
        if debug_mode:
            print(f"[=] {self.graph.number_of_nodes()} nodes added.")

        # 添加边信息

        trj_and_groups = df.groupby("trjid")
        for trjid, group in richprogress.track(trj_and_groups, description="[+] adding edges"):
            node_a = None
            for row_tuple in group.itertuples():
                node_b = getattr(row_tuple, node_id_col)
                if node_a is None:
                    node_a = node_b
                    continue
                else:
                    if remove_self_loops and node_a == node_b:
                        continue
                    if self.graph.has_edge(node_a, node_b):
                        self.graph.edges[node_a, node_b]["weight"] += 1
                    else:
                        self.graph.add_edge(node_a, node_b, weight=1)
                    node_a = node_b
        if debug_mode:
            print(f"[=] {self.graph.number_of_edges()} edges added.")

    def find_major_nodes(self, max_distance: float = 20):
        # 比所有邻居都大
        major_nodes = set()
        # 其他
        minor_nodes = set()
        merge_map = dict()

        for node_name in self.graph.nodes():
            if node_name in major_nodes or node_name in minor_nodes:
                continue

            # 如果在循环后flag仍然是True
            # 那么自己就是major
            flag = True

            # 自己节点的状态未知
            for neighbor_name in self.graph.neighbors(node_name):
                # 不考虑地理距离过远的邻居节点
                if not self.within_distance(node_name, neighbor_name, max_distance):
                    continue

                # 如果邻居是major
                # 则自己一定比邻居小
                # 自己一定是minor
                if neighbor_name in major_nodes:
                    minor_nodes.add(neighbor_name)
                    flag = False
                    continue

                node_popu = self.get_popular(node_name)
                neighbor_popu = self.get_popular(neighbor_name)
                # 如果邻居比自己大
                # 那么自己一定是minor
                # 邻居则无法确定
                if node_popu < neighbor_popu:
                    minor_nodes.add(node_name)
                    flag = False
                    continue
                # 如果与邻居相等，那么两者一定都是minor
                elif node_popu == neighbor_popu:
                    minor_nodes.add(neighbor_name)
                    minor_nodes.add(node_name)
                    flag = False
                    continue
                # 如果自己比邻居大
                # 则邻居一定是minor
                # 自己仍无法确定
                elif node_popu > neighbor_popu:
                    minor_nodes.add(neighbor_name)
                    continue
                else:
                    raise NotImplementedError

            # 如果在循环后flag仍然是True
            # 那么自己就是major
            if flag is True:
                major_nodes.add(node_name)
                merge_map[node_name]: set = {x for x in self.graph.neighbors(node_name) if self.within_distance(node_name, x, max_distance)}

        # print(merge_map)
        return merge_map

    def merge_nodes(
            self,
            major_node_name,
            minor_node_name,
            remove_minor_node: bool = True,
            debug_mode: bool = False,
    ):
        # 如果minor节点已经消失了
        # 代表此节点已经被其他节点率先合并
        if not self.graph.has_node(minor_node_name):
            if debug_mode:
                print("[!] 已被合并，跳过合并")
            return

        assert minor_node_name in self.graph.neighbors(major_node_name)

        major_node_trj = self.graph.nodes[major_node_name]["trj"]
        major_node_popu = self.graph.nodes[major_node_name]["popu"]

        minor_node_trj = self.graph.nodes[minor_node_name]["trj"]
        minor_node_popu = self.graph.nodes[minor_node_name]["popu"]

        assert major_node_popu > minor_node_popu, (f""
                                                   f"major-{major_node_name} has {major_node_popu} trj, "
                                                   f"while minor-{minor_node_name} has {minor_node_popu} trj. "
                                                   f"distance = {np.linalg.norm(self.get_coordibate(major_node_name) - self.get_coordibate(minor_node_name)):.2f}"
                                                   )

        # 节点属性-轨迹集合合并
        self.graph.nodes[major_node_name]["trj"] = list(set(major_node_trj) | set(minor_node_trj))
        self.graph.nodes[major_node_name]["popu"] = len(self.graph.nodes[major_node_name]["trj"])

        # 复制minor节点的邻居节点和连接权重到major节点
        for pre_node_name in self.graph.predecessors(minor_node_name):
            if pre_node_name == major_node_name:
                continue
            self.graph.add_edge(
                pre_node_name, major_node_name,
                weight=self.graph.edges[pre_node_name, minor_node_name]["weight"]
            )
        for suc_node_name in self.graph.successors(minor_node_name):
            if suc_node_name == major_node_name:
                continue
            self.graph.add_edge(
                major_node_name, suc_node_name,
                weight=self.graph.edges[minor_node_name, suc_node_name]["weight"]
            )

        # 删除minor节点
        if remove_minor_node:
            self.graph.remove_node(minor_node_name)

    def get_popular(self, node_name):
        return self.graph.nodes[node_name]["popu"]

    def get_coordibate(self, node_name):
        return np.array(self.graph.nodes[node_name]["x"], self.graph.nodes[node_name]["y"])

    def within_distance(self, node_a, node_b, max_distance):
        return np.linalg.norm(self.get_coordibate(node_a) - self.get_coordibate(node_b)) < max_distance

    def plot(
            self,
            node_size: int,
            arrow_size: int,
            with_labels: bool,
            alpha: float,
            fig_size: tp.Union[int, tp.Tuple[int, int]] = (12, 12),
    ):
        layout = {
            node_data[0]: (node_data[1]["x"], node_data[1]["y"])
            for node_data in self.graph.nodes.data()
        }
        fig_size = (fig_size, fig_size) if isinstance(fig_size, int) else fig_size
        plt.figure(figsize=fig_size)
        plt.axis("off")
        nx.draw_networkx(
            self.graph,
            node_size=node_size,
            arrowsize=arrow_size,
            with_labels=with_labels,
            alpha=alpha,
            pos=layout
        )

        # num_edges = self.graph.number_of_edges()
        # nx.draw_networkx_nodes(
        #     self.graph,
        #     node_size=node_size,
        #     pos=layout,
        #     node_color="blue"
        # )
        # nx.draw_networkx_edges(
        #     self.graph,
        #     pos=layout,
        #     # arrowstyle="->",
        #     alpha=alpha,
        #     arrowsize=arrow_size,
        #     edge_color=range(2, 2 + num_edges),
        #     edge_cmap=matplotlib.colormaps["YlOrRd"]
        # )

        plt.show()
        plt.close()

    def plot_ext(
            self,
            node_size: int,
            arrow_size: int,
            alpha: float,
            fig_size: tp.Union[int, tp.Tuple[int, int]] = (12, 12),
            xlim: tp.Optional[tp.Tuple[float, float]] = None,
            ylim: tp.Optional[tp.Tuple[float, float]] = None,
    ):
        layout = {
            node_data[0]: (node_data[1]["x"], node_data[1]["y"])
            for node_data in self.graph.nodes.data()
        }
        fig_size = (fig_size, fig_size) if isinstance(fig_size, int) else fig_size
        plt.figure(figsize=fig_size)

        if xlim is not None and ylim is not None:
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xticks(np.arange(int(xlim[0]), int(xlim[1]) + 1, 0.5))
            plt.yticks(np.arange(int(ylim[0]), int(ylim[1]) + 1, 0.5))
            plt.grid()
        else:
            plt.axis("off")

        num_edges = self.graph.number_of_edges()
        nx.draw_networkx_nodes(
            self.graph,
            node_size=node_size,
            pos=layout,
            node_color="blue",
        )
        nx.draw_networkx_edges(
            self.graph,
            pos=layout,
            arrowstyle="->",
            alpha=alpha,
            arrowsize=arrow_size,
            edge_cmap=matplotlib.colormaps["YlOrRd"],
            edge_color=range(2, 2 + num_edges),
        )
        # for index in range(num_edges):
        #     plt_edges[index].set_alpha((5 + index) / (num_edges + 4))

        # plt.savefig("./img.png", bbox_inches="tight", pad_inches=0)

        plt.show()
        plt.close()


if __name__ == "__main__":
    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v2")

    builder = GraphBuilder()

    # ======================================================================

    builder.build_graph_from_csv(
        from_csv=r"data_02_preprocessed_data/YJMob100K/p2_caid_added/12days_7_22/day10_to_day21_num200.csv",
        split_by_day=True,
        remove_self_loops=True,
        debug_mode=True,
        node_id_col="caid_50"
    )

    # builder.plot(
    #     node_size=2,
    #     arrow_size=5,
    #     with_labels=False,
    #     alpha=0.1,
    #     fig_size=32
    # )

    builder.plot_ext(
        node_size=2,
        arrow_size=5,
        alpha=0.5,
        fig_size=32,
    )

    # ======================================================================

    # cache_num_nodes = 0
    # cache_num_edges = 0
    # for _ in range(2):
    #     merge_dict = builder.find_major_nodes(max_distance=2)
    #     # import pprint
    #     # pprint.pprint(merge_map)
    #     for major, minor_set in merge_dict.items():
    #         for minor in minor_set:
    #             builder.merge_nodes(major, minor)
    #     num_nodes = builder.graph.number_of_nodes()
    #     num_edges = builder.graph.number_of_edges()
    #     if num_nodes == cache_num_nodes and num_edges == cache_num_edges:
    #         break
    #     else:
    #         cache_num_nodes = num_nodes
    #         cache_num_edges = num_edges
    #         print(f"[=] {num_nodes} nodes left.")
    #         print(f"[=] {num_edges} edges left.")
    #         print(f"[-] =============================================")
    #
    # builder.plot(
    #     node_size=2,
    #     arrow_size=5,
    #     with_labels=False,
    #     alpha=0.1,
    #     fig_size=32
    # )

    # ======================================================================
