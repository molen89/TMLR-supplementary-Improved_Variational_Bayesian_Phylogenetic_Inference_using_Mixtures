import ete3
import pandas as pd
import glob
import networkx as nx
import numpy as np
import json
import os
import logging
import scipy

class UT():
    def __init__(self, path=None):
        if path:
            self.load_unique_trees(path)
        else:
            self.unique_trees = {} #{tree : [lnl, frq, logq, tree]}
    def add(self, tree, lnl, logq, origin):
        nw_tree = tree.write(format=9)
        tree = ete3.Tree(nw_tree) #making sure the format is the same
        id = tree.get_topology_id()
        tmp = self.unique_trees.get(id)
        if tmp:
            tmp[4][origin] = tmp[4][origin]+1 if tmp[4].get(origin) else 1
            if lnl > tmp[0]:
                self.unique_trees[id] = [lnl, tmp[1] + 1, logq, nw_tree, tmp[4]]
            else:
                self.unique_trees[id] = [tmp[0], tmp[1] + 1, tmp[2], tmp[3], tmp[4]]
        else:
            self.unique_trees[id] = [lnl, 1, logq, nw_tree, {origin: 1}]

    def add_vbpi(self, trees, lnls, logq, origin, rename=True):
        for i in range(len(trees)):

            if rename:
                #since vbpi is 0 indexed and mrbayes is 1 indexed
                for leaf in trees[i].get_leaves():
                    leaf.name = leaf.name + 1

            self.add(trees[i], lnls[i].item(), logq[i].item(), origin)


    def add_vbpi(self, trees, origin, rename=True):
        for i in range(len(trees)):
            if rename:
                for leaf in trees[i].get_leaves():
                    leaf.name = leaf.name + 1
            self.add(trees[i], 0, 0, origin)

    def rebase_ut(self, tree_pos=0):
        new_ut = {}
        for id, data in self.unique_trees.items():
            t = ete3.Tree(data[tree_pos])
            data[tree_pos] = t
            for leaf in t.get_leaves():
                leaf.name = str(int(leaf.name) + 1)
            t.unroot()
            id = t.get_topology_id()

            new_ut[id] = data
        self.unique_trees = new_ut


    def save_unique_trees(self, path, postfix=""):
        logging.info(f"Saving unique trees: {path}")
        with open(path + f'/unique_trees{postfix}.json', 'w') as outfile:
            json.dump(self.unique_trees, outfile)

    def load_unique_trees(self, path, postfix=""):
        path = path + f'/unique_trees{postfix}.json'
        logging.info(f"Loading unique trees: {path}")
        with open(path) as json_unique_trees:
            self.unique_trees = json.load(json_unique_trees)
        logging.info(f"Finished loading unique trees, len={len(self.unique_trees)}")





def nexus_2_unique_trees(path_dir, burnin=0.25):
    """
    path_dir: path to result of mrbayes
    creates fils with trees in newick format from .t files of mrbayes
    """

    ut = UT()

    glob_dir = path_dir + "/*.t"
    files = glob.glob(glob_dir)
    for t_file in files:
        p_file = t_file[:-1] + "p"

        # get total amount of sampels
        n_samples = sum(1 for _ in open(p_file))
        burnin_limit = int(burnin*n_samples)

        logging.info(f"Reading {t_file} with burnin {burnin}, total number of samples used: {n_samples - burnin_limit}")

        count = 0
        with open(t_file, "r") as t:
            with open(p_file, "r") as p:
                p.readline(), p.readline() #skip the first two lines
                t_line = t.readline()
                while t_line:
                    t_line = t.readline().strip()
                    if t_line[:8] == "tree gen":
                        count += 1
                        p_line = p.readline()
                        if count > burnin_limit:
                            lnl = float(p_line.split("\t")[1])
                            # ut.add(ete3.Tree(t_line.split("=")[1][5:]).write(format=9), lnl, 0)
                            ut.add(ete3.Tree(t_line.split("=")[1][5:]), lnl, 0, 0)

    ut.save_unique_trees(path_dir)


def credible_set_mask(df, credible_set=0.95, attr="frq"):

    """
    Parameters
    ----------
    df: pandas dataframe with all data with access to "attr"
    credible_set: what level of credible set we are looking to trim based on
    attr: what key is used 

    Returns
    -------
    df: sorted data
    mask: corresponding mask of items inside the .95 credible set
    """

    if attr == "frq":
        tot = df[attr].sum()
        limit = tot*credible_set
        df.sort_values(attr, inplace=True, ascending=False)
        mask = np.zeros(df.shape[0], dtype=np.bool)
        cumsum = 0
        for i in range(df.shape[0]):
            cumsum += df[attr][i]
            mask[i] = 1
            if cumsum > limit:
                break

    elif attr == "logq":

        tot = np.exp(df[attr]).sum()
        df.sort_values(attr, inplace=True, ascending=False)
        limit = tot*credible_set

        mask = np.zeros(df.shape[0], dtype=np.bool)
        cumsum = 0
        for i in range(df.shape[0]):
            cumsum += np.exp(df[attr][i])
            mask[i] = True
            if cumsum > limit and i > 2:
                break

    logging.info(f"Credible set: {credible_set} with attr {attr}, tot: {tot}, final amount: {mask.sum()} at limit: {limit}, final cs: {cumsum / tot}")

    return df, mask


def trim_unique_trees(unique_trees, nlargest=2048, trim_attr="frq", credible_set=0.95, return_dropped=False):
    """
    Unique_trees: dict with {newick tree : (LnL, frequenzy)}
    quantile: only save the top 1-quantile of attribut
    attr_type: 0=LnL, 1=frequenzy
    """

    df = pd.DataFrame.from_dict(unique_trees).T
    start_n = df.shape[0]

    df = df.rename(columns={0: "lnl", 1:"frq", 2:"logq", 3:"trees"})
    df = df.astype({
        "trees": "str",
        "frq": "int64",
        "lnl": "float64",
        "logq": "float64",
    })

    if credible_set >= 1.0:
        df_trimmed = df.nlargest(nlargest, trim_attr)
    else:
        df, mask = credible_set_mask(df, attr=trim_attr, credible_set=credible_set)

        if mask.sum() > nlargest:
            df_trimmed = df.nlargest(nlargest, trim_attr)
        else:
            df_trimmed = df[mask]


    logging.info(f"Start amount: {start_n}, End amount: {df_trimmed.shape[0]}, Density final: {df_trimmed.frq.sum()/df.frq.sum()}")

    if return_dropped:

        df_dropped = pd.concat([df, df_trimmed]).drop_duplicates(keep=False)

        return df_trimmed, df_dropped
    else:
        return df_trimmed.T.to_dict(orient="list")


from io import StringIO


def support_2_graph(unique_trees, rspr_path, result_path, n_proc):
    df = pd.DataFrame.from_dict(unique_trees).T
    n = df.shape[0]
    df = df.reset_index(level=0)

    df = df.astype({"index": "str", 0: "str",})

    data = get_rspr_matrix(df[0], rspr_path=rspr_path, n_proc=n_proc, max_dist=1)
    data[data == -1] = 0  # remove edges if they are negative
    G = nx.from_numpy_array(data, create_using=nx.Graph)

    for i in range(df.shape[0]):
        row = df.iloc[i]
        G.nodes[i]["id"] = row["index"]
        G.nodes[i]["trees"] = row[0]

    logging.info(f"Number of nodes: {G.number_of_nodes()}\tNumber of edges: {G.number_of_edges()}")
    save_graph(G, result_path, postfix="_mix_template")

def unique_trees_mix_2_graph(G, unique_trees, result_path):

    for node in G.nodes:
        node = G.nodes[node]
        row = unique_trees.get(node["id"])
        node["logqt"] = row[1]
        for s in range(len(row)-2):
            node[f"logqt_{s}"] = row[2 + s]
        node["largest_origin"] = np.argmax(row[2:])

    logging.info(f"Number of nodes: {G.number_of_nodes()}\tNumber of edges: {G.number_of_edges()}")
    save_graph(G, result_path, postfix="_mix")






def unique_trees_2_graph_cluster(unique_trees, max_distance=1, rspr_path='./rspr', attr = "frq", max_cluster =10, n_proc=5):

    logging.info(f"Creating graph with clusters and rspr")

    df = pd.DataFrame.from_dict(unique_trees).T
    n = df.shape[0]
    df = df.reset_index(level=0)

    if df.shape[1] == 4: #hence old unique trees and no origin column
        df[4] = 0

    df = df.rename(columns={"index": "id", 0: "lnl", 1:"frq", 2:"logq", 3:"trees", 4:"origin"})
    df = df.astype({
        "id":"str",
        "trees": "str",
        "frq": "int64",
        "lnl": "float64",
        "logq": "float64",
        "origin": "object",
    })

    df.sort_values(attr, inplace=True, ascending=False)
    df = df.reset_index(level=0, drop=True)

    logging.info(f"Amount of nodes: {n}")

    if n_proc==1:
        # commands = [rspr_path, "-q", "-pairwise", "-unrooted", "-split_approx", str(max_distance)]
        commands = [rspr_path, "-q", "-pairwise", "-unrooted"]
        out = run(commands, input="\n".join(df.trees), capture_output=True, text=True, encoding='ascii', shell=False)

        # load and fill matrix
        logging.info(f"Fill matrix")
        # data_str = out.stdout.split("\n", 1)[1]
        data_str = out.stdout

        c = StringIO(data_str)
        data = np.genfromtxt(c, delimiter=",")
        for k in range(n):
            row = data[k, k:]
            data[k:, k] = row
    else:
        data = get_rspr_matrix(df.trees, rspr_path=rspr_path, n_proc=n_proc)

    logging.info(f"Matrix found of size: {data.shape}")

    cluster = np.zeros(n)
    cluster_peak_r = np.zeros(n)
    selector = df[attr]

    logging.info(f"Find clusters")

    for cluster_id in range(1, max_cluster+1):
        not_clustered = cluster == 0
        n_not_clustered = len(cluster[not_clustered])
        if n_not_clustered <= 2:
            logging.info(f"Stopping with {n_not_clustered} not clustered and {cluster_id-1} clusters")
            break

        #find peak
        peak = (selector*not_clustered).argmax()
        cluster[peak] = cluster_id
        # print(f"Peak: {selector[peak]} at {peak}")

        #get distances
        dist2peak = data[peak, :]
        dist2peak_mask = (dist2peak*not_clustered) > 0 # 0 all that is already clustered
        # print(dist2peak_mask.sum())

        #find r
        r = dist2peak[dist2peak_mask].mean() - dist2peak[dist2peak_mask].std()
        cluster_peak_r[peak] = r

        #assigne trees to clusters
        dist2peak_r_mask = (dist2peak < r)*not_clustered #all in radius \ all that are clustered
        cluster[dist2peak_r_mask] = cluster_id

    # df["cluster"] = cluster
    # df["cluster_peak_r"] = cluster_peak_r

    logging.info(f"Create graph")

    # G = nx.from_numpy_matrix(data, create_using=nx.Graph)
    G = nx.from_numpy_array(data, create_using=nx.Graph)
    for i in range(df.shape[0]):
        G.nodes[i]["id"] = df.id[i]
        G.nodes[i]["trees"] = df.trees[i]
        G.nodes[i]["lnl"] = df.lnl[i]
        G.nodes[i]["frq"] = df.frq[i]
        G.nodes[i]["logq"] = df.logq[i]
        G.nodes[i]["cluster"] = cluster[i]
        G.nodes[i]["cluster_peak_r"] = cluster_peak_r[i]

        largest = 0
        max = 0
        for k,v in df.origin[i].items():
            if max < v:
                largest = k
            G.nodes[i][f"S{k}"] = v
        G.nodes[i]["largest_origin"] = largest

    logging.info(f"Number of nodes: {G.number_of_nodes()}\tNumber of edges: {G.number_of_edges()}")

    return G

import multiprocessing
from multiprocessing import Process, Pipe
from subprocess import run
import subprocess


def get_rspr_matrix(trees_str, rspr_path='./rspr', n_proc=5, max_dist=-1):
    n = len(trees_str)
    pairs = [(j, k) for j in range(n) for k in range(j + 1, n)]
    logging.info(f"Amount of pairs: {len(pairs)} for {n} trees, for {multiprocessing.cpu_count()}")

    if max_dist > 0:
        commands = [rspr_path, "-q", "-pairwise","0", "1", "1", "-unrooted", "-pairwise_max", str(max_dist), "-approx"] #approximation?
    else:
        commands = [rspr_path, "-q", "-pairwise","0", "1", "1", "-unrooted"]
    def get_rspr(t1,t2):
        out = run(commands, input=f"{t1}\n{t2}", capture_output=True, text=True, encoding='ascii', shell=False)
        dist = int(out.stdout)
        return dist

    def get_edges(conn, tree_pairs, proc):
        edges = []
        n_max = len(tree_pairs)
        logging.info(f"Process: {proc} have {n_max} pairs")

        # for n, (i, j) in enumerate(tree_pairs):
        for (i, j) in tree_pairs:
            dist = get_rspr(trees_str[i], trees_str[j])
            edges.append((i,j, dist))


        logging.info(f"amount of edges: {len(edges)} in process: {proc}")
        conn.send(edges)
        conn.close()

    n_splits = min(len(pairs), n_proc)
    processes = []
    splits = np.array_split(pairs, n_splits)
    for proc in range(n_splits):
        logging.info(f"starting process: {proc + 1}")
        parent_conn, child_conn = Pipe()
        p = Process(target=get_edges, args=(child_conn, splits[proc], proc + 1))
        p.start()
        processes.append((p, parent_conn))

    data = np.zeros((n,n))
    for proc in range(n_splits):
        edges = processes[proc][1].recv()
        for (i,j,d) in edges:
            data[i,j]=data[j,i]=d

        processes[proc][0].join()
        logging.info(f"Closing process: {proc + 1}")

    return data


def save_graph(G, path, postfix=""):
    nx.write_graphml(G, os.path.join(path,f"graph{postfix}.graphml"))
    # G_trimmed = trim_graph(G)

    for (u, v) in G.edges:
        if G[u][v]["weight"] > 1.0:
            G.remove_edge(u, v)

    nx.write_graphml(G, os.path.join(path, f"graph_trimmed{postfix}.graphml"))

def load_graph(path, postfix="", trimmed=True):
    if trimmed:
        p = os.path.join(path, f"graph_trimmed{postfix}.graphml")
    else:
        p = os.path.join(path, f"graph{postfix}.graphml")
    return nx.read_graphml(p)


import scipy
def distances_scipy(x, y, dec=10, from_support=-1): #from support = -1 for none, 1, for x and 2 for y

    ids = list(set(x.keys()).union(set(y.keys())))
    logging.info(f"from_support: {from_support}, length of sets: {len(x)}, {len(y)} \t union: {len(ids)}")


    xs = np.zeros(len(ids))
    ys = np.zeros(len(ids))

    if from_support == 1:
        xs = -np.inf*np.ones(len(ids))
    elif from_support == 2:
        ys = -np.inf*np.ones(len(ids))


    for i, id in enumerate(ids):
        if x.get(id):
            xs[i] = x[id][1]

        if y.get(id):
            ys[i] = y[id][1]


    # normalize
    if from_support==1:
        xs = np.exp(xs)
        ys = ys/ys.sum()
    elif from_support==2:
        xs = xs/xs.sum()
        ys = np.exp(ys)
    else:
        xs = xs/xs.sum()
        ys = ys/ys.sum()


    logging.info(f"xs.sum: {xs.sum()}, ys.sum: {ys.sum()}")

    kl_xy = scipy.special.kl_div(xs,ys)
    rel_entr_xy = scipy.special.rel_entr(xs,ys)
    energy = scipy.stats.energy_distance(xs, ys)
    wasserstein = scipy.stats.wasserstein_distance(xs, ys)
    jsd = scipy.spatial.distance.jensenshannon(xs, ys)

    tot_var = 0.5*np.abs(xs-ys).sum()

    return round(kl_xy.sum(), dec), round(rel_entr_xy.sum(), dec), round(energy, dec), round(wasserstein, dec), round(jsd, dec), round(tot_var, dec)


def masses(x,y, dec=10):
    ids = list(set(x.keys()).union(set(y.keys())))
    xs = np.zeros(len(ids))
    ys = np.zeros(len(ids))
    for i, id in enumerate(ids):
        if x.get(id):
            xs[i] = x[id][1]
        if y.get(id):
            ys[i] = y[id][1]

    # normalize
    xs = xs/xs.sum()
    ys = ys/ys.sum()

    # if x is approximation
    mass_of_y_if_not_x = 0
    mass_of_x_if_not_y = 0
    mass_of_x_correct = 0
    amount_x_zero = (xs == 0).sum()
    for i, id in enumerate(ids):
        if xs[i] > 0 and ys[i] > 0: #x>0, y>0
            mass_of_x_correct += xs[i]

        elif ys[i] > 0 and xs[i]==0:
            mass_of_y_if_not_x += ys[i]

        elif ys[i] == 0 and xs[i]>0:
            mass_of_x_if_not_y += xs[i]

        elif xs[i]==0 and ys[i]==0:
            logging.ERROR("something is wrong in masses")
    return round(mass_of_x_correct, dec), round(mass_of_x_if_not_y, dec),  round(mass_of_y_if_not_x, dec), amount_x_zero

def entropy_scipy(x, dec=6):

    ids = list(set(x.keys()))
    xs = np.zeros(len(ids))
    for i, id in enumerate(ids):
        if x.get(id):
            xs[i] = x[id][1]

    xs = xs/xs.sum()

    entr = scipy.special.entr(xs)
    return round(entr.sum(), dec)



def get_metrics(golden_ut, vbpi_ut):
    golden_ut_trimmed = trim_unique_trees(golden_ut, credible_set=0.95, nlargest=9999999999)
    vbpi_ut_trimmed = trim_unique_trees(vbpi_ut, credible_set=1.0, nlargest=len(golden_ut_trimmed))

    logging.info(f"q=vbpi, q'=vbpi trimmed, p=golden, p'=trimmed golden. \n(KL(x|y)=inf when x>0 and y=0, hence missing in golden but exists in vbpi, For=mean-seeking, Rev=mode-seeking)")

    logging.info(f"Entropy(q): {entropy_scipy(vbpi_ut)}")
    logging.info(f"Entropy(q'): {entropy_scipy(vbpi_ut_trimmed)}")
    logging.info(f"Entropy(p): {entropy_scipy(golden_ut)}")
    logging.info(f"Entropy(p'): {entropy_scipy(golden_ut_trimmed)}")

    # reverse
    D = distances_scipy(vbpi_ut, golden_ut)
    logging.info(f"Rev KL(q|p): {D[0]} \tRel Entropy: {D[1]} \tEnergy: {D[2]} \tWasserstein: {D[3]} \tJSD: {D[4]} \tTotVar: {D[5]}")

    D = distances_scipy(vbpi_ut_trimmed, golden_ut)
    logging.info(f"Rev KL(q'|p): {D[0]} \tRel Entropy: {D[1]} \tEnergy: {D[2]} \tWasserstein: {D[3]} \tJSD: {D[4]} \tTotVar: {D[5]}")

    D = distances_scipy(vbpi_ut, golden_ut_trimmed)
    logging.info(f"Rev KL(q|p'): {D[0]} \tRel Entropy: {D[1]} \tEnergy: {D[2]} \tWasserstein: {D[3]} \tJSD: {D[4]} \tTotVar: {D[5]}")

    D = distances_scipy(vbpi_ut_trimmed, golden_ut_trimmed)
    logging.info(f"Rev KL(q'|p'): {D[0]} \tRel Entropy: {D[1]} \tEnergy: {D[2]} \tWasserstein: {D[3]} \tJSD: {D[4]} \tTotVar: {D[5]}")

    D = distances_scipy(golden_ut_trimmed, golden_ut)
    logging.info(f"Rev KL(p'|p): {D[0]} \tRel Entropy: {D[1]} \tEnergy: {D[2]} \tWasserstein: {D[3]} \tJSD: {D[4]} \tTotVar: {D[5]}")

    # forward
    D = distances_scipy(golden_ut, vbpi_ut)
    logging.info(f"For KL(p|q): {D[0]} \tRel Entropy: {D[1]} \tEnergy: {D[2]} \tWasserstein: {D[3]} \tJSD: {D[4]} \tTotVar: {D[5]}")

    D = distances_scipy(golden_ut, vbpi_ut_trimmed)
    logging.info(f"For KL(p|q'): {D[0]} \tRel Entropy: {D[1]} \tEnergy: {D[2]} \tWasserstein: {D[3]} \tJSD: {D[4]} \tTotVar: {D[5]}")

    D = distances_scipy(golden_ut_trimmed, vbpi_ut)
    logging.info(f"For KL(p'|q): {D[0]} \tRel Entropy: {D[1]} \tEnergy: {D[2]} \tWasserstein: {D[3]} \tJSD: {D[4]} \tTotVar: {D[5]}")

    D = distances_scipy(golden_ut_trimmed, vbpi_ut_trimmed)
    logging.info(f"For KL(p'|q'): {D[0]} \tRel Entropy: {D[1]} \tEnergy: {D[2]} \tWasserstein: {D[3]} \tJSD: {D[4]} \tTotVar: {D[5]}")

    D = distances_scipy(golden_ut, golden_ut_trimmed)
    logging.info(f"For KL(p|p'): {D[0]} \tRel Entropy: {D[1]} \tEnergy: {D[2]} \tWasserstein: {D[3]} \tJSD: {D[4]} \tTotVar: {D[5]}")

    mass = masses(vbpi_ut, golden_ut_trimmed)
    logging.info(f"Masses(q,p') Both>0: {mass[0]} \tApprox=0: {mass[1]} \tTarget=0: {mass[2]} \tNumber of target is zero: {mass[3]}")

    mass = masses(vbpi_ut, golden_ut)
    logging.info(f"Masses(q,p) Both>0: {mass[0]} \tApprox=0: {mass[1]} \tTarget=0: {mass[2]} \tNumber of target is zero: {mass[3]}")


    return




def get_metrics_from_support(golden_ut, vbpi_ut):
    logging.info(f"q=vbpi, q'=vbpi trimmed, p=golden, p'=trimmed golden. \n(KL(x|y)=inf when x>0 and y=0, hence missing in golden but exists in vbpi, For=mean-seeking, Rev=mode-seeking)")

    logging.info(f"Entropy(q): {entropy_scipy(vbpi_ut)}")
    logging.info(f"Entropy(p): {entropy_scipy(golden_ut)}")

    # reverse
    D = distances_scipy(vbpi_ut, golden_ut, from_support=1)
    logging.info(f"Rev KL(q|p): {D[0]} \tRel Entropy: {D[1]} \tEnergy: {D[2]} \tWasserstein: {D[3]} \tJSD: {D[4]} \tTotVar: {D[5]}")

    # forward
    D = distances_scipy(golden_ut, vbpi_ut, from_support=2)
    logging.info(f"For KL(p|q): {D[0]} \tRel Entropy: {D[1]} \tEnergy: {D[2]} \tWasserstein: {D[3]} \tJSD: {D[4]} \tTotVar: {D[5]}")

    return




































