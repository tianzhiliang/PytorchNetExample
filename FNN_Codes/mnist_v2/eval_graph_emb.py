import sys,os,math,random

def loadv2(file):
    f = open(file, "r")
    datas = []
    for line in f:
        line = line.split()
        src = line[0]
        tgts = line[1:]
        datas.append([src, tgts])
    f.close()
    return datas

def load(file):
    f = open(file, "r")
    datas = []
    for line in f:
        line = line.split("\t")[:2]
        src = line[0]
        tgts = line[1].split()
        datas.append([src, tgts])
    f.close()
    return datas

def load_emb(file):
    f = open(file, "r")
    embs = []
    emb_dict = {}
    for line in f:
        slots = line.split()
        if len(slots) == 2:
            continue
        word = slots[0]
        emb = slots[1:]
        embs.append(emb)
        emb_dict[word] = len(embs) - 1
    f.close()
    return embs, emb_dict

def cos_sim(alist, blist):
    alist = list(map(float, alist))
    blist = list(map(float, blist))
    aa, bb, ab = 0, 0, 0
    for a, b in zip(alist, blist):
        aa += a*a
        bb += b*b
        ab += a*b

    aabb = math.sqrt(aa * bb)
    if aabb == 0:
        return 0

    return ab / aabb

def avg_cos_sim(vec, vecs):
    if len(vecs) == 0:
        return None, 0
    cos_sims = []
    for v in vecs:
        sim = cos_sim(v, vec)
        cos_sims.append(sim)
    return cos_sims, sum(cos_sims) / len(cos_sims)

def cmp_positive_and_negative(embs, emb_dict, graph, nodes_dict):
    nodes = nodes_dict.keys()
    nega_avg_sims_all, posi_avg_sims_all, posi_num, nega_num, total_cnt = [0] * 5
    lnodes = len(nodes)
    for node_edges in graph:
        snode = node_edges[0]
        if snode not in emb_dict:
            continue
        snode_vec = embs[emb_dict[snode]]

        tnodes = node_edges[1]
        tnodes_vecs = [embs[emb_dict[tn]] for tn in tnodes if tn in emb_dict]
        lent = len(tnodes)
        negative_nodes = random.sample(nodes, lent)
        negative_vecs = [embs[emb_dict[nn]] for nn in negative_nodes if nn in emb_dict]

        posi_sims, posi_avg_sims = avg_cos_sim(snode_vec, tnodes_vecs)
        nega_sims, nega_avg_sims = avg_cos_sim(snode_vec, negative_vecs)
        print("posi_avg_sims:", posi_avg_sims, "nega_avg_sims:", nega_avg_sims, \
             "posi_num:", len(tnodes_vecs), "nega_num:", len(negative_vecs))
        sys.stdout.flush()
        posi_avg_sims_all += posi_avg_sims
        nega_avg_sims_all += nega_avg_sims
        posi_num += len(tnodes_vecs)
        nega_num += len(negative_vecs)
        total_cnt += 1
    print("posi_avg_sims_all:", posi_avg_sims_all/total_cnt, \
          "nega_avg_sims_all:", nega_avg_sims_all/total_cnt, \
          "posi_num_all:", posi_num, \
          "nega_num_all:", nega_num)

def get_uniq_nodes(datas):
    nodes = {}
    for data in datas:
        src, tgts = data
        if src not in nodes:
            nodes[src] = 0
        nodes[src] += 1

        for tgt in tgts:
            if tgt not in nodes:
                nodes[tgt] = 0
            nodes[tgt] += 1

    return nodes

def get_uniq_edges(datas):
    edges = {}
    for data in datas:
        src, tgts = data
        for tgt in tgts:
            src_tgt = src + "_" + tgt
            if src_tgt not in edges:
                edges[src_tgt] = 0
            edges[src_tgt] += 1

    return edges

def get_avg_degree(nodes, edges):
    return len(edges.keys()) / len(nodes.keys())

def main():
    embfile = sys.argv[1]
    graph_file = sys.argv[2]

    embs, emb_dict = load_emb(embfile)
    graph = loadv2(graph_file)
    nodes_dict = get_uniq_nodes(graph)

    cmp_positive_and_negative(embs, emb_dict, graph, nodes_dict)

main()
