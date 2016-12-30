import pandas as pd
import numpy as np
import json
import networkx as nx
from networkx.readwrite import json_graph
from config import SQLALCHEMY_DATABASE_URI
from sqlalchemy import create_engine
engine = create_engine(SQLALCHEMY_DATABASE_URI)

from ExtractFeatures import FeaturesByYr

db_yrs = pd.read_sql('select distinct yr from comtrade', engine).values
DB_END_YR = max(db_yrs)[0]


def build_graph(feature, threshold, k=1):
    ctry_names = pd.read_sql('select code, name, region from countries', engine).set_index('code').to_dict()
    neighbour_dict = FeaturesByYr(DB_END_YR-2,DB_END_YR).neighbour_dict(feature)
    adjacency_matrix = FeaturesByYr(DB_END_YR-2,DB_END_YR).adjacency_matrix(feature)
    ctries = adjacency_matrix.index.values
    G = nx.DiGraph()
    for i, code in enumerate(ctries):
        G.add_node(i)
        G.node[i]['code'] = code
        G.node[i]['name'] = ctry_names['name'][code]
        neighbour_names = [ctry_names['name'][c] for c in neighbour_dict[code][0][:4]]
        neighbour_sentence = ', '.join(neighbour_names[:-1])+' and '+ neighbour_names[-1]+'.'
        G.node[i]['neighbours'] = neighbour_sentence
        G.node[i]['region'] = ctry_names['region'][code]
        for neighbour in list(neighbour_dict[code][0])[:k]:  # add edges for first k neighbours
            j = list(ctries).index(neighbour)
            G.add_edge(i, j)
    # add additional edges wherever similarity > threshold
    if feature == 'distance':
        rows, cols = np.where(adjacency_matrix < threshold)
    else:
        rows, cols = np.where(adjacency_matrix > threshold)
    edges = zip(rows.tolist(), cols.tolist())
    G.add_edges_from(edges)
    # remove edges from nodes where adjacency matrix is missing
    for code in adjacency_matrix[(adjacency_matrix == 0).all()].index:
        i = list(ctries).index(code)
        G.remove_edges_from(G.edges([i]))
        G.node[i]['neighbours'] = 'none.'
    for code in adjacency_matrix[(adjacency_matrix == 40000).all()].index:
        i = list(ctries).index(code)
        G.remove_edges_from(G.edges([i]))
        G.node[i]['neighbours'] = 'none.'
    # write json formatted data
    d = json_graph.node_link_data(G)
    json.dump(d, open('docs/json/%s.json' % feature, 'w'))

if __name__ == "__main__":

    build_graph('rca', 0.25)
    build_graph('imports', 0.85)
    build_graph('export_destination', 0.9)
    build_graph('import_origin', 0.9)
    build_graph('intensity', 0.025)
    build_graph('distance', 1000, k = 0)

