import pandas as pd
import numpy as np
import pickle  
from loguru import logger

import networkx as nx
from domirank import domirank 
from SpringRank import get_ranks


def load_benchmark_pickle_to_df(pickle_file):
    infile = open(pickle_file,'rb')  
    df = pickle.load(infile)
    return df


def get_graph_from_edgelist(edgelist: np.ndarray):
    # build graph from edgelist
    G = nx.Graph()
    G.add_edges_from(edgelist)

    return G


# Note - all of the helper functions below start with the same two lines - creating a graph G and adding edges from the edgelist

def apply_domirank_to_edgelist(graphProperties: str, G: nx.Graph, network_index: str, sigma: float = 0.0000000001):
    
    if not ("Undirected" in graphProperties):
        logger.info(f"Domirank can only be applied to Undirected networks. Network {network_index}'s properties are {graphProperties}.")
        return " "

    try:
        dr = domirank(G)
    except Exception as e:
        logger.info(f'Error: {e}')
        return " "
    
    return dr

def get_degree_from_edgelist(G: nx.Graph):

    try:
        deg = nx.degree_centrality(G)
    except Exception as e:
        logger.info(f'Error: {type(e)}')
        return " "
    
    return deg


def get_in_degree_from_edgelist(G: nx.Graph):

    try:
        deg = nx.in_degree_centrality(G)
    except Exception as e:
        logger.info(f'Error: {e}')
        return " "

    return deg


def get_out_degree_from_edgelist(G: nx.Graph):

    try:
        deg = nx.out_degree_centrality(G)
    except Exception as e:
        logger.info(f'Error: {e}')
        return " "

    return deg


def get_eigen_cent_from_edgelist(G: nx.Graph):

    try:
        eig = nx.eigenvector_centrality_numpy(G)
    except Exception as e:
        logger.info(f'Error: {e}')
        return " "

    return eig


def get_pageRank(G: nx.Graph):

    try:
        pr = nx.pagerank(G)
    except Exception as e:
        logger.info(f'Error: {e}')
        return " "

    return pr


def get_springRank(G: nx.Graph):

    A = nx.adjacency_matrix(G).todense()
    
    try:
        sr = get_ranks(A)
    except Exception as e:
        logger.info(f'Error: {e}')
        return " "

    return sr
    

def get_closeness(G: nx.Graph):

    try:
        close = nx.closeness_centrality(G)
    except Exception as e:
        logger.info(f'Error: {e}')
        return " "

    return close



def main():

    # load df from pickle
    pickle_file = 'CommunityFitNet-data/Benchmark_updated/CommunityFitNet_updated.pickle'
    df = load_benchmark_pickle_to_df(pickle_file)
    logger.info("Loaded the benchmarking dataset...")

    # save to pickle in Benchmark_updated
    out_file_path = 'CommunityFitNet-data/Benchmark_updated/CommunityFitNet_updated_'

    # create new df - output/cluster results df
    out_df = pd.DataFrame(columns=['network_index'])
    temp_df = None

    # Add the graph column from edgelist
    df['graph'] = df['edges_id'].apply(get_graph_from_edgelist)
    logger.info("Added graph column...")

    # load in cluster metadata df
    meta_df = pd.read_csv('CommunityFitNet-data/Benchmark_updated/cluster_method_meta.csv')
    for index, row in meta_df.iterrows():
        
        df_filter = df.copy()


        if row['method_name'] == "DomiRank":

            # domirank cluster scores column & and any other future
            df_filter['domirank sm sig'] = df_filter.apply(lambda x: apply_domirank_to_edgelist(x['graphProperties'],
                                                                                                x['graph'], 
                                                                                                x['network_index']), axis=1)            
            
            df_filter['domirank lrg sig'] = df_filter.apply(lambda x: apply_domirank_to_edgelist(x['graphProperties'],
                                                                                                x['graph'], 
                                                                                                x['network_index'],
                                                                                                sigma=0.5), axis=1)
            # add id and domirank columns to new df            
            temp_df = df_filter.loc[:, ['network_index', 'domirank sm sig', 'domirank lrg sig']]
            logger.info(f"Completed adding {row['method_name']} centrality column to the dataframe...")


        if row['method_name'] == "degree":

            # degree centrality scores column 
            df_filter['degree'] = df_filter.apply(lambda x: get_degree_from_edgelist(x['graph']), axis=1)

            # add id and degree_centrality columns to new df
            temp_df = df_filter.loc[:, ['network_index', 'degree']]
            logger.info(f"Completed adding {row['method_name']} centrality column to the dataframe...")


        if row['method_name'] == "in_out_degree":

            # filter df to only graphs that have properties in the list
            df['has_graph_props'] = df['graphProperties'].apply(lambda x: "True" if row['properties'] in x else "False")
            df_filter = df[df['has_graph_props'] == "True"]

            # add in and out degree centrality scores column 
            df_filter['in_degree'] = df_filter.apply(lambda x: get_in_degree_from_edgelist(x['graph']), axis=1)
            df_filter['out_degree'] = df_filter.apply(lambda x: get_out_degree_from_edgelist(x['graph']), axis=1)

            # add id and degree_centrality columns to new df
            temp_df = df_filter.loc[:, ['network_index', 'in_degree', 'out_degree']]
            logger.info(f"Completed adding {row['method_name']} centrality column to the dataframe...")


        if row['method_name'] == 'Eigenvector':
            
            df_filter['eigenvector'] = df_filter.apply(lambda x: get_eigen_cent_from_edgelist(x['graph']), axis=1)

            # add id and eigenvector centrality columns to new df
            temp_df = df_filter.loc[:, ['network_index', 'eigenvector']]
            logger.info(f"Completed adding {row['method_name']} centrality column to the dataframe...")

       
        if row['method_name'] == 'PageRank':

            df_filter['pagerank'] = df_filter.apply(lambda x: get_pageRank(x['graph']), axis=1)
            
            # add id and eigenvector centrality columns to new df
            temp_df = df_filter.loc[:, ['network_index', 'pagerank']]
            logger.info(f"Completed adding {row['method_name']} centrality column to the dataframe...")


        if row['method_name'] == 'SpringRank':

            # add sr centrality measure
            df_filter['springrank'] = df_filter.apply(lambda x: get_springRank(x['graph']), axis=1)
            
            # add id and eigenvector centrality columns to new df
            temp_df = df_filter.loc[:, ['network_index', 'springrank']]
            logger.info(f"Completed adding {row['method_name']} centrality column to the dataframe...")


        if row['method_name'] == 'Closeness':

            df_filter['closeness'] = df_filter.apply(lambda x: get_closeness(x['graph']), axis=1)

            # add id and closeness centrality columns to new df
            temp_df = df_filter.loc[:, ['network_index', 'closeness']]
            logger.info(f"Completed adding {row['method_name']} centrality column to the dataframe...")


        # outer join temp_df to out_df 
        if isinstance(temp_df, pd.DataFrame):
            out_df = out_df.merge(right=temp_df, how="outer", on="network_index")
        
    out_df.to_pickle(out_file_path + "centralities.pickle")

    return 


if __name__ == "__main__":
    main()