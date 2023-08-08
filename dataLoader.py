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


# Note - all of the helper functions below start with the same two lines - creating a graph G and adding edges from the edgelist

def apply_domirank_to_edgelist(graphProperties: str, edgelist: np.ndarray, network_index: str, sigma: float = 0.0000000001):
    
    if not ("Undirected" in graphProperties):
        logger.info(f"Domirank can only be applied to Undirected networks. Network {network_index}'s properties are {graphProperties}.")
        return " "

    # build graph from edgelist
    G = nx.Graph()
    G.add_edges_from(edgelist)

    # apply domirank
    dr = domirank(G)
    # logger.info(f"Successfully added domirank for {network_index}.")

    return dr

def get_degree_from_edgelist(edgelist: np.ndarray):

    G = nx.Graph()
    G.add_edges_from(edgelist)

    try:
        deg = nx.degree_centrality(G)
    except Exception as e:
        logger.info(f'Error: {type(e)}')
        return " "
    
    return deg

def get_in_degree_from_edgelist(edgelist: np.ndarray):

    G = nx.Graph()
    G.add_edges_from(edgelist)

    try:
        deg = nx.in_degree_centrality(G)
    except Exception as e:
        logger.info(f'Error: {e}')
        return " "

    return deg

def get_out_degree_from_edgelist(edgelist: np.ndarray):

    G = nx.Graph()
    G.add_edges_from(edgelist)

    try:
        deg = nx.out_degree_centrality(G)
    except Exception as e:
        logger.info(f'Error: {e}')
        return " "

    return deg

def get_eigen_cent_from_edgelist(edgelist: np.ndarray):

    G = nx.Graph()
    G.add_edges_from(edgelist)

    try:
        eig = nx.eigenvector_centrality_numpy(G)
    except Exception as e:
        logger.info(f'Error: {e}')
        return " "

    return eig

def get_pageRank(edgelist: np.ndarray):

    G = nx.Graph()
    G.add_edges_from(edgelist)

    try:
        pr = nx.pagerank(G)
    except Exception as e:
        logger.info(f'Error: {e}')
        return " "

    return pr

def get_springRank(edgelist: np.ndarray):

    G = nx.Graph()
    G.add_edges_from(edgelist)
    A = nx.adjacency_matrix(G).todense()
    
    try:
        sr = get_ranks(A)
    except Exception as e:
        logger.info(f'Error: {e}')
        return " "

    return sr
    


def main():

    # load df from pickle
    pickle_file = 'CommunityFitNet-data/Benchmark_updated/CommunityFitNet_updated.pickle'
    df = load_benchmark_pickle_to_df(pickle_file)

    # save to pickle in Benchmark_updated
    out_file_path = 'CommunityFitNet-data/Benchmark_updated/CommunityFitNet_updated_'

    # create new df - output/cluster results df
    out_df = pd.DataFrame(columns=['network_index'])
    temp_df = None


    # load in cluster metadata df
    meta_df = pd.read_csv('CommunityFitNet-data/Benchmark_updated/cluster_method_meta.csv')
    for index, row in meta_df.iterrows():
        
        df_filter = df.copy()


        if row['method_name'] == "DomiRank":

            # filter df to only graphs that have properties in the list  
            # df['has_graph_props'] = df['graphProperties'].apply(lambda x: "True" if row['properties'] in x else "False")
            # df_filter = df[df['has_graph_props'] == "True"]

            # domirank cluster scores column & and any other future
            df_filter['domirank sm sig'] = df_filter.apply(lambda x: apply_domirank_to_edgelist(x['graphProperties'],
                                                                            x['edges_id'], 
                                                                            x['network_index']), axis=1)            
            
            df_filter['domirank lrg sig'] = df_filter.apply(lambda x: apply_domirank_to_edgelist(x['graphProperties'],
                                                                            x['edges_id'], 
                                                                            x['network_index'],
                                                                            sigma=0.5), axis=1)
            # add id and domirank columns to new df            
            temp_df = df_filter.loc[:, ['network_index', 'domirank sm sig', 'domirank lrg sig']]
            logger.info(f"Completed adding {row['method_name']} centrality column to the dataframe.")


        if row['method_name'] == "degree":

            # filter df to only graphs that have properties in the list
            # df['has_graph_props'] = df['graphProperties'].apply(lambda x: "True" if row['properties'] in x else "False")
            # df_filter = df[df['has_graph_props'] == "True"]

            # degree centrality scores column 
            df_filter['degree'] = df_filter.apply(lambda x: get_degree_from_edgelist(x['edges_id']), axis=1)

            # add id and degree_centrality columns to new df
            temp_df = df_filter.loc[:, ['network_index', 'degree']]
            logger.info(f"Completed adding {row['method_name']} centrality column to the dataframe.")


        if row['method_name'] == "in_out_degree":

            # filter df to only graphs that have properties in the list
            # df['has_graph_props'] = df['graphProperties'].apply(lambda x: "True" if row['properties'] in x else "False")
            # df_filter = df[df['has_graph_props'] == "True"]

            # add in and out degree centrality scores column 
            df_filter['in_degree'] = df_filter.apply(lambda x: get_in_degree_from_edgelist(x['edges_id']), axis=1)
            df_filter['out_degree'] = df_filter.apply(lambda x: get_out_degree_from_edgelist(x['edges_id']), axis=1)

            # add id and degree_centrality columns to new df
            temp_df = df_filter.loc[:, ['network_index', 'in_degree', 'out_degree']]
            logger.info(f"Completed adding {row['method_name']} centrality column to the dataframe.")


        if row['method_name'] == 'Eigenvector':
            
            # Don't need to filter the dataset
            # add eigenvector centrality measure
            # df_filter = df.copy()
            df_filter['eigenvector'] = df_filter.apply(lambda x: get_eigen_cent_from_edgelist(x['edges_id']), axis=1)

            # add id and eigenvector centrality columns to new df
            temp_df = df_filter.loc[:, ['network_index', 'eigenvector']]
            logger.info(f"Completed adding {row['method_name']} centrality column to the dataframe.")

       
        if row['method_name'] == 'PageRank':

            # No filters needed - PageRank converts undirected graphs to directed graphs
            # add pr centrality measure
            # df_filter = df.copy()
            df_filter['pagerank'] = df_filter.apply(lambda x: get_pageRank(x['edges_id']), axis=1)
            
            # add id and eigenvector centrality columns to new df
            temp_df = df_filter.loc[:, ['network_index', 'pagerank']]
            logger.info(f"Completed adding {row['method_name']} centrality column to the dataframe.")


        if row['method_name'] == 'SpringRank':

            # filter df to only graphs that have properties in the list  
            # df['has_graph_props'] = df['graphProperties'].apply(lambda x: "True" if row['properties'] in x else "False")
            # df_filter = df[df['has_graph_props'] == "True"]

            # add sr centrality measure
            df_filter['springrank'] = df_filter.apply(lambda x: get_springRank(x['edges_id']), axis=1)
            
            # add id and eigenvector centrality columns to new df
            temp_df = df_filter.loc[:, ['network_index', 'springrank']]
            logger.info(f"Completed adding {row['method_name']} centrality column to the dataframe.")


        # outer join temp_df to out_df 
        if isinstance(temp_df, pd.DataFrame):
            out_df = out_df.merge(right=temp_df, how="outer", on="network_index")
        
    out_df.to_pickle(out_file_path + "centralities.pickle")

    return 


if __name__ == "__main__":
    main()