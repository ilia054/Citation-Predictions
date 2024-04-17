import math
import networkx as nx
import igraph as ig
import numpy as np
import stellargraph as sg
from sklearn.metrics.pairwise import cosine_similarity
from stellargraph.data import BiasedRandomWalk

def coarsen_weighted_graph(original_graph, communities, edge_weights):
    # Determine the unique communities and create a mapping for super-nodes
    unique_communities = set(communities.membership)
    community_to_super_node = {community: i for i, community in enumerate(unique_communities)}
    final_weights = []
    # Create a new igraph graph for the coarsened version
    coarsened_graph = ig.Graph(directed=True)
    coarsened_graph.add_vertices(len(unique_communities))

    # Initialize a dictionary to sum weights for edges between different communities
    coarsened_edges_weights = {}

    for edge in original_graph.es:
        source_community = communities.membership[edge.source]
        target_community = communities.membership[edge.target]

        # Proceed only if source and target are in different communities
        if source_community != target_community:
            edge_weight = edge_weights[edge.index]

            super_node_source = community_to_super_node[source_community]
            super_node_target = community_to_super_node[target_community]

            # Sum weights for the same super-node edge
            if (super_node_source, super_node_target) not in coarsened_edges_weights:
                coarsened_edges_weights[(super_node_source, super_node_target)] = edge_weight
            else:
                coarsened_edges_weights[(super_node_source, super_node_target)] += edge_weight

    # Add aggregated edges and their summed weights to the coarsened graph
    for (source, target), weight in coarsened_edges_weights.items():
        coarsened_graph.add_edge(source, target)
        final_weights.append(weight)

    coarsened_graph.es['weight']=final_weights

    return coarsened_graph

def calculate_edge_weights(graph, features):
    """
    Calculates weights for each edge in the graph based on feature vector similarity,
    adjusting the cosine similarity to be within the range [0, 1].

    Parameters:
    - graph: An igraph Graph object.
    - features: A dictionary mapping node IDs to feature vectors.

    Returns:
    - weights: A list of weights corresponding to each edge in the graph.
    - cites: A list of the nodes that have the edged connecting them.
    """
    weights = []
    cites = []
    for edge in graph.es:
        source_id = edge.source
        target_id = edge.target
        edge_dict = {source_id, target_id}
        cites.append(edge_dict)

        # Retrieve the feature vectors for the source and target nodes
        source_vector = np.array(features[source_id]).reshape(1, -1)
        target_vector = np.array(features[target_id]).reshape(1, -1)

        # Calculate cosine similarity and adjust it to be in the range [0, 1]
        similarity = cosine_similarity(source_vector, target_vector)[0][0]
        weight = (similarity + 1) / 2  # Adjusting the range
        weights.append(weight)

    return weights, cites

def convert_igraph_to_networkx(igraph_graph):
    """
    Convert an igraph graph to a NetworkX graph, including edge weights.

    Parameters:
    - igraph_graph: The igraph graph to convert.

    Returns:
    - A NetworkX graph.
    """
    # Initialize a directed NetworkX graph
    nx_graph = nx.DiGraph()

    # Add nodes with attributes
    for vertex in igraph_graph.vs:
        nx_graph.add_node(vertex.index, **vertex.attributes())

    # Add edges with weights
    for edge in igraph_graph.es:
        source, target = edge.tuple
        # Check if the edge has a 'weight' attribute; if not, default to 1.0
        weight = edge['weight']
        nx_graph.add_edge(source, target, weight=weight)

    return nx_graph

def calculate_average_shortest_path_length_for_components(graph):
    """
    This function calculates the avarage length of all the shortest paths in the graph.

    """
    if nx.is_directed(graph):
        components = nx.strongly_connected_components(graph)   

    avg_lengths = []

    for component in components:
        subgraph = graph.subgraph(component)
        if len(subgraph) > 1:  # Ensuring the component has more than one node
            avg_length = nx.average_shortest_path_length(subgraph)
            avg_lengths.append(avg_length)
    
    if avg_lengths:  # Check if the list is not empty
        overall_avg_length = np.mean(avg_lengths)
        return math.ceil(overall_avg_length)
    else:
        return 1
    
def filter_invalid_walks(walks, graph):
    """
    Filter walks that include steps not present in the directed graph.
    
    Args:
    walks (list of lists): The generated walks to be filtered.
    graph (nx.DiGraph): The directed graph used for validating walks.
    
    Returns:
    list of lists: Filtered walks that only include valid steps.
    """
    valid_walks = []
    for walk in walks:
        valid_walk = [walk[0]]  # Start each walk with its first node
        for i in range(1, len(walk)):
            # If the step from walk[i-1] to walk[i] exists in the graph, keep it
            if graph.has_edge(valid_walk[-1], walk[i]):
                valid_walk.append(walk[i])
            else:
                # Optional: break here if you want to discard the rest of the walk
                # as soon as an invalid step is found
                # break
                pass  # Or continue without breaking to try to keep valid segments
        if len(valid_walk) > 1 and valid_walk not in valid_walks:
            valid_walks.append(valid_walk)
    return valid_walks

def builder_sampling_strategy(networkX_Graph):

    #create a directed graph of type StellarGraph
	StellarGraph = sg.StellarDiGraph.from_networkx(networkX_Graph)
	rw = BiasedRandomWalk(StellarGraph)
    #calculate length of the avarage shortest paths in the graph
	av_path_length = calculate_average_shortest_path_length_for_components(networkX_Graph)
	# Perform the random walks 
	walks = rw.run(
		  nodes=list(StellarGraph.nodes()),  # starting nodes
		  length=av_path_length+1,
		  n=10,
		  p=1,
		  q=1,
		  weighted=True,)
    
    #filter walks to take only valid ones, because random walk doesnt consider the direction of the graph, and generates walks in length of 1
	return filter_invalid_walks(walks,networkX_Graph)

def compute_cluster_averages_from_list(membership, features):
    """
    Computes the average feature vector for each cluster using a membership list.
    
    Args:
        membership (list): A list where the index represents the node ID and the value at each index is the cluster ID.
        features (dict): A mapping from node ID (as integer) to feature vector.
        
    Returns:
        dict: A mapping from cluster ID to its average feature vector.
    """
    cluster_sums = {}
    cluster_counts = {}

    # Iterate over each node and its cluster membership
    for node_id, cluster_id in enumerate(membership):
        # Convert node_id to int if necessary (depends on how features keys are represented)
        node_features = np.array(features[node_id], dtype=np.float64)

        if cluster_id not in cluster_sums:
            cluster_sums[cluster_id] = node_features
            cluster_counts[cluster_id] = 1
        else:
            cluster_sums[cluster_id] += node_features
            cluster_counts[cluster_id] += 1

    # Compute the average feature vector for each cluster
    cluster_averages = {cluster: cluster_sums[cluster] / cluster_counts[cluster] for cluster in cluster_sums}
    
    return cluster_averages
