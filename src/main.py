import pandas as pd
import networkx as nx
import igraph as ig
import pandas as pd
import config as con
import utils
import graphUtilities as gutils
import CitationPrediction as cp

def main(): 
    # GLOBALS #
    data = pd.read_csv(con.coraGraphFile_path, sep='\t', names=['cited_paper_id', 'citing_paper_id'])
    # Create a directed graph from the dataframe
    G = nx.from_pandas_edgelist(data, source='citing_paper_id', target='cited_paper_id', create_using=nx.DiGraph())
    # Initialize the dictionaries
    edge_sampled = {}  # Key: (source, destination), Value: (sampled_count, success_count)
    unique_ids = set()
    with open(con.coraGraphFile_path, 'r') as file:
        for line in file:
            source, target = line.strip().split()
            unique_ids.update([source, target])

    # Create a mapping from original IDs to new continuous integer IDs
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

    # Create a list of edges with the new continuous IDs
    edges_with_new_ids = []
    with open(con.coraGraphFile_path, 'r') as file:
        for line in file:
            target, source = line.strip().split()
            new_source = id_mapping[source]
            new_target = id_mapping[target]
            edges_with_new_ids.append((new_source, new_target))

    # Now create the graph with the correctly mapped edges
    g = ig.Graph(edges=edges_with_new_ids, directed=True)
    feature_vectors, original_ID_node_field = utils.readCoraMetaData(con.metaDataPath, id_mapping)
    edge_weights,edges = gutils.calculate_edge_weights(g,feature_vectors)

    cp.resize_feature_vectors(feature_vectors)
    g.es['weight'] = edge_weights

    for iteration in range (con.NUM_RUNS):
        print(f"Training Iteartion Number {iteration+1}")
        current_edges_sampled = cp.LinkPrediction(iteration+1, g, feature_vectors, id_mapping)
        # Iterate through the current edges sampled and accumulate the counts
        for edge, (current_sampled, current_success) in current_edges_sampled.items():
            if edge in edge_sampled:
                # If the edge already exists, add the new counts to the existing counts
                edge_sampled[edge] = (edge_sampled[edge][0] + current_sampled, edge_sampled[edge][1] + current_success)
            else:
                # If the edge does not exist, initialize it with the current counts
                edge_sampled[edge] = (current_sampled, current_success)

    # After all runs, calculate success rates
    for edge, counts in edge_sampled.items():
        sampled_count = counts[0]
        successes = counts[1]
        success_rate = successes / sampled_count
        edge_sampled[edge] = (sampled_count, success_rate)
  
    # Generate the final excel sheet for the results:
    utils.create_excel(edge_sampled, original_ID_node_field)

    utils.plotResult()

if __name__ == "__main__":
  main()