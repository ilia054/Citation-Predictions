import math
import random
import pandas as pd
import networkx as nx
import igraph as ig
import numpy as np
import torch
import torch.nn as nn
import torch_optimizer as optim
import plotly.graph_objects as go
import stellargraph as sg
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec
from stellargraph.data import BiasedRandomWalk
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold


EMBEDDING_DIMENSION = 256
NUM_EPOCHS = 50
BATCH_SIZE = 8
LR = 0.0001
EVALUATION_FREQUENCY = 10
K_FOLD_NUM = 10


#read cora.cites file
file_path = "C:\\Users\\ilia0\\Desktop\\Final Semester\\Cora\\cora\\cora.cites"
metaDataPath= "C:\\Users\\ilia0\\Desktop\\Final Semester\\Cora\\cora\\cora.content"
output_file_path = "C:\\Users\\ilia0\\Desktop\\Final Semester\\Cora\\cora\\output.txt"
data = pd.read_csv(file_path, sep='\t', names=['cited_paper_id', 'citing_paper_id'])
# Create a directed graph from the dataframe
G = nx.from_pandas_edgelist(data, source='citing_paper_id', target='cited_paper_id', create_using=nx.DiGraph())

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, EMBEDDING_DIMENSION), # Adjust to your architecture needs
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION),       # This should match the concatenated vector size
            # Add more layers if necessary
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION // 2),       # Input layer now takes 256-dimensional vector
            nn.LeakyReLU(),
            nn.Linear(EMBEDDING_DIMENSION // 2, EMBEDDING_DIMENSION // 4),
            nn.LeakyReLU(),
            nn.Linear(EMBEDDING_DIMENSION // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.model(features)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.01, save_embeddings=False):
        """
        Args:
            patience (int): Number of epochs to wait after min has been hit.
                            Training will stop if no improvement after this many epochs.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_embeddings (bool): Flag to determine whether to save embeddings.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_embeddings = save_embeddings
        self.best_embeddings = None

    def __call__(self, val_loss, current_embeddings=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.save_embeddings and current_embeddings is not None:
                self.best_embeddings = current_embeddings
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_embeddings and current_embeddings is not None:
                self.best_embeddings = current_embeddings
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print('Early stopping triggered')
                self.early_stop = True
# End of Class
           
def print_graph(g):
  # Get the positions of the nodes using a layout
  layout = g.layout("kk")
  # Extracting the node positions from the layout
  Xn = [layout[k][0] for k in range(len(layout))]
  Yn = [layout[k][1] for k in range(len(layout))]
  # Creating the edge traces
  edge_traces = []
  for edge in g.es:
      x0, y0 = layout[edge.source]
      x1, y1 = layout[edge.target]
      edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                    mode='lines',
                                    line=dict(width=1, color='black'),
                                    hoverinfo='none'
                                    ))

  # Creating the node trace
  node_trace = go.Scatter(x=Xn, y=Yn,
                          mode='markers',
                          marker=dict(size=10, color='blue'),
                          text=[str(k) for k in range(len(g.vs))],
                          hoverinfo='text')

  # Create the figure
  fig = go.Figure(data=edge_traces + [node_trace],
                  layout=go.Layout(showlegend=False, hovermode='closest',
                                  margin=dict(b=0,l=0,r=0,t=0)))

  fig.show()
  
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
        g.es[g.ecount()-1]['weight'] = weight
        final_weights.append(weight)

    coarsened_graph.es['weight']=final_weights

    return coarsened_graph

def print_commuinities_toFile(comm):
  # Write community memberships to a file
  with open("community_memberships.txt", "w") as f:
    for membership in comm.membership:
          f.write(f"{membership}\n")

def readCoraMetaData(file_path, id_mapping):
    """
    Reads the cora.content file and generates feature vectors for each node,
    aligning node IDs with a provided mapping.

    Parameters:
    - file_path: Path to the cora.content file.
    - id_mapping: A dictionary mapping original node IDs to continuous integer IDs.

    Returns:
    - features: A dictionary where keys are new node IDs (based on id_mapping)
                and values are the feature vectors.
    - labels: A dictionary where keys are new node IDs and values are class labels.
    """
    features = {}
    labels = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            original_id = parts[0]  # Original node ID
            feature_vector = [int(x) for x in parts[1:-1]]  # Convert feature values to integers
            label = parts[-1]  # Class label

            # Map the original ID to the new ID
            if original_id in id_mapping:
                new_id = id_mapping[original_id]
                features[new_id] = feature_vector
                labels[new_id] = label
            else:
                print(f"Warning: Node ID {original_id} not found in id_mapping.")

    return features, labels

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

def group_walks_by_length(walks):
    # Initialize a dictionary to hold walks grouped by their length
    grouped_walks = {}
    
    # Iterate over each walk in the input list
    for walk in walks:
        # Determine the length of the current walk
        length = len(walk)
        
        # If this length hasn't been encountered yet, initialize a new list
        if length not in grouped_walks:
            grouped_walks[length] = []
        
        # Append the current walk to the appropriate list based on its length
        grouped_walks[length].append(walk)
    
    # Now, grouped_walks contains walks grouped by length. Let's convert it to a list of lists.
    # Note: This step is optional depending on whether you need the output as a dictionary or a list of lists.
    # The following line extracts only the values (which are lists of walks) and converts them to a list
    grouped_walks_list = list(grouped_walks.values())
    
    return grouped_walks_list

def get_old_id_by_new_id(new_id_from,new_id_to, id_mapping):
    reverse_mapping = {v: k for k, v in id_mapping.items()}
    old_id_from = reverse_mapping.get(new_id_from)
    old_id_to = reverse_mapping.get(new_id_to)
    return old_id_from,old_id_to

def Node2VecAlg(graph):

  #Ensure your graph 'G' is a directed NetworkX graph as you've created with nx.DiGraph()

  # Initialize Node2Vec model. Parameters can be tuned as per your requirement.
  # For directed graphs, set 'walk_length', 'num_walks', and 'workers' as per your computational resources.
  node2vec = Node2Vec(graph, dimensions=128, walk_length=3, num_walks=200, workers=1, p=1, q=1)

  # Train model
  model = node2vec.fit(window=10, min_count=1, batch_words=4)

  # Save embeddings
  model.wv.save_word2vec_format("embeddings.emb")

  # If you want to use the embeddings directly, you can do so like this:
  embeddings = model.wv
  # Extract embeddings into a dictionary
  embeddings_dictionary = {node: model.wv[node] for node in model.wv.key_to_index.keys()}

  return embeddings,embeddings_dictionary

def pairs_to_tensor(pairs, node_embeddings):
    tensors_list = []
    for a, b in pairs:
        # Convert each embedding from numpy.ndarray to a torch tensor
        embedding_a = torch.tensor(node_embeddings[str(a)], dtype=torch.float32)
        embedding_b = torch.tensor(node_embeddings[str(b)], dtype=torch.float32)
        
        # Concatenate the tensors
        concatenated = torch.cat((embedding_a, embedding_b), dim=0)
        tensors_list.append(concatenated)
    
    # Stack all tensors to create a single tensor
    return torch.stack(tensors_list)

def generate_real_neighbor_pairs(graph, embeddings_dict):
    # List to store concatenated embeddings of real neighbor pairs
    real_pairs_list = []

    # Iterate over each edge in the graph
    for edge in graph.edges():
        node_a, node_b = edge
        # Retrieve the embeddings for node_a and node_b
        embedding_a = torch.tensor(embeddings_dict[str(node_a)], dtype=torch.float32)
        embedding_b = torch.tensor(embeddings_dict[str(node_b)], dtype=torch.float32)
        # Concatenate the embeddings
        concatenated = torch.cat((embedding_a, embedding_b), dim=0)
        # Add the concatenated embeddings to the list
        real_pairs_list.append(concatenated)

    # Convert the list of tensors to a single tensor
    real_neighbor_pairs = torch.stack(real_pairs_list)
    return real_neighbor_pairs

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

def print_sample_mapping(node_to_community, sample_size=5):
    print("Sample Node to Community Mapping:")
    # If the graph is small or you want to print all mappings, you can adjust sample_size accordingly
    sample_nodes = random.sample(list(node_to_community.keys()), min(sample_size, len(node_to_community)))
    for node in sample_nodes:
        print(f"Node {node} -> Community {node_to_community[node]}")

def LF_NetLay(graph, feature_vectors):
  # Apply the Infomap algorithm
  communities = graph.community_infomap(edge_weights='weight')

  # Determine the number of unique communities
  num_communities = len(set(communities.membership))

  print(f"Number of communities detected: {num_communities}")
  # Coarsen the graph based on detected communities
  g1 = coarsen_weighted_graph(graph, communities ,graph.es['weight'])
  
  #Create a mapping from nodes to their communities
  node_to_community = {node: comm for node, comm in enumerate(communities.membership)}

  #calculate feature vector for each cluster
  clusters_feature_vectors = compute_cluster_averages_from_list(communities.membership, feature_vectors)

  #print_sample_mapping(node_to_community)
  return g1, node_to_community, clusters_feature_vectors

def normalize_weights(weights, epsilon=1e-4):
    # Find minimum and maximum weights
    min_weight = min(weights)
    max_weight = max(weights)

    # Normalize the weights
    if max_weight > min_weight:
        # Scale weights to [epsilon, 1]
        return [(weight - min_weight) / (max_weight - min_weight) * (1 - epsilon) + epsilon for weight in weights]
    else:
        # If all weights are the same, set them to 1
        return [1 for _ in weights]

def bounded_min_max_normalization(weights, lower_bound=0.1, upper_bound=1):
    min_weight = min(weights)
    max_weight = max(weights)
    scale = upper_bound - lower_bound
    return [lower_bound + (weight - min_weight) / (max_weight - min_weight) * scale for weight in weights]

def resize_feature_vectors(feature_vectors_dict, target_dim=EMBEDDING_DIMENSION // 2):
    """
    Resize all feature vectors using PCA and update the dictionary in place.
    
    Args:
        feature_vectors_dict (dict): A dictionary mapping nodes to their feature vectors.
        target_dim (int): The target dimensionality for the PCA output.
    """
    # Convert dictionary to a matrix (n_samples, n_features)
    nodes, feature_matrix = zip(*feature_vectors_dict.items())
    feature_matrix = np.array(feature_matrix)
    
    # Apply PCA
    pca = PCA(n_components=target_dim)
    resized_feature_matrix = pca.fit_transform(feature_matrix)
    
    # Update the dictionary with resized feature vectors
    for node, resized_feature_vec in zip(nodes, resized_feature_matrix):
        feature_vectors_dict[node] = resized_feature_vec

def pre_train_G(graph, real_neighbor_pairs):
    latent_dim = EMBEDDING_DIMENSION
    generator = Generator(latent_dim)
    # Initialize the RAdam optimizer for the generator
    optimizer_G = optim.RAdam(generator.parameters(), lr=LR)
    # Use MAE (L1 Loss) as the loss function
    loss_function = nn.L1Loss()  
    early_stopping = EarlyStopping(patience=int(graph.number_of_nodes() ** 0.5), min_delta=0.01, save_embeddings=True)

																									
    for epoch in range(NUM_EPOCHS):  # Number of epochs
        # Generate noise vectors
        z = torch.randn(real_neighbor_pairs.size(0), latent_dim, device=real_neighbor_pairs.device)  # Ensure device compatibility

        # Generate fake embeddings from noise
        generated_embeddings = generator(z)

        # Calculate the MAE loss between generated embeddings and real neighbor pairs
        loss_G = loss_function(generated_embeddings, real_neighbor_pairs)

        # Early stopping check
        early_stopping(loss_G.item(), generated_embeddings)  # Pass loss.item() for early stopping
        if early_stopping.early_stop:
            print("Stopping training...")
            break

        # Backpropagation
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if epoch % 10 == 0:  # Log the progress every 10 epochs
            print(f'Epoch {epoch}: Generator Pre-training Loss: {loss_G.item()}')

    return early_stopping.best_embeddings if early_stopping.save_embeddings else None, generator, optimizer_G, loss_function

def pre_train_D(real_neighbor_pairs, fake_embeddings):
    discriminator = Discriminator() 
    optimizer_D = optim.RAdam(discriminator.parameters(), lr=LR)
    loss_function = nn.BCELoss()
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)  # Adjust patience as needed

    # Create labels for real and fake data
    real_labels = torch.ones(real_neighbor_pairs.size(0), 1)
    fake_labels = torch.zeros(fake_embeddings.size(0), 1)
    
    # Combine real and fake embeddings and labels
    combined_embeddings = torch.cat([real_neighbor_pairs, fake_embeddings], dim=0)
    combined_labels = torch.cat([real_labels, fake_labels], dim=0)

    # Pre-training loop
    for epoch in range(NUM_EPOCHS):
        # Forward pass to get discriminator outputs for combined embeddings
        predictions = discriminator(combined_embeddings)

        # Calculate loss
        loss = loss_function(predictions, combined_labels)

        # Backward and optimize
        optimizer_D.zero_grad()
        loss.backward()
        optimizer_D.step()

        # Early stopping check
        early_stopping(loss)
        if early_stopping.early_stop:
            print("Stopping training...")
            break

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}], Loss: {loss.item()}")

    return discriminator, optimizer_D, loss_function
 
def EmbedGAN(networkX_graph, node_embeddings, generator, discriminator, optimizer_G, optimizer_D, loss_G, loss_D):
  
  #Random Walk
  generated_walks = builder_sampling_strategy(networkX_graph)

  #Train GAN
  generator, discriminator = train_GAN(
        graph = networkX_graph,
        walks = generated_walks,
        node_embeddings = node_embeddings,
        generator = generator,
        discriminator = discriminator,
        optimizer_G = optimizer_G,
        optimizer_D = optimizer_D,
        loss_G = loss_G,
        loss_D = loss_D,
        num_epochs = NUM_EPOCHS,
        batch_size = BATCH_SIZE
    )

  return generator, discriminator

def train_GAN(graph, walks, node_embeddings, generator, discriminator, optimizer_G, optimizer_D, loss_G, loss_D, num_epochs, batch_size):
    
    positive_pairs_walks = []
    # Extract positive samples
    for walk in walks:
        start_node, end_node = walk[0], walk[-1]
        positive_pair_walk = (start_node, end_node)
        if start_node != end_node and positive_pair_walk not in positive_pairs_walks:# and graph.has_edge(start_node, end_node):
            positive_pairs_walks.append(positive_pair_walk)

    # Direct link pairs (actual edges in the graph)
    direct_link_pairs = list(graph.edges())
    
    # Combine walk-derived and direct-link pairs=
    combined_positive_pairs = list(set(positive_pairs_walks + direct_link_pairs))
    positive_tensor = pairs_to_tensor(combined_positive_pairs, node_embeddings) # Positive examples tensor

     # Initialize KFold
    kf = KFold(n_splits=K_FOLD_NUM, shuffle=True, random_state=None)
    for fold, (train_index, test_index) in enumerate(kf.split(positive_tensor)):
        train_tensor, test_tensor = positive_tensor[train_index], positive_tensor[test_index]
        for epoch in range(num_epochs):
            # Shuffle positive pairs each epoch
            random_positive_sample_index = torch.randperm(train_tensor.size(0))
            train_tensor = train_tensor[random_positive_sample_index]

            total_batches = train_tensor.size(0) // batch_size

            for batch_num in range(total_batches):
                start_index = batch_num * batch_size
                end_index = start_index + batch_size

                pos_batch = train_tensor[start_index:end_index]

                # Train discriminator on real data
                optimizer_D.zero_grad()
                real_loss = loss_D(discriminator(pos_batch), torch.ones(batch_size, 1))
                
                # Generate fake data
                z = torch.randn(batch_size, EMBEDDING_DIMENSION)  # Adjust the dimension if necessary
                neg_batch = generator(z)
                
                # Train discriminator on fake data
                fake_loss = loss_D(discriminator(neg_batch), torch.zeros(batch_size, 1))
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_D.step()

                # Train generator
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, EMBEDDING_DIMENSION)  # Adjust the dimension if necessary
                gen_data = generator(z)
                g_loss = loss_G(discriminator(gen_data), torch.ones(batch_size, 1))
                g_loss.backward()
                optimizer_G.step()
                if(epoch == NUM_EPOCHS -1):
                    content = f'Epoch: {epoch+1}, Batch: {batch_num+1}/{total_batches}, D_loss: {d_loss.item()} real loss: {real_loss}, fake loss: {fake_loss}, G_loss: {g_loss.item()}\n'
                    file.write(content)
                    #print(f'Epoch: {epoch+1}, Batch: {batch_num+1}/{total_batches}, D_loss: {d_loss.item()} real loss: {real_loss}, fake loss: {fake_loss}, G_loss: {g_loss.item()}')

        # Evaluation phase
        generator.eval()  # Switch to evaluation mode
        discriminator.eval()

        with torch.no_grad():
            #assign labels for evaluation
            evaluation_labels = [1 if (combined_positive_pairs[i] in direct_link_pairs) else 0 for i in test_index]
            # Get predictions from the discriminator for the evaluation pairs
            predictions = discriminator(test_tensor).squeeze()
            # Convert predictions to binary (0 or 1) using 0.5 as a threshold
            binary_predictions = (predictions >= 0.5).long().cpu().numpy()
            # Calculate Precision, Recall, and F1 Score for the evaluation pairs
            precision, recall, f1, _ = precision_recall_fscore_support(evaluation_labels, binary_predictions, average='binary')
            content = f"Fold: {fold+1}, Precision: {precision}, Recall: {recall}, F1: {f1}\n"
            #print(f"Fold: {fold+1}, Precision: {precision}, Recall: {recall}, F1: {f1}")
            file.write(content)
            # Switch back to training mode
            generator.train()
            discriminator.train()
        print(f"Fold[{fold+1}] has been completed")

    return generator, discriminator

def GAHNRL(g, feature_vectors):
  coarsed_graphs = [g]
  graphs_node_to_community = [{}]
  networkx_graphs = []
  feature_vectors_dict_list = [feature_vectors]

  while(len(g.get_edgelist()) > 1):
    G,current_graph_node_mapping,current_graph_feature_vec = LF_NetLay(g, feature_vectors_dict_list[-1])
    if(len(G.get_edgelist()) < 5):
      break
    else:
      coarsed_graphs.append(G)
      graphs_node_to_community.append(current_graph_node_mapping)
      feature_vectors_dict_list.append(current_graph_feature_vec)
      #print(f"Before normalization:{G.es['weight']}")
      normalized_weights = bounded_min_max_normalization(G.es['weight'])
      G.es['weight'] = normalized_weights  # Update the graph with normalized weights
      #print(f"After normalization:{G.es['weight']}")
      g = G

  print(g.summary())
  print(coarsed_graphs)

  for graph in coarsed_graphs:
    networkXg = convert_igraph_to_networkx(graph)
    networkx_graphs.append(networkXg)

  Gn=networkx_graphs[len(networkx_graphs)-1]
  embeddings,embedding_dictionary = Node2VecAlg(Gn)

  # Iterate over each node in the graph
  for node in Gn.nodes():
    # Node2Vec uses string identifiers by default, so ensure consistency in key type
    if node in embeddings:
      # Assign the embedding vector to the node
      Gn.nodes[node]['embedding'] = embeddings[node]


# Generate real neighbor pairs
  real_neighbor_pairs = generate_real_neighbor_pairs(Gn, embedding_dictionary)
  # Pre-train the generator with real neighbor pairs
  fake_embeddings,generator,optimizer_G, loss_G = pre_train_G(Gn,real_neighbor_pairs)
  fake_embeddings_detached = fake_embeddings.detach()

  discriminator, optimizer_D, loss_D = pre_train_D(real_neighbor_pairs, fake_embeddings_detached)

  #add embedding_dictionary as the embeddings for the last layer
  #all other layers are according to the node_to_community mapping
  node_embeddings = embedding_dictionary
  feature_vectors_dict_len = len(feature_vectors_dict_list)
  cnt = 2
  for Gi, node_comm_mapping in zip(reversed(networkx_graphs), reversed(graphs_node_to_community)):
    print(f"Current graph iteration: {cnt-1} out of {len(networkx_graphs)}")
    generator,discriminator = EmbedGAN(
        Gi, 
        node_embeddings, 
        generator, 
        discriminator,
        optimizer_G, 
        optimizer_D, 
        loss_G,
        loss_D)
    
    if node_comm_mapping:
        node_embeddings = introduce_embedding_variations(node_embeddings, feature_vectors_dict_list[feature_vectors_dict_len - cnt], node_comm_mapping)
    cnt = cnt+1
    #feature_vectors_dict_list
    #node_embeddings = assign_community_embeddings(Gi, node_comm_mapping, current_layer_node_embeddings)
        
def calculate_average_shortest_path_length_for_components(graph):

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

	StellarGraph = sg.StellarDiGraph.from_networkx(networkX_Graph)

	# print("The StellaGraph edges are:\n")
	# for edge in StellarGraph.edges():
	# 	print(edge)  

	rw = BiasedRandomWalk(StellarGraph)
	av_path_length = calculate_average_shortest_path_length_for_components(networkX_Graph)
	# Perform the random walks 
	walks = rw.run(
		  nodes=list(StellarGraph.nodes()),  # starting nodes
		  length=av_path_length+1,
		  n=10,
		  p=1,
		  q=1,
		  weighted=True,)
	return filter_invalid_walks(walks,networkX_Graph)

def get_individual_walk_embeddings(walks, node_embeddings):
    individual_embeddings = []
    for walk in walks:
        current_walk = []
        for node in walk:
            current_walk.append(node_embeddings[str(node)])
        individual_embeddings.append(current_walk)
    return individual_embeddings
  
def assign_community_embeddings(graph, node_to_community, community_embeddings):
    """
    Assigns community embeddings to each node in the graph based on their community
    and returns a dictionary where keys are nodes (as strings) and values are embeddings.

    :param graph: networkX graph (layer Gi).
    :param node_to_community: Dictionary mapping each node to its community in layer Gi+1.
    :param community_embeddings: Embeddings of the communities from layer Gi+1 as a PyTorch tensor.
    :return: Dictionary of node embeddings where keys are node identifiers as strings.
    """
    # Initialize the dictionary to store node embeddings
    node_embeddings_dict = {}
    
    for node in graph.nodes():
        # Get the community this node belongs to
        community = node_to_community.get(node)
        
        # Assign the community's embedding to this node
        if community is not None:
            # Assuming community_embeddings is indexed by community ID
            # Convert node to string for the key
            node_embeddings_dict[str(node)] = community_embeddings[community].tolist()  # Convert tensor to list for storage
        else:
            # Handle cases where a node's community might not be in the mapping
            # Convert node to string for the key
            node_embeddings_dict[str(node)] = None
    
    return node_embeddings_dict

def introduce_embedding_variations(embeddings, features, node_to_community_mapping, scale=0.05):
    """
    Assigns community embeddings to nodes and introduces variations to these embeddings based on feature vectors.
    
    Args:
        embeddings (dict): Community embeddings keyed by community ID from graph Gn.
        features (dict): Feature vectors keyed by node ID from graph Gn-1.
        node_to_community_mapping (dict): Mapping of node IDs in Gn-1 to community IDs in Gn.
        scale (float): Scaling factor for variations, controls the "strength" of the variation.
        
    Returns:
        dict: Updated embeddings for nodes in Gn-1 after assigning community embeddings and introducing variations.
    """
    updated_embeddings = {} #Gn-1 embeddings
    
    for node, community in node_to_community_mapping.items():
        # Assign the community embedding to the node
        community_embedding = np.array(embeddings[str(community)])
        # Retrieve the feature vector for this node
        feature_vector = np.array(features[node])
        # Generate variations based on the feature vector and scale
        variation = np.random.normal(loc=0, scale=scale, size=community_embedding.shape)
        # Apply the variation to the community embedding
        updated_embedding = community_embedding + variation * feature_vector
        updated_embeddings[str(node)] = updated_embedding
     
    
    return updated_embeddings

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

# MAIN #
# Read the edgelist file and create a set of unique IDs
unique_ids = set()
with open(file_path, 'r') as file:
    for line in file:
        source, target = line.strip().split()
        unique_ids.update([source, target])

# Create a mapping from original IDs to new continuous integer IDs
id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

# Create a list of edges with the new continuous IDs
edges_with_new_ids = []
with open(file_path, 'r') as file:
    for line in file:
        target, source = line.strip().split()
        new_source = id_mapping[source]
        new_target = id_mapping[target]
        edges_with_new_ids.append((new_source, new_target))

# Now create the graph with the correctly mapped edges
g = ig.Graph(edges=edges_with_new_ids, directed=True)

print(g.summary())

feature_vectors, node_labels = readCoraMetaData(metaDataPath, id_mapping)
edge_weights,edges = calculate_edge_weights(g,feature_vectors)

resize_feature_vectors(feature_vectors)

# and edge_weights
g.es['weight'] = edge_weights

file = open(output_file_path, 'w')

#g is of type igraph
GAHNRL(g, feature_vectors)