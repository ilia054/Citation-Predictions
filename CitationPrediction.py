import pandas as pd
import networkx as nx
import igraph as ig
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
import stellargraph as sg
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec
from torch_geometric.data import Data
from stellargraph.data import BiasedRandomWalk
from vose_sampler import VoseAlias

#read cora.cites file
file_path = "C:\\Users\\ilia0\\Desktop\\Final Semester\\Cora\\cora\\cora.cites"
metaDataPath="C:\\Users\\ilia0\\Desktop\\Final Semester\\Cora\\cora\\cora.content"
data = pd.read_csv(file_path, sep='\t', names=['cited_paper_id', 'citing_paper_id'])

# Create a directed graph from the dataframe
G = nx.from_pandas_edgelist(data, source='citing_paper_id', target='cited_paper_id', create_using=nx.DiGraph())

# Class Declerations:
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        # Define your network layers here
        self.fc = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        # Define the forward pass
        return self.fc(z)
class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output probability of being a real citation
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
    final_weights = [];
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

  return embeddings

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

def convert_networkx_to_pyg(graph):
    # Convert NetworkX graph to edge list and node attributes (embeddings)
    edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
    x = None

    node_embeddings = nx.get_node_attributes(graph, 'embedding')
    if node_embeddings:
        # If embeddings are present, prepare the node feature matrix 'x'
        embeddings = list(node_embeddings.values())  # Get the embeddings as a list
        x = torch.tensor(embeddings, dtype=torch.float)


    # If the graph has edge weights, prepare the edge attribute matrix 'edge_attr'
    edge_attr = None
    if nx.is_weighted(graph):
        edge_weights = [graph[u][v]['weight'] for u, v in graph.edges()]
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def LF_NetLay(graph):
  # Apply the Infomap algorithm
  communities = graph.community_infomap(edge_weights='weight')

  # Determine the number of unique communities
  num_communities = len(set(communities.membership))

  print(f"Number of communities detected: {num_communities}")

  g1 = coarsen_weighted_graph(graph,communities,graph.es['weight'])
  return g1

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

def print_pyg_graph_summary(data):
    print("Graph Summary:")
    # The number of nodes can be inferred from the size of the x tensor if it exists
    num_nodes = data.num_nodes if data.num_nodes is not None else data.x.size(0) if data.x is not None else 0
    print(f"- Number of nodes: {num_nodes}")

    # The number of edges is half the size of the edge_index tensor (since it's [2, num_edges])
    print(f"- Number of edges: {data.edge_index.size(1)}")

    # Check if the graph has node features and print them
    if data.x is not None:
        print("- Node Embeddings:")
        for idx, embedding in enumerate(data.x):
            print(f"  Node {idx}: Embedding {embedding.tolist()}")
    else:
        print("- Node feature matrix: Not present")

    # Optionally, print a sample of edges
    print("- Sample Edges (first 5):")
    for i in range(min(data.edge_index.size(1), 5)):
        edge = data.edge_index[:, i]
        print(f"  Edge from Node {edge[0].item()} to Node {edge[1].item()}")

def pre_train_G(pyg_graph):

  # Instantiate the Generator
  latent_dim = pyg_graph.x.shape[1]  # Size of the noise vector
  pretrain_epochs = 100 # full passes through the dataset
  output_dim = pyg_graph.x.shape[1]  # Size of Node2Vec embeddings'
  generator = Generator(latent_dim, output_dim)

  # Define the optimizer
  optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.01)

  # Loss function could be Mean Squared Error or similar
  loss_function = nn.MSELoss()

  # Assuming pyg_graph is your PyTorch Geometric graph object
  number_of_vertices = pyg_graph.num_nodes # This gets the number of nodes (vertices) in the graph

  # Calculate the square of the number of vertices
  square_of_vertices = int(number_of_vertices ** 0.5)



  early_stopping = EarlyStopping(patience=square_of_vertices, min_delta=0.01, save_embeddings=True)

  # Pre-training the Generator
  for epoch in range(pretrain_epochs):
      # Generate random noise vectors
      z = torch.randn(pyg_graph.x.size(0), latent_dim)

      # Generate fake embeddings from noise
      generated_embeddings = generator(z)
      # Calculate the loss based on how close the generated embeddings are to the Node2Vec ones
      loss_G = loss_function(generated_embeddings, pyg_graph.x)

      early_stopping(loss_G, generated_embeddings)
      if early_stopping.early_stop:
        print("Stopping training...")
        break

      # Backpropagation
      optimizer_G.zero_grad()
      loss_G.backward()
      optimizer_G.step()

      # Log the progress
      print(f'Epoch {epoch}: Generator Pre-training Loss: {loss_G.item()}')

  return early_stopping.best_embeddings

def pre_train_D(pyg_graph, fake_embeddings):

  # Combine real and fake embeddings and labels
  real_embeddings = pyg_graph.x
  combined_embeddings = torch.cat([real_embeddings, fake_embeddings], dim=0)
  real_labels = torch.ones(real_embeddings.size(0), 1)
  fake_labels = torch.zeros(fake_embeddings.size(0), 1)
  combined_labels = torch.cat([real_labels, fake_labels], dim=0)

  feature_dim = pyg_graph.x.shape[1]
  discriminator = Discriminator(feature_dim)
  pretrain_epochs = 100 # full passes through the dataset
  optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
  loss_function = nn.BCELoss()

  # Assuming pyg_graph is your PyTorch Geometric graph object
  number_of_vertices = pyg_graph.num_nodes # This gets the number of nodes (vertices) in the graph

  # Calculate the square of the number of vertices
  square_of_vertices = int(number_of_vertices ** 0.5)

  early_stopping = EarlyStopping(patience=square_of_vertices, min_delta=0.01)

  # Pre-training loop
  for epoch in range(pretrain_epochs):
      # Forward pass
      predictions = discriminator(combined_embeddings)
      #print(f"predictions type {type(predictions)}, value: {predictions}")
      loss = loss_function(predictions, combined_labels)


      # Backward and optimize
      optimizer.zero_grad()
      loss.backward(retain_graph=True)
      optimizer.step()

      if epoch % 10 == 0:
          print(f"Epoch [{epoch}/{pretrain_epochs}], Loss: {loss.item()}")

      early_stopping(loss)
      if early_stopping.early_stop:
          print("Stopping training...")
          break

def EmbedGAN(pyg_graph,networkX_graph):
  
  fake_embeddings = pre_train_G(pyg_graph)

  # Assuming fake_embeddings is the output from the pre-trained generator
  fake_embeddings_detached = fake_embeddings.detach()
  pre_train_D(pyg_graph, fake_embeddings_detached)

  #Random Walk && Alias Method
  builder_sampling_strategy(networkX_graph)

def GAHNRL(g):
  coarsed_graphs = [g]
  networkx_graphs = []
  while(len(g.get_edgelist()) > 1):
    G = LF_NetLay(g)
    if(len(G.get_edgelist()) < 2):
      break
    else:
      coarsed_graphs.append(G)
      #print(f"Before normalization:{G.es['weight']}")
      normalized_weights = bounded_min_max_normalization(G.es['weight']);
      G.es['weight'] = normalized_weights  # Update the graph with normalized weights
      #print(f"After normalization:{G.es['weight']}")
      g = G

  print(g.summary())
  print(coarsed_graphs)

  for graph in coarsed_graphs:
    networkXg = convert_igraph_to_networkx(graph)
    networkx_graphs.append(networkXg)

  Gn=networkx_graphs[len(networkx_graphs)-1]
  embeddings = Node2VecAlg(Gn)

  # Iterate over each node in the graph
  for node in Gn.nodes():
    # Node2Vec uses string identifiers by default, so ensure consistency in key type
    if node in embeddings:
      # Assign the embedding vector to the node
      Gn.nodes[node]['embedding'] = embeddings[node]

  pyg_graph = convert_networkx_to_pyg(Gn)

  EmbedGAN(pyg_graph,Gn)

def builder_sampling_strategy(networkX_Graph):
	StellarGraph = sg.StellarDiGraph.from_networkx(networkX_Graph)
	rw = BiasedRandomWalk(StellarGraph)

	# Perform the random walks 
	walks = rw.run(
		  nodes=list(StellarGraph.nodes()),  # starting nodes
		  length=5,
		  n=10,
		  p=1,
		  q=1,
		  weighted=True,)

	# Assuming 'walks' is the list of walks generated by BiasedRandomWalk
	print(f"Total walks generated: {len(walks)}")

	# Print the length of the first few walks
	for i, walk in enumerate(walks[:5]):
		print(f"Length of walk {i+1}: {len(walk)}")

	# Print the first few walks to see their paths
	print(f"Example walks and amount :")
	for walk in walks:
		print(walk)

  # Convert the walks into a form suitable for the Alias Method
  # This involves calculating transition probabilities for each node
  # Let's say you have a function that does this:
#   transition_probs = calculate_transition_probabilities(walks)

#    # Now, for each node, use Vose-Alias-Method to create alias tables
#    alias_tables = {}
#    for node, probs in transition_probs.items():
#        alias_tables[node] = VoseAlias(probs)

#    # Now you can efficiently sample the next node in the walk using the alias tables
#    # Here's how you might do a single step of a walk from a given node
#    current_node = 'some_node'
#    next_node = alias_tables[current_node].alias_draw()
  
  


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

# Assuming g is your igraph Graph object
# and edge_weights is the list of weights you've calculated
g.es['weight'] = edge_weights

GAHNRL(g)