import numpy as np
import torch
import config as con
import models
import graphUtilities as gutil
from node2vec import Node2Vec
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold

def Node2VecAlg(graph):

  #Ensure that graph 'G' is a directed NetworkX graph as created with nx.DiGraph()
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

def LF_NetLay(graph, feature_vectors):
  # Apply the Infomap algorithm
  communities = graph.community_infomap(edge_weights='weight')

  # Determine the number of unique communities
  num_communities = len(set(communities.membership))

  # Coarsen the graph based on detected communities
  g1 = gutil.coarsen_weighted_graph(graph, communities ,graph.es['weight'])
  print(f"Number of communities detected: {num_communities}, Number of edges detected: {g1.ecount()}")
  
  #Create a mapping from nodes to their communities
  node_to_community = {node: comm for node, comm in enumerate(communities.membership)}

  #calculate feature vector for each cluster
  clusters_feature_vectors = gutil.compute_cluster_averages_from_list(communities.membership, feature_vectors)

  return g1, node_to_community, clusters_feature_vectors

def bounded_min_max_normalization(weights, lower_bound=0.1, upper_bound=1):
    min_weight = min(weights)
    max_weight = max(weights)
    scale = upper_bound - lower_bound
    return [lower_bound + (weight - min_weight) / (max_weight - min_weight) * scale for weight in weights]

def resize_feature_vectors(feature_vectors_dict, target_dim=con.EMBEDDING_DIMENSION // 2):
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

def EmbedGAN(networkX_graph, node_embeddings, generator, discriminator, optimizer_G, optimizer_D, loss_G, loss_D, output_file):
  
  #Random Walk
  generated_walks = gutil.builder_sampling_strategy(networkX_graph)

  #Train GAN
  generator, discriminator, optimizer_G, optimizer_D = train_GAN(
        graph = networkX_graph,
        walks = generated_walks,
        node_embeddings = node_embeddings,
        generator = generator,
        discriminator = discriminator,
        optimizer_G = optimizer_G,
        optimizer_D = optimizer_D,
        loss_G = loss_G,
        loss_D = loss_D,
        NUM_EPOCHS = con.NUM_EPOCHS,
        BATCH_SIZE = con.BATCH_SIZE,
        output_file = output_file
    )

  return generator, discriminator, optimizer_G, optimizer_D

def train_GAN(graph, walks, node_embeddings, generator, discriminator, optimizer_G, optimizer_D, loss_G, loss_D, NUM_EPOCHS, BATCH_SIZE, output_file):
    
    fold_num = con.K_FOLD_NUM
    positive_pairs_walks = []
    # Extract positive samples
    for walk in walks:
        start_node, end_node = walk[0], walk[-1]
        positive_pair_walk = (start_node, end_node)
        if start_node != end_node and positive_pair_walk not in positive_pairs_walks:
            positive_pairs_walks.append(positive_pair_walk)

    # Direct link pairs (actual edges in the graph)
    direct_link_pairs = list(graph.edges())
    # Combine walk-derived and direct-link pairs
    combined_positive_pairs = list(set(positive_pairs_walks + direct_link_pairs))
    # Positive examples tensor
    positive_tensor = pairs_to_tensor(combined_positive_pairs, node_embeddings)

    #  # Initialize KFold
    if(graph.number_of_nodes()> 2000):
        fold_num = 10

    kf = KFold(n_splits=fold_num, shuffle=True, random_state=None)
    for fold, (train_index, test_index) in enumerate(kf.split(positive_tensor)):
        train_tensor, test_tensor = positive_tensor[train_index], positive_tensor[test_index]
        for epoch in range(con.NUM_EPOCHS):
            # Shuffle positive pairs each epoch
            random_positive_sample_index = torch.randperm(train_tensor.size(0))
            train_tensor = train_tensor[random_positive_sample_index]

            total_batches = train_tensor.size(0) // con.BATCH_SIZE
            avarage_metrics = [0,0,0,0] #Dloss,Dreal,Dfake,Gloss
            for batch_num in range(total_batches):
                start_index = batch_num * con.BATCH_SIZE
                end_index = start_index + con.BATCH_SIZE

                pos_batch = train_tensor[start_index:end_index]

                # Train discriminator on real data
                optimizer_D.zero_grad()
                real_loss = loss_D(discriminator(pos_batch), torch.ones(con.BATCH_SIZE, 1))
                
                # Generate fake data
                z = torch.randn(con.BATCH_SIZE, con.EMBEDDING_DIMENSION)
                neg_batch = generator(z)
                
                # Train discriminator on fake data
                fake_loss = loss_D(discriminator(neg_batch), torch.zeros(con.BATCH_SIZE, 1))
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_D.step()

                # Train generator
                optimizer_G.zero_grad()
                z = torch.randn(con.BATCH_SIZE, con.EMBEDDING_DIMENSION)
                gen_data = generator(z)
                g_loss = loss_G(discriminator(gen_data), torch.ones(con.BATCH_SIZE, 1))
                g_loss.backward()
                optimizer_G.step()
                avarage_metrics[0] = avarage_metrics[0] + d_loss.item()
                avarage_metrics[1] = avarage_metrics[1] + real_loss
                avarage_metrics[2] = avarage_metrics[2] + fake_loss
                avarage_metrics[3] = avarage_metrics[3] + g_loss.item()

            content = f'Epoch: {epoch+1}, D_loss: {avarage_metrics[0] / total_batches} real loss: {avarage_metrics[1] / total_batches}, fake loss: {avarage_metrics[2] / total_batches}, G_loss: {avarage_metrics[3] / total_batches}\n'
            output_file.write(content)

        # Evaluation phase
        generator.eval()  # Switch to evaluation mode
        discriminator.eval()

        with torch.no_grad():
            #assign labels for evaluation
            evaluation_labels = [1 if (combined_positive_pairs[i] in direct_link_pairs) else 0 for i in test_index]
            pos = 0
            neg = 0
            for i in evaluation_labels:
                if i == 1:
                    pos = pos + 1
                else:
                    neg = neg + 1

            ratio = neg / len(evaluation_labels)
            # Get predictions from the discriminator for the evaluation pairs
            predictions = discriminator(test_tensor).squeeze()
            # Convert predictions to binary (0 or 1) using 0.5 as a threshold
            binary_predictions = (predictions >= 0.5).long().cpu().numpy()
            # Calculate Precision, Recall, and F1 Score for the evaluation pairs
            precision, recall, f1, _ = precision_recall_fscore_support(evaluation_labels, binary_predictions, average='binary')
            content = f"Fold: {fold+1}, Precision: {precision}, Recall: {recall}, F1: {f1}\n"
            output_file.write(content)
            
            # Switch back to training mode
            generator.train()
            discriminator.train()
        print(f"Fold[{fold+1}] has been completed")

    return generator, discriminator, optimizer_G, optimizer_D

def introduce_embedding_variations(embeddings, features, node_to_community_mapping, scale=0.05):
    """
    Assigns community embeddings to nodes and introduces variations to these embeddings based on feature vectors.
    
    Args:
        embeddings (dict): Community embeddings keyed by community ID from graph Gi.
        features (dict): Feature vectors keyed by node ID from graph Gi-1.
        node_to_community_mapping (dict): Mapping of node IDs in Gi-1 to community IDs in Gi.
        scale (float): Scaling factor for variations, controls the "strength" of the variation.
        
    Returns:
        dict: Updated embeddings for nodes in Gn-1 after assigning community embeddings and introducing variations.
    """
    updated_embeddings = {} #Gi-1 embeddings
    
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

def GAHNRL(g, feature_vectors, id_mapping, output_file, predictions_file):
    #graph layers of type igraph in the Heirarchial network
    coarsed_graphs = [g]
    #mapping of nodes in graph Gi to cluster in Gi+1
    graphs_node_to_community = [{}]
    #graph layers of type networkX in the Heirarchial network
    networkx_graphs = []
    #feature vectors list for each layer in the Heirarchial network
    feature_vectors_dict_list = [feature_vectors]

    while(len(g.get_edgelist()) > 1):
        #Layer the graph
        G,current_graph_node_mapping,current_graph_feature_vec = LF_NetLay(g, feature_vectors_dict_list[-1])
        if(len(G.get_edgelist()) < 5):
            break
        else:
            #store curret layer
            coarsed_graphs.append(G)
            graphs_node_to_community.append(current_graph_node_mapping)
            feature_vectors_dict_list.append(current_graph_feature_vec)
            #normalize weights
            normalized_weights = bounded_min_max_normalization(G.es['weight'])
            G.es['weight'] = normalized_weights  # Update the graph with normalized weights
            g = G

    #convert all layers in the Heirarchial network from igraph to networkX
    for graph in coarsed_graphs:
        networkXg = gutil.convert_igraph_to_networkx(graph)
        networkx_graphs.append(networkXg)

    #achieve initial embeddings for layer Gn
    Gn=networkx_graphs[len(networkx_graphs)-1]
    embeddings,embedding_dictionary = Node2VecAlg(Gn)

	# Iterate over each node in the graph
    for node in Gn.nodes():
        # Node2Vec uses string identifiers by default, so ensure consistency in key type
        if node in embeddings:
            # Assign the embedding vector to the node
            Gn.nodes[node]['embedding'] = embeddings[node]

	# Generate real neighbor pairs (embeddings of all edges in the graph) for final layer
    real_neighbor_pairs = generate_real_neighbor_pairs(Gn, embedding_dictionary)
    
	# Pre-train the generator with real neighbor pairs
    fake_embeddings,generator,optimizer_G, loss_G = models.pre_train_G(Gn,real_neighbor_pairs)
    fake_embeddings_detached = fake_embeddings.detach()

    #pre-train the discriminator with real neighbor pairs and fake embeddings from the generator
    discriminator, optimizer_D, loss_D = models.pre_train_D(real_neighbor_pairs, fake_embeddings_detached)

	#add embedding_dictionary as the embeddings for the last layer
	#all other layers are according to the node_to_community mapping
    node_embeddings = embedding_dictionary
    feature_vectors_dict_len = len(feature_vectors_dict_list)
    cnt = 2
    #Embed GAN recursive loop
    for Gi, node_comm_mapping in zip(reversed(networkx_graphs), reversed(graphs_node_to_community)):
        print(f"Current graph Layer: {cnt-1} out of {len(networkx_graphs)}")
        generator,discriminator,optimizer_G,optimizer_D = EmbedGAN(
            Gi, 
            node_embeddings, 
            generator, 
            discriminator,
            optimizer_G, 
            optimizer_D, 
            loss_G,
            loss_D,
            output_file)
        #add variations to node embeddings for next layer
        if node_comm_mapping:
            node_embeddings = introduce_embedding_variations(node_embeddings, feature_vectors_dict_list[feature_vectors_dict_len - cnt], node_comm_mapping)
        cnt = cnt+1

    # Generate real neighbor pairs (embeddings of all edges in the graph) for original graph
    neighbor_pairs = generate_real_neighbor_pairs(networkx_graphs[0], node_embeddings)
    
    edge_sampled = make_final_predictions(neighbor_pairs, 
                                          discriminator,
                                          list(networkx_graphs[0].edges()),
                                          id_mapping, 
                                          predictions_file)
    
    return edge_sampled

def make_final_predictions(neighbor_pairs, discriminator,edge_list, id_mapping, predictions_file):
    content = f"Start final prediction phase"
    predictions_file.write(content+ "\n")
    print(content)

    edge_sampled = {}

	# Reverse the id_mapping to get from new IDs back to original IDs
    reverse_id_mapping = {new_id: old_id for old_id, new_id in id_mapping.items()}

     # Calculate the number of pairs to select
    num_pairs = neighbor_pairs.shape[0]
    num_select = int(con.PREDICTION_PRECENTAGE * num_pairs)

    # Shuffle the indices and select the first 30%
    shuffled_indices = torch.randperm(num_pairs)
    selected_indices = shuffled_indices[:num_select]

    # Extract the selected pairs
    selected_pairs_tensor = neighbor_pairs[selected_indices] #size num_select

    predictions = discriminator(selected_pairs_tensor).squeeze()
    # Convert predictions to binary (0 or 1) using 0.5 as a threshold
    binary_predictions = (predictions >= 0.5).long().cpu().numpy()

    num_ones = np.sum(binary_predictions == 1)
    num_zeros = np.sum(binary_predictions == 0)

    print(f"Number of relevant connections: {num_ones}")
    print(f"Number of irrelevant: {num_zeros}")
    print(f"Success Rate: {num_ones / (num_zeros + num_ones)}")

    content = f"Number of relevant connections: {num_ones}\nNumber of irrelevant: {num_zeros}\nSuccess Rate: {num_ones / (num_zeros + num_ones)}\n"
    predictions_file.write(content)

    # Iterate over selected pairs and their predictions
    for idx, (edge_index, prediction) in enumerate(zip(selected_indices, binary_predictions)):
        # Get the original IDs for the selected pair
        a_new, b_new = edge_list[edge_index]  # Assuming tensor to list conversion works here
        a, b = reverse_id_mapping[a_new], reverse_id_mapping[b_new]
        
        # Initialize or update the sampled count
        if (a, b) not in edge_sampled:
            edge_sampled[(a, b)] = [1, 0]  # Initialize with 1 sample, 0 successes
        else:
            edge_sampled[(a, b)][0] +=1  # Increment sample count
        
        # Update success count based on prediction
        if prediction == 1:  # Success
            edge_sampled[(a, b)][1] +=1  # Increment success count
            
    content = f"End final prediction phase\n"
    predictions_file.write(content)
    print(content)
    return edge_sampled

def LinkPrediction(iteration, original_graph, feature_vectors, id_mapping):
    #prepare files for next iteration
    output_file_path_run = con.output_file_path + f"{iteration}.txt"
    predictions_results_file_path_run = con.predictions_results_file_path + f"{iteration}.txt"
    
    output_file = open(output_file_path_run, 'w')
    predictions_file = open(predictions_results_file_path_run, 'w')

    #graph type is igraph
    edge_sampled = GAHNRL(original_graph, feature_vectors, id_mapping, output_file, predictions_file)

    output_file.close()
    predictions_file.close()

    return edge_sampled