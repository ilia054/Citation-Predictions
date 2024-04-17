import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import math
import config as conf

def create_excel(edge_sampled, original_ID_node_field):
    # Transform the edge_sampled dict into a list of lists, each sublist containing the data for one row
    data = [[source, 
             destination, 
             original_ID_node_field[source], 
             original_ID_node_field[destination], 
             sampled_count, 
             success_percentage] for (source, destination), (sampled_count, success_percentage) in edge_sampled.items()]
    
    # Create a DataFrame with the data and the specified column names
    df = pd.DataFrame(data, columns=['Source Node', 'Destination Node','Source Field', 'Destination Field','Sampled Amount', 'Success Rate'])
    
    # Save the DataFrame to an Excel file
    df.to_excel('Resources/output/linkPredictionResults.xlsx', index=False)

def readCoraMetaData(coraGraphFile_path, id_mapping):
    """
    Reads the cora.content file and generates feature vectors for each node,
    aligning node IDs with a provided mapping.

    Parameters:
    - coraGraphFile_path: Path to the cora.content file.
    - id_mapping: A dictionary mapping original node IDs to continuous integer IDs.

    Returns:
    - features: A dictionary where keys are new node IDs (based on id_mapping)
                and values are the feature vectors.
    - labels: A dictionary where keys are new node IDs and values are class labels.
    """
    features = {}
    labels = {}

    with open(coraGraphFile_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            original_id = parts[0]  # Original node ID
            feature_vector = [int(x) for x in parts[1:-1]]  # Convert feature values to integers
            label = parts[-1]  # Class label

            # Map the original ID to the new ID
            if original_id in id_mapping:
                new_id = id_mapping[original_id]
                features[new_id] = feature_vector
                labels[original_id] = label
            else:
                print(f"Warning: Node ID {original_id} not found in id_mapping.")

    return features, labels

def to_percentage_string(value):
    # Round up to the nearest whole number and convert to integer
    ceiling_value = math.ceil(value * 100)
    # Format as a percentage string
    return f"{ceiling_value}%"

def plotResults():

    # Initialize lists to store the accumulated data for all runs
    all_d_Combinedlosses = []
    all_d_Fakelosses = []
    all_d_Reallosses = []
    all_g_losses = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []

    Epoch_count = 0  # Initialize fold count

    # Process each output file
    for i in range(1, conf.NUM_RUNS):
        output_file_path = conf.output_file_path + f"{i+1}.txt"

        # Initialize lists to store the extracted data for the current run
        d_Combinedlosses = []
        d_Fakelosses = []
        d_Reallosses = []
        g_losses = []
        precisions = []
        recalls = []
        f1_scores = []
        # Open and read the current output file
        with open(output_file_path, 'r') as file:
            for line in file:
                # Adjusted condition for parsing training data
                if "Epoch" in line:
                    epoch_match = re.search(r'Epoch: (\d+)', line)
                    d_Combinedloss_match = re.search(r'D_loss: ([\d.]+)', line)
                    d_Fakeloss_match = re.search(r'fake loss: ([\d.]+)', line)
                    d_Realloss_match = re.search(r'real loss: ([\d.]+)', line)
                    g_loss_match = re.search(r'G_loss: ([\d.]+)', line)
                    if epoch_match and d_Fakeloss_match and g_loss_match and d_Realloss_match:
                        Epoch_count += 1  # Increment epoch count 
                        d_Fakelosses.append(float(d_Fakeloss_match.group(1)))
                        d_Reallosses.append(float(d_Realloss_match.group(1)))
                        g_losses.append(float(g_loss_match.group(1)))
                        d_Combinedlosses.append(float(d_Combinedloss_match.group(1)))
                
                # Check for and parse evaluation data
                elif "Fold:" in line:
                    precision_match = re.search(r'Precision: ([\d.]+)', line)
                    recall_match = re.search(r'Recall: ([\d.]+)', line)
                    f1_match = re.search(r'F1: ([\d.]+)', line)
                    if precision_match and recall_match and f1_match:
                        precisions.append(float(precision_match.group(1)))
                        recalls.append(float(recall_match.group(1)))
                        f1_scores.append(float(f1_match.group(1)))

        # Append the data from the current run to the accumulated lists
        all_d_Combinedlosses.append(d_Combinedlosses)
        all_d_Fakelosses.append(d_Fakelosses)
        all_d_Reallosses.append(d_Reallosses)
        all_g_losses.append(g_losses)
        all_precisions.append(precisions)
        all_recalls.append(recalls)
        all_f1_scores.append(f1_scores)

    # Convert lists of lists to NumPy arrays for easier averaging
    all_d_Combinedlosses = np.mean(np.array(all_d_Combinedlosses), axis=0)
    all_d_Fakelosses = np.mean(np.array(all_d_Fakelosses), axis=0)
    all_d_Reallosses = np.mean(np.array(all_d_Reallosses), axis=0)
    all_g_losses = np.mean(np.array(all_g_losses), axis=0)
    all_precisions = np.mean(np.array(all_precisions), axis=0)
    all_recalls = np.mean(np.array(all_recalls), axis=0)
    all_f1_scores = np.mean(np.array(all_f1_scores), axis=0)

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    epoch_cnt = int(Epoch_count/(conf.NUM_RUNS))


    # Adjusted plot for training losses
    axs[0].plot(range(1, epoch_cnt+1), all_d_Fakelosses, label=f'Average Discriminator Fake Loss -'+to_percentage_string(all_d_Fakelosses[-1]))
    axs[0].plot(range(1, epoch_cnt+1), all_d_Reallosses, label=f'Average Discriminator Real Loss -'+to_percentage_string(all_d_Reallosses[-1]))
    axs[0].plot(range(1, epoch_cnt+1), all_g_losses, label=f'Average Generator Loss - '+to_percentage_string(all_g_losses[-1]))
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Average Training Losses Over Epochs')
    axs[0].legend()

    # Plot for average evaluation metrics
    folds = range(1, len(all_precisions) + 1)
    axs[1].plot(folds, all_precisions, marker='o', label=f'Average Precision - '+to_percentage_string(all_precisions[-1]))
    axs[1].plot(folds, all_recalls, marker='o', label=f'Average Recall - '+to_percentage_string(all_recalls[-1]))
    axs[1].plot(folds, all_f1_scores, marker='o', label=f'Average F1 Score - '+to_percentage_string(all_f1_scores[-1]))
    axs[1].set_xlabel('Fold')
    axs[1].set_ylabel('Score')
    axs[1].set_title('Average Evaluation Metrics Across Folds')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def plotResult():
    # Initialize lists to store the extracted data
    Epoch_counts = []  # To track the fold count for each epoch
    d_Fakelosses = []
    d_Reallosses = []
    g_losses = []
    precisions = []
    recalls = []
    f1_scores = []

    Epoch_count = 0  # Initialize fold count

    # Open and read the output file
    with open(conf.output_file_path+"1.txt", 'r') as file:
        for line in file:
            # Adjusted condition for parsing training data
            if "Epoch" in line:
                epoch_match = re.search(r'Epoch: (\d+)', line)
                d_Fakeloss_match = re.search(r'fake loss: ([\d.]+)', line)
                d_Realloss_match = re.search(r'real loss: ([\d.]+)', line)
                g_loss_match = re.search(r'G_loss: ([\d.]+)', line)
                if epoch_match and d_Fakeloss_match and g_loss_match and d_Realloss_match:
                    Epoch_count += 1  # Increment epoch count 
                    Epoch_counts.append(Epoch_count)
                    d_Fakelosses.append(float(d_Fakeloss_match.group(1)))
                    d_Reallosses.append(float(d_Realloss_match.group(1)))
                    g_losses.append(float(g_loss_match.group(1)))
            
            # Check for and parse evaluation data
            elif "Fold:" in line:
                precision_match = re.search(r'Precision: ([\d.]+)', line)
                recall_match = re.search(r'Recall: ([\d.]+)', line)
                f1_match = re.search(r'F1: ([\d.]+)', line)
                if precision_match and recall_match and f1_match:
                    precisions.append(float(precision_match.group(1)))
                    recalls.append(float(recall_match.group(1)))
                    f1_scores.append(float(f1_match.group(1)))

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Adjusted plot for training losses to include fold information
    axs[0].plot(Epoch_counts, d_Fakelosses, label='Discriminator Fake Loss')
    axs[0].plot(Epoch_counts, d_Reallosses, label='Discriminator Real Loss')
    axs[0].plot(Epoch_counts, g_losses, label='Generator Loss')
    axs[0].set_xlabel('Epoch')  # Adjusted xlabel to represent fold
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Losses Over Epochs')
    axs[0].legend()

    # Plot for evaluation metrics
    folds = range(1, len(precisions) + 1)
    axs[1].plot(folds, precisions, marker='o', label='Precision')
    axs[1].plot(folds, recalls, marker='o', label='Recall')
    axs[1].plot(folds, f1_scores, marker='o', label='F1 Score')
    axs[1].set_xlabel('Fold')
    axs[1].set_ylabel('Score')
    axs[1].set_title('Evaluation Metrics Across Folds')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
