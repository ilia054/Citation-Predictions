import matplotlib.pyplot as plt
import re

output_file_path = "C:\\Users\\ilia0\\Desktop\\Final Semester\\Cora\\cora\\output.txt"

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
with open(output_file_path, 'r') as file:
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
