import matplotlib.pyplot as plt
import re
output_file_path = "C:\\Users\\ilia0\\Desktop\\Final Semester\\Cora\\cora\\output.txt"
# Initialize lists to store the extracted data
epoch_numbers = []
d_losses = []
g_losses = []
precisions = []
recalls = []
f1_scores = []

# Open and read the output file
with open(output_file_path, 'r') as file:
    for line in file:
        # Check for and parse training data
        if "Epoch" in line:
            epoch_match = re.search(r'Epoch: (\d+)', line)
            d_loss_match = re.search(r'D_loss: ([\d.]+)', line)
            g_loss_match = re.search(r'G_loss: ([\d.]+)', line)
            if epoch_match and d_loss_match and g_loss_match:
                epoch_numbers.append(int(epoch_match.group(1)))
                d_losses.append(float(d_loss_match.group(1)))
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

# Plot for training losses
axs[0].plot(epoch_numbers, d_losses, label='Discriminator Loss')
axs[0].plot(epoch_numbers, g_losses, label='Generator Loss')
axs[0].set_xlabel('Epoch')
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
