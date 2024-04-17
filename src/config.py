import pandas as pd
import networkx as nx

# CONSTANTS
EMBEDDING_DIMENSION = 256
NUM_EPOCHS = 50
BATCH_SIZE = 8
LR = 0.0002
G_LR = 0.0001
K_FOLD_NUM = 5
NUM_RUNS = 1
PREDICTION_PRECENTAGE = 0.3

# file paths
coraGraphFile_path = "Resources/cora/cora.cites"
metaDataPath= "Resources/cora/cora.content"
output_file_path = "Resources/output/evaluation_metrics/evaluation"
predictions_results_file_path = "Resources/output/predictions_results/predictions"
