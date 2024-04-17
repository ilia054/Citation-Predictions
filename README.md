
# Citation Prediction using Generative Adversarial Networks (GANs)

## Overview

**Citation Prediction using GANs** project is aimed at automating the evaluation of citation relevance within scholarly articles. Utilizing a GAN-based algorithm, we construct and analyze a directed, weighted citation graph from the Cora dataset. The weights of the edges represent the cosine similarity of word vectors for each article, facilitating the discernment of relevant citations from potentially irrelevant ones and thus upholding research integrity.

In the images below, we present a classification of the results and examples, followed by a histogram and evaluation metrics.

- **Relevant Citations**: Detection success rate of 100%.
- **High Likelihood of Relevant Citations**: Success rates between 75% and 99%.
- **Lower Likelihood of Relevant Citations**: Success rates between 50% and 74%.
- **High Likelihood of Irrelevant Citations**: Success rate between 25% and 49%.
- **Irrelevant Citations**: Success rate below 25%.

The histogram showcases the distribution of success rates across various thresholds, illustrating the model's effectiveness at distinguishing between different levels of citation relevance. Additionally, the evaluation metrics image displays average precision, recall, and F1 score across different folds, indicating the robustness of our model.

![Results Classification and Examples](https://github.com/ilia054/Citation-Predictions/assets/88554020/f51ed6c6-ec85-45f8-9766-599cb15b1b67)
![Results Histogram and Evaluation Metrices](https://github.com/ilia054/Citation-Predictions/assets/88554020/d1d26675-b323-4025-8010-1a2cba74cc73)

## Motivation

The integrity of citations is crucial in academia. Our project combats the issue of irrelevant citations by employing advanced machine learning techniques, ensuring scholarly articles maintain their academic validity.

## Methodology

A directed weighted graph is constructed from the Cora dataset. Edge weights are calculated by the cosine similarity of word vectors representing each article. Feature vectors for each node are generated using PCA on the word vectors to add variability to the embeddings. The hierarchical breakdown algorithm, named **Lev Fridman NetLay**, incorporates the Infomap method of Martin Rosvall and Carl T. Bergstrom for abstract graph layering. The **EmbedGAN** algorithm is used recursively for model training, aiming to systematically predict citation relevance and improve the quality of scholarly literature.

## Results

Our GAN-based model demonstrates excellent precision and recall rates, proving its effectiveness in helping researchers identify and remove irrelevant citations with efficiency.

## Technologies and Dependencies

- **Python** 3.8.0
- **PyTorch** 2.2.1 for GAN implementation
- **NetworkX** 2.8.8 for graph manipulation
- **Node2Vec** 0.4.6 for node embeddings
- **scikit-learn** 1.3.2 for machine learning utilities
- **igraph** 0.11.4 for efficient graph analysis
- **stellargraph** 1.2.1 for machine learning on graph data
- **pandas** 2.0.3 for data manipulation and analysis
- **torch-optimizer** 0.3.0 for PyTorch optimization algorithms
- **matplotlib** 3.7.5 for creating visualizations
- **plotly** 5.20.0 for interactive graphing
- **openpyxl** 3.1.2 for Excel files support

## Installation

To set up the project environment:

1. Clone the repository: `git clone https://github.com/ilia054/Citation-Predictions`.
2. Install the dependencies: `pip install -r requirements.txt`.

## Usage

To run the project:

```bash
python main.py
```
## The Project's Book

This accompanying book details the foundational theories and methodologies employed in our Citation Prediction project, focusing on the utilization of Generative Adversarial Networks (GANs) for the automated assessment of citation relevance. It serves as a comprehensive guide, offering insights into the challenges faced and the innovative solutions adopted, enriching readers’ understanding of advanced machine learning applications in academic research integrity.

## Contributors

- **Biran Fridman**
- **Ilya Lev**

Special thanks to **Dr. Renata Avros** and **Prof. Zeev Volkovich** for their guidance and support.

## Acknowledgements

We are grateful to the scholarly community and the numerous open-source projects that have made this research possible.

## License

This project is open-sourced under the MIT License. For more details, see the [LICENSE](LICENSE) file.
