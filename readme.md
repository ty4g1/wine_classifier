# Wine Classification Methods Comparison

This repository contains three Jupyter notebooks demonstrating different approaches to wine classification using the scikit-learn wine dataset. The notebooks explore unsupervised learning (K-means clustering), supervised learning (neural networks), and a hybrid approach using pseudo-labels.

## Dataset

The scikit-learn wine dataset contains 178 samples with 13 features derived from chemical analysis of wines grown in a specific area of Italy. The wines are classified into three different cultivars, making it an ideal dataset for testing classification methods.

Features include:
- Alcohol content
- Malic acid
- Ash
- Alcalinity of ash
- Magnesium
- Total phenols
- Flavanoids
- Nonflavanoid phenols
- Proanthocyanins
- Color intensity
- Hue
- OD280/OD315 of diluted wines
- Proline

## Notebooks

### 1. wine_classifier_kmeans
Demonstrates unsupervised learning using K-means clustering.
- Explores data preprocessing and normalization
- Uses the elbow method to determine optimal cluster count
- Visualizes clustering results using PCA
- Evaluates clustering performance against known labels
- Includes feature importance analysis through PCA components

### 2. wine_classifier_nn
Implements a supervised learning approach using neural networks.
- Data preprocessing and train-test splitting
- Simple neural network architecture implementation
- Training process with loss and accuracy monitoring
- Model evaluation on test set
- Visualization of training metrics

### 3. wine_classifier_pseudo
Combines unsupervised and supervised learning through pseudo-labeling.
- Initial clustering to generate pseudo-labels
- Training neural network using pseudo-labels
- Evaluation against true labels
- Comparison with pure supervised approach
- Discussion of approach limitations and potential applications

## Methodology & Assumptions

### K-means Clustering
- Assumes spherical clusters
- Assumes similar cluster sizes
- Features are scaled to have equal importance
- Number of clusters matches known classes

### Neural Network (Supervised)
- Simple architecture (13->8->3)
- ReLU activation for hidden layer
- Cross-entropy loss for classification
- Adam optimizer with default parameters
- 80-20 train-test split

### Pseudo-labeling Approach
- Clusters from K-means can map to true classes
- Neural network can learn from noisy labels
- Same architecture as supervised approach
- Training process identical to supervised

## Key Findings

1. K-means Clustering
- Successfully identified natural groupings in the data
- Clusters roughly corresponded to true classes
- PCA revealed feature importance patterns

2. Supervised Learning
- Achieved high accuracy on test set
- Quick convergence during training
- Stable performance across multiple runs

3. Pseudo-labeling
- Performance between random and supervised
- ~61% accuracy vs 33% random baseline
- Demonstrated viability of approach

## Limitations

1. Dataset Specific
- Small dataset size
- Well-separated classes
- Limited feature set
- Balanced classes

2. Methodological
- Simple neural network architecture
- Basic preprocessing
- Limited hyperparameter tuning
- Single train-test split (no cross-validation)

## Potential Applications

The pseudo-labeling approach shows particular promise for:
- Subjective classification tasks (e.g., emotion, mood)
- Scenarios with limited labeled data
- Cases where ground truth is fuzzy or uncertain
- Initial labeling of large datasets

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Matplotlib
- Seaborn

## Usage

Notebooks should be run in the following order:
1. wine_classifier_kmeans.ipynb
2. wine_classifier_nn.ipynb
3. wine_classifier_pseudo.ipynb

Each notebook contains detailed comments and markdown cells explaining the process and decisions made at each step.

## License

MIT License
