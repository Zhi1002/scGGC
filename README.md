# scGGC: A Two-Stage Strategy for Single-Cell Clustering through Cellular Gene Pathway Construction
# Overview
    we propose scGGC, a semi-supervised clustering method based on generative adversarial networks. First, we construct the overall neighbourhood matrix by integrating cell-
    gene interaction information to more comprehensively reflect the complex interactions between cells and genes. Based on this, the graph autoencoder is employed to reduce 
    dimensionality and capture the complex nonlinear structure of the data. The resulting low-dimensional latent variables are then preliminarily clustered using Kmeans.Based 
    on the preliminary clustering results, the distance of each cell to the centre of mass within its cluster is calculated, and the cell closest to the centre of mass is 
    selected as a high-confidence sample to more accurately represent the structural characteristics of the cluster. Using the obtained high-confidence samples, the GAN model 
    is trained to optimise the clustering results again, thus improving the generalisation ability and accuracy of the model. The phased design of the method allows us to 
    effectively deal with the high-dimensional nature and complexity of single-cell data, significantly improving the accuracy and biological explanatory power of clustering.


# Overview of the repository
    processing.py Data Preprocessing
    model.py      Graph Autoencoder and Adversarial Model
    scGGC.py      Main Execution File
    training.py   Model Training Module
    
# Clone the Repository
git clone https://github.com/Zhi1002/scGGC.git

# Installation
pip install torch==1.13.1+cu116 numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2


    
