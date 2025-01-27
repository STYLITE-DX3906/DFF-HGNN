# DFF-HGNN
This repository contains the demo code of the paper:
DFF-HGNN: Dual-Feature Fusion Heterogeneous Graph Neural Network
# Datasets
The dataset used in this paper is available at https://pan.quark.cn/s/4db3dece00dc.   Extraction code：2wcx

# Relation-based Feature Enhancement Strategy(RFE-Strategy)

The RFE-Strategy proposed in this paper aims to improve the performance of HGNNs on non-attributed heterogeneous graphs, and its process is shown in the following figure, which consists of three steps: (1) Relational feature extraction: extract relational features from the original adjacency matrix of the heterogeneous graphs; (2) Identity feature encoding: encode the identity of each node using the One-hot encoding to ensure uniqueness; (3) Feature Enhancement: combining relational features and identity features to form an enhanced node representation, which is then used as an input to the HGNN.
![image](https://github.com/user-attachments/assets/c161962b-4336-43f3-8d7c-8f415c4f3b70)


# Dual-Feature Fusion Heterogeneous Graph Neural Network(DFF-HGNN)

This section describes the proposed Dual-Feature Fusion Heterogeneous Graph Neural Network (DFF-HGNN), whose structure is shown in the following figure.DFF-HGNN aims to capture the heterogeneity and complexity of attributed heterogeneous graphs by fusing attribute and relational features, and the framework consists of four modules:(1) Separate Pre-transformation: mapping the node features to the generalized feature space by using the information entropy-based dimensionality reduction method to retain key feature information and reduce redundancy; (2) Intra-type Feature Encoder Based on Attributes and Relations: uses an attention mechanism to independently aggregate attribute features and relational features, and fuses them to form a unified node representation; (3) Complementary Attention-based Inter-type Feature Encoder: uses a complementary attention mechanism to fuse features from different types of neighbors, balancing local and global attention to capture the complexity of the interactions; (4) Embedding update encoder: updating node embeddings by combining features of the target node with aggregated neighbor features to obtain a final node representation.
![image](https://github.com/user-attachments/assets/d41cb351-14a0-4a89-99f6-ca11b0d6e34b)
