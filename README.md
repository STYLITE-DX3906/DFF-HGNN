# DFF-HGNN
This repository contains the demo code of the paper:
DFF-HGNN: Dual-Feature Fusion Heterogeneous Graph Neural Network
# Datasets
The dataset used in this paper is available at https://pan.quark.cn/s/4db3dece00dc.   Extraction code：2wcx
# Relation-based Feature Enhancement Strategy(RFE-Strategy)
![image](https://github.com/user-attachments/assets/0a4ea892-3af3-4728-8f27-1b1cf6952a64)
The RFE-Strategy proposed in this paper aims to improve the performance of HGNNs on non-attributed heterogeneous graphs, and its process is shown in Fig. 2, which consists of three steps: (1) Relational feature extraction: extracting the relational features from the original adjacency matrix of the heterogeneous graphs; (2) Identity feature encoding: encoding the identity of each node using the One-hot encoding to ensure uniqueness; and (3) Feature Enhancement: combining relational features and identity features to form an enhanced node representation, which is then used as an input to the HGNN.
# Dual-Feature Fusion Heterogeneous Graph Neural Network(DFF-HGNN)
![image](https://github.com/user-attachments/assets/7f5ccb4d-411f-4a9f-a819-bfeed7b187e8)
This section describes the proposed Dual-Feature Fusion Heterogeneous Graph Neural Network (DFF-HGNN), whose structure is shown in Fig. 4.DFF-HGNN aims to capture the heterogeneity and complexity of attributed heterogeneous graphs by fusing attribute and relational features, and the framework consists of four modules:(1) Separate Pre-transformation: mapping the node features to the generalized feature space by using the information entropy-based dimensionality reduction method to retain key feature information and reduce redundancy; (2) Intra-type Feature Encoder Based on Attributes and Relations: uses an attention mechanism to independently aggregate attribute features and relational features, and fuses them to form a unified node representation; (3) Complementary Attention-based Inter-type Feature Encoder: uses a complementary attention mechanism to fuse features from different types of neighbors, balancing local and global attention to capture the complexity of the interactions; (4) Embedding update encoder: updating node embeddings by combining features of the target node with aggregated neighbor features to obtain a final node representation.
