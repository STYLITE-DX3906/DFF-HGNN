import scipy
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix, save_npz
import os



def Generate_ACM_Relation_Feature(prefix=r'E:\山东科技大学\模型\模型代码测试2.0-V2优化实验\acm'):

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()

    # 提取 P 类型节点的行
    P_nodes_rows = adjM[:4019, :]

    # 提取 A 类型节点的行
    A_nodes_rows = adjM[4019:4019 + 7167, :]

    # 提取 S 类型节点的行
    S_nodes_rows = adjM[4019 + 7167:4019 + 7167 + 60, :]

    # 保存路径
    # save_path = r'E:\山东科技大学\模型\模型代码测试2.0-V2优化实验\acm'

    # 确保保存路径存在
    # os.makedirs(save_path, exist_ok=True)

    # 保存为 .npz 文件
    # save_npz(os.path.join(save_path, 'P_nodes_rows.npz'), csr_matrix(P_nodes_rows))
    # save_npz(os.path.join(save_path, 'A_nodes_rows.npz'), csr_matrix(A_nodes_rows))
    # save_npz(os.path.join(save_path, 'S_nodes_rows.npz'), csr_matrix(S_nodes_rows))

    # P_nodes_rows = torch.FloatTensor(P_nodes_rows)
    # A_nodes_rows = torch.FloatTensor(A_nodes_rows)
    # S_nodes_rows = torch.FloatTensor(S_nodes_rows)
    #
    # Relation_features = [P_nodes_rows, A_nodes_rows, S_nodes_rows]
    # return Relation_features

    return P_nodes_rows , A_nodes_rows , S_nodes_rows


# Relation_features[0].shape=torch.Size([4019, 11246])
# print(a=load_ACM_relation_feature())


def Generate_IMDB_Relation_Feature(prefix=r'E:\山东科技大学\模型\模型代码测试2.0-V2优化实验\imdb'):

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()

    # 提取 M 类型节点的行
    M_nodes_rows = adjM[:4278, :]

    # 提取 D 类型节点的行
    D_nodes_rows = adjM[4278:4278 + 2081, :]

    # 提取 A 类型节点的行
    A_nodes_rows = adjM[4278 + 2081:4278 + 2081 + 5257, :]

    # # 保存路径
    # save_path = r'E:\山东科技大学\模型\模型代码测试2.0-V2\imdb'
    #
    # # 确保保存路径存在
    # os.makedirs(save_path, exist_ok=True)

    # # 保存为 .npz 文件
    # save_npz(os.path.join(save_path, 'M_nodes_rows.npz'), csr_matrix(M_nodes_rows))
    # save_npz(os.path.join(save_path, 'D_nodes_rows.npz'), csr_matrix(D_nodes_rows))
    # save_npz(os.path.join(save_path, 'A_nodes_rows.npz'), csr_matrix(A_nodes_rows))

    # M_nodes_rows = torch.FloatTensor(M_nodes_rows)
    # D_nodes_rows = torch.FloatTensor(D_nodes_rows)
    # A_nodes_rows = torch.FloatTensor(A_nodes_rows)

    # Relation_features = [M_nodes_rows, D_nodes_rows, A_nodes_rows]

    return M_nodes_rows ,D_nodes_rows ,A_nodes_rows



def Generate_DBLP_Relation_Feature(prefix=r'E:\山东科技大学\模型\模型代码测试2.0-V2\dblp'):

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()

    # 提取 A 类型节点的行
    A_nodes_rows = adjM[:4057, :]

    # 提取 P 类型节点的行
    P_nodes_rows = adjM[4057:4057 + 14328, :]

    # 提取 T 类型节点的行
    T_nodes_rows = adjM[4057 + 14328:4057 + 14328 + 7723, :]

    # 提取 C 类型节点的行
    C_nodes_rows = adjM[4057 + 14328 + 7723:4057 + 14328 + 7723 + 20, :]



    # # 保存路径
    # save_path = r'E:\山东科技大学\模型\模型代码测试2.0-V2\dblp'
    #
    # # 确保保存路径存在
    # os.makedirs(save_path, exist_ok=True)
    #
    # # 保存为 .npz 文件
    # save_npz(os.path.join(save_path, 'A_nodes_rows.npz'), csr_matrix(A_nodes_rows))
    # save_npz(os.path.join(save_path, 'P_nodes_rows.npz'), csr_matrix(P_nodes_rows))
    # save_npz(os.path.join(save_path, 'T_nodes_rows.npz'), csr_matrix(T_nodes_rows))
    # save_npz(os.path.join(save_path, 'C_nodes_rows.npz'), csr_matrix(C_nodes_rows))

    # A_nodes_rows = torch.FloatTensor(A_nodes_rows)
    # P_nodes_rows = torch.FloatTensor(P_nodes_rows)
    # T_nodes_rows = torch.FloatTensor(T_nodes_rows)
    # C_nodes_rows = torch.FloatTensor(C_nodes_rows)

    # Relation_features = [A_nodes_rows, P_nodes_rows, T_nodes_rows,C_nodes_rows]

    return A_nodes_rows ,P_nodes_rows ,T_nodes_rows ,C_nodes_rows