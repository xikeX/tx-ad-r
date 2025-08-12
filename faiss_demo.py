# faiss_demo.py

import os
from pathlib import Path
import struct
import faiss
import numpy as np
from typing import Optional

def read_fbin(filename: str) -> np.ndarray:
    """
    读取 .fbin 格式的向量文件。
    文件格式：前8字节为 int32 的 n 和 d，接着是 n*d 个 float32 数据。
    返回 shape 为 [n, d] 的 numpy array。
    """
    with open(filename, "rb") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
        # 根据文件后缀以不同的方式读取数据
        if filename.endswith(".fbin"):
            data = np.fromfile(f, dtype=np.float32, count=n * d)
        elif filename.endswith(".u64bin"):
            data = np.fromfile(f, dtype=np.uint64, count=n)
        # data = np.fromfile(f, dtype=np.float32, count=n * d)
        return data.reshape(n, d)

def load_emb(load_path):
    """
    从二进制文件加载 Embedding

    Args:
        load_path: 文件路径

    Returns:
        embedding: numpy array, shape [num_points, num_dimensions]
    """
    with open(load_path, 'rb') as f:
        # 读取前8字节：num_points 和 num_dimensions (两个uint32)
        header = f.read(8)
        num_points, num_dimensions = struct.unpack('II', header)
        # 读取后面的向量数据
        emb = np.fromfile(f, dtype=np.float32).reshape(num_points, num_dimensions)
    return emb


def read_u64bin(filename: str) -> np.ndarray:
    """
    读取 .u64bin 格式的 ID 文件。
    返回一维 uint64 数组。
    """
    return np.fromfile(filename, dtype=np.uint64)


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1] if emb.ndim > 1 else 1  # 向量的维度
    print(f'saving {save_path}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)



def run_faiss_ann_search(
    dataset_vector_file_path: str,
    dataset_id_file_path: str,
    query_vector_file_path: str,
    result_id_file_path: str,
    query_ann_top_k: int = 10,
    faiss_M: int = 64,
    faiss_ef_construction: int = 1280,
    query_ef_search: int = 640,
    faiss_metric_type: int = 1,  # 0: L2, 1: Inner Product
) -> None:
    """
    使用 Faiss 构建 HNSW 索引并执行近似最近邻搜索，保存 Top-K 结果 ID。

    参数:
        dataset_vector_file_path (str): 物料向量文件路径 (.fbin)
        dataset_id_file_path (str): 物料 ID 文件路径 (.u64bin)
        query_vector_file_path (str): 查询向量文件路径 (.fbin)
        result_id_file_path (str): 检索结果 ID 输出路径 (.u64bin)
        query_ann_top_k (int): 返回每个查询的 Top-K 个最近邻
        faiss_M (int): HNSW 中每个节点的邻居数量
        faiss_ef_construction (int): 构建时的探索深度
        query_ef_search (int): 查询时的探索深度
        faiss_metric_type (int): 距离度量类型 (0=L2, 1=内积)
    """
    print("Loading dataset vectors...")
    xb = read_fbin(dataset_vector_file_path)  # [N, D]
    ids = read_fbin(dataset_id_file_path)   # [N]

    print("Loading query vectors...")
    xq = read_fbin(query_vector_file_path)    # [Q, D]

    assert len(ids) == len(xb), "Error: 向量数量和 ID 数量不匹配！"

    dim = xb.shape[1]
    metric = faiss.METRIC_L2 if faiss_metric_type == 0 else faiss.METRIC_INNER_PRODUCT

    print("Building HNSW index...")
    index = faiss.IndexHNSWFlat(dim, faiss_M, metric)
    index.hnsw.efConstruction = faiss_ef_construction
    index.hnsw.efSearch = query_ef_search
    index.verbose = True

    # 包装成 IDMap 以保留原始 ID
    index = faiss.IndexIDMap2(index)  # 包装成 IDMap
    ids = ids.ravel()
    index.add_with_ids(xb, ids)

    # ✅ 正确方式：通过 .index 访问底层 HNSW 索引
    # index.index.hnsw.efSearch = query_ef_search

    print("Searching for nearest neighbors...")
    distances, indices = index.search(xq, query_ann_top_k)

    print(f"Writing Top-{query_ann_top_k} results to {result_id_file_path}")
    save_emb(indices.astype(np.uint64),result_id_file_path)

    # 输出平均距离
    print(f"Average distance: {np.mean(distances)}")
    print("ANN search completed.")