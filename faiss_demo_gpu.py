# faiss_demo_gpu.py
import os
from pathlib import Path
import struct
import numpy as np
import faiss
import time


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
    dataset_vector_file_path: str = None,
    dataset_id_file_path: str = None,
    query_vector_file_path: str = None,
    result_id_file_path: str = None,
    dataset_vector=None,
    dataset_id=None,
    query_vector=None,
    query_ann_top_k: int = 10,
    cagra_M: int = 64,
    cagra_intermediate_graph_degree: int = 128,
    cagra_graph_degree: int = 64,
    cagra_build_algo: str = "nn_descent",  # "nn_descent", "ivf"
    cagra_cluster_factor: float = 32.0,    # 用于 IVF 阶段（如果启用）
    search_L: int = 64,                    # 搜索时的候选集大小
    max_search_iterations: int = 20,
    parallel_refine: bool = True,
    faiss_metric_type: int = 1,  # 0: L2, 1: Inner Product
    gpu_id: int = 0,
) -> np.ndarray:
    """
    使用 Faiss CAGRA (GPU) 构建图索引并执行近似最近邻搜索，保存 Top-K 结果 ID。

    参数:
        dataset_vector_file_path: 物料向量文件路径 (.fbin)
        dataset_id_file_path: 物料 ID 文件路径 (.u64bin)
        query_vector_file_path: 查询向量文件路径 (.fbin)
        result_id_file_path: 检索结果 ID 输出路径 (.u64bin)
        query_ann_top_k: 返回每个查询的 Top-K 个最近邻
        cagra_M: 构建时每个点的连接数（影响图质量）
        cagra_intermediate_graph_degree: 中间图度数（NN-Descent 阶段）
        cagra_graph_degree: 最终图的出度（即每个节点保留的邻居数）
        cagra_build_algo: 图构建算法 ("nn_descent" 推荐)
        cagra_cluster_factor: 若使用 IVF 阶段，控制聚类数量（如 total / factor）
        search_L: 搜索时的候选列表大小（越大越准越慢）
        max_search_iterations: 搜索最大迭代次数
        parallel_refine: 是否并行优化路径
        faiss_metric_type: 0=L2, 1=Inner Product
        gpu_id: 使用的 GPU 编号
    """
    # Step 1: 加载数据
    if dataset_vector is None:
        dataset_vector = read_fbin(dataset_vector_file_path)  # [N, D]
    if dataset_id is None:
        dataset_id = read_u64bin(dataset_id_file_path)  # [N]
    if query_vector is None:
        query_vector = read_fbin(query_vector_file_path)  # [Q, D]

    assert len(dataset_id) == len(dataset_vector), "Error: 向量数量和 ID 数量不匹配！"
    assert dataset_vector.dtype == np.float32, "CAGRA requires float32"

    dim = dataset_vector.shape[1]
    metric = faiss.METRIC_L2 if faiss_metric_type == 0 else faiss.METRIC_INNER_PRODUCT

    # Step 2: 设置 GPU 资源
    res = faiss.StandardGpuResources()
    res.setTempMemory(1024 * 1024 * 1024)  # 1GB 临时内存
    # res.setDevice(gpu_id)

    # Step 3: 创建 CAGRA 索引
    index = faiss.GpuIndexCagra(
        d=dim,
        metric=metric,
        intermediate_graph_degree=cagra_intermediate_graph_degree,
        graph_degree=cagra_graph_degree,
        build_algo=cagra_build_algo,
        cluster_factor=cagra_cluster_factor,
        M=cagra_M,
        reserveVecs=dataset_vector.shape[0],
        store_pairs=True,  # 保留双向边（提升质量）
        rerank=True,
        min_iterations=3,
        max_iterations=max_search_iterations,
        parallel_refine=parallel_refine,
    )

    print(f"开始构建 CAGRA 索引，数据量: {dataset_vector.shape[0]}, 维度: {dim}")
    start_time = time.time()

    # 注意：CAGRA 不支持直接 add_with_ids，但 GpuIndexCagra 支持带 ID 的添加
    ids = dataset_id.ravel().astype(np.int64)
    index.add_with_ids(dataset_vector, ids)

    build_time = time.time() - start_time
    print(f"CAGRA 构建耗时: {build_time:.2f}s")

    # Step 4: 设置搜索参数
    index.setNumRunaways(1)           # 多路径搜索（通常设为1）
    index.setSearchK(search_L)        # 搜索宽度
    index.setMinIterations(3)
    index.setMaxIterations(max_search_iterations)

    # Step 5: 执行搜索
    print(f"开始搜索 {query_vector.shape[0]} 个查询...")
    start_time = time.time()
    distances, indices = index.search(query_vector, query_ann_top_k)
    search_time = time.time() - start_time
    print(f"搜索耗时: {search_time:.2f}s, 平均每查询: {search_time / query_vector.shape[0]:.4f}s")

    # Step 6: 保存结果
    if result_id_file_path is not None:
        print(f"保存 Top-{query_ann_top_k} 结果到 {result_id_file_path}")
        save_emb(indices.astype(np.uint64), result_id_file_path)

    return indices