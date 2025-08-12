import os
import random

from infer_class import Infer
if os.environ.get("DEBUG_MODE","") == 'True':
    print("in debug model")
    os.environ['TRAIN_LOG_PATH']='/HOME/hitsz_mszhang/hitsz_mszhang_1/RL_ERC/tecent_recommondation/log_path'
    os.environ['TRAIN_TF_EVENTS_PATH']='/HOME/hitsz_mszhang/hitsz_mszhang_1/RL_ERC/tecent_recommondation/log_path'
    os.environ['TRAIN_DATA_PATH']='/HOME/hitsz_mszhang/hitsz_mszhang_1/RL_ERC/tecent_recommondation/TencentGR_1k'
    os.environ['TRAIN_CKPT_PATH']='/HOME/hitsz_mszhang/hitsz_mszhang_1/RL_ERC/tecent_recommondation/checkpoint'
    os.environ['USER_CACHE_PATH']='/HOME/hitsz_mszhang/hitsz_mszhang_1/RL_ERC/tecent_recommondation/user_cache_file'
    os.environ['DEBUG_MODE'] = 'True'
    # 设置单GPU
os.environ['EVAL_RESULT_PATH'] = './eval_result'

"""
训练主脚本：用于训练嵌入模型（Embedding Model），并加载到下游模型中进行后续任务。
"""

import os
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 自定义模块
from my_dataset import BaseDataset, TrainDataset, EmbeddingDataset, ValidDataset
from embedding_model import BaselineModel as EmbeddingModel
from model import BaselineModel as DownstreamModel


def set_seed(seed=42):
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练序列推荐模型（含特征嵌入）")

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128, help='训练/验证批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--maxlen', type=int, default=101, help='序列最大长度')

    # 模型结构参数
    parser.add_argument('--hidden_units', type=int, default=32, help='隐藏层维度')
    parser.add_argument('--num_blocks', type=int, default=1, help='Transformer 块数')
    parser.add_argument('--num_epochs', type=int, default=3, help='训练总轮数')
    parser.add_argument('--num_heads', type=int, default=1, help='多头注意力头数')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout 比例')
    parser.add_argument('--l2_emb', type=float, default=0.0, help='嵌入层L2正则强度')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备: cpu 或 cuda')
    parser.add_argument('--inference_only', action='store_true', help='仅推理模式')
    parser.add_argument('--state_dict_path', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--norm_first', action='store_true', help='是否在Transformer中先归一化')
    parser.add_argument('--mm_emb_id', nargs='+', type=str, default=['81'],
                        choices=[str(s) for s in range(81, 87)],
                        help='多模态嵌入特征ID列表')
    parser.add_argument('--checkpoint_path', type=str, help='载入模型的路径')

    return parser.parse_args()


def main():
    set_seed(42)

    # 环境变量检查
    required_env_vars = ['TRAIN_LOG_PATH', 'TRAIN_TF_EVENTS_PATH', 'TRAIN_DATA_PATH', 'USER_CACHE_PATH']
    for var in required_env_vars:
        if not os.environ.get(var):
            raise EnvironmentError(f"缺少必需的环境变量: {var}")

    # 创建日志与事件目录
    Path(os.environ['TRAIN_LOG_PATH']).mkdir(parents=True, exist_ok=True)
    Path(os.environ['TRAIN_TF_EVENTS_PATH']).mkdir(parents=True, exist_ok=True)

    log_file = open(Path(os.environ['TRAIN_LOG_PATH']) / 'train.log', 'w')
    writer = SummaryWriter(os.environ['TRAIN_TF_EVENTS_PATH'])

    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device

    # 数据加载
    data_path = os.environ['TRAIN_DATA_PATH']
    base_dataset = BaseDataset(data_path, args)
    train_idx, valid_idx = base_dataset.split_index([0.9, 0.1])

    # 加载嵌入权重到下游模型（示例）
    downstream_model = DownstreamModel(base_dataset.usernum, base_dataset.itemnum, base_dataset.feat_statistics, base_dataset.feature_types, args).to(device)
    print("Downstream Model:\n", downstream_model)

    try:
        downstream_model.load_state_dict(torch.load(args.checkpoint_path / "model.pt", map_location=device))
        print("✅ 下游模型成功加载嵌入权重")
    except Exception as e:
        print(f"❌ 下游模型加载权重失败: {e}")

    test_dataset = ValidDataset(base_dataset, sample_index=valid_idx)  # 可替换为独立测试集



    # 模型初始化
    usernum, itemnum = base_dataset.usernum, base_dataset.itemnum
    feat_stats, feat_types = base_dataset.feat_statistics, base_dataset.feature_types

    model = DownstreamModel(usernum, itemnum, feat_stats, feat_types, args).to(device)
    print("Downstream Model:\n", model)

    # 加载预训练 embedding（可选）
    if args.state_dict_path:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=device))
            print(f"✅ 已加载预训练权重: {args.state_dict_path}")
        except Exception as e:
            print(f"⚠️ 权重加载失败: {e}")

    # ========== 推理 ==========
    candidate_path = os.path.join(os.environ['TRAIN_DATA_PATH'], 'item_feat_dict.json')
    infer = Infer(args, model, eval_dataset=test_dataset, candidate_path=candidate_path)
    infer.infer()
    print("✅ 推理完成")

    # 清理资源
    writer.close()
    log_file.close()
    
if __name__ == '__main__':
    main()
    
    