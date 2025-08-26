# main.py
from datetime import datetime
import os
import pickle
import random

if os.environ.get("DEBUG_MODE","") == 'True':
    print("in debug model")
    os.environ['TRAIN_LOG_PATH']='./log_path'
    os.environ['TRAIN_TF_EVENTS_PATH']='./log_path'
    os.environ['TRAIN_DATA_PATH']='../TencentGR_1k'
    os.environ['TRAIN_CKPT_PATH']='./checkpoint'
    os.environ['USER_CACHE_PATH']='../user_cache_file'
    os.makedirs(os.environ['USER_CACHE_PATH'], exist_ok = True)
    os.environ['DEBUG_MODE'] = 'True'
    # 设置单GPU
# 获取当前目录（'.'）的绝对路径
current_abs_path = os.path.abspath('./temp')
os.environ['TEMP_PATH'] = current_abs_path
os.makedirs(os.environ['TEMP_PATH'], exist_ok = True)
current_abs_path = os.path.abspath('./eval_result')
os.environ['EVAL_RESULT_PATH'] = current_abs_path
os.makedirs(os.environ['EVAL_RESULT_PATH'], exist_ok = True)
from my_dataset_v1_aug import BaseDataset, ValidDataset
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from infer_class_v1 import Infer
from baseline_model_v3_infonce import BaselineModel



def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default="global_step_2/model.pt", type=str)
    parser.add_argument('--train_name',type=str, default="v3", help='训练名称')
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args
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
    if os.environ.get('DEBUG_MODE',"")=="True":
        writer = SummaryWriter(Path(os.environ['TRAIN_TF_EVENTS_PATH'])/datetime.now().strftime('%H-%M-%S'))
    else:
        writer = SummaryWriter(os.environ['TRAIN_TF_EVENTS_PATH'])

    args = get_args()

    # 数据加载
    data_path = os.environ['TRAIN_DATA_PATH']
    base_dataset = BaseDataset(data_path, args)
    # 构建嵌入训练与验证数据集
    usernum, itemnum = base_dataset.usernum, base_dataset.itemnum
    feat_stats, feat_types = base_dataset.feat_statistics, base_dataset.feature_types
    with open(os.environ.get("USER_CACHE_PATH")+"/valid_idx.pkl", "rb") as f:
        valid_idx = pickle.load(f)
    # ========== 推理 ==========
    test_dataset = ValidDataset(base_dataset, sample_index=valid_idx)  # 可替换为独立测试集

    
    usernum, itemnum = base_dataset.usernum, base_dataset.itemnum
    feat_statistics, feat_types = base_dataset.feat_statistics, base_dataset.feature_types

    downstream_model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    save_dir = Path(os.environ.get('USER_CACHE_PATH')) /args.train_name 
    # 删除旧的模型文件
    if os.path.exists(save_dir) and  len(os.listdir(save_dir))!=0:
       args.state_dict_path = save_dir/os.listdir(save_dir)[0]/"model.pt"
    try:
        print(args.state_dict_path)
        downstream_model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
    except:
        print('failed loading state_dicts, pls check file path: ', end="")
        print(args.state_dict_path)
        raise RuntimeError('failed loading state_dicts, pls check file path!')


    print("评估开始")
    eval_candidate_path = os.path.join(os.environ['TRAIN_DATA_PATH'], 'item_feat_dict.json')
    infer = Infer(args, downstream_model, eval_dataset=test_dataset, candidate_path=eval_candidate_path,name='global_test',query_ann_top_k=10)
    hitrate_eval = infer.infer()
    print("✅ 评估结果")
    print("eval:", hitrate_eval)

if __name__ == "__main__":
    main()

# infer_main.py