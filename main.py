# main.py
from collections import defaultdict
from datetime import datetime
import os

from infer_class import Infer
if os.environ.get("DEBUG_MODE","") == 'True':
    print("in debug model")
    os.environ['TRAIN_LOG_PATH']='./log_path'
    os.environ['TRAIN_TF_EVENTS_PATH']='./log_path'
    os.environ['TRAIN_DATA_PATH']='./TencentGR_1k'
    os.environ['TRAIN_CKPT_PATH']='./checkpoint'
    os.environ['USER_CACHE_PATH']='./user_cache_file'
    os.environ['DEBUG_MODE'] = 'True'
    # 设置单GPU
import os
from transformers import get_cosine_schedule_with_warmup
# 获取当前目录（'.'）的绝对路径
current_abs_path = os.path.abspath('./temp')
os.environ['TEMP_PATH'] = current_abs_path
os.makedirs(os.environ['TEMP_PATH'], exist_ok = True)
current_abs_path = os.path.abspath('./eval_result')
os.environ['EVAL_RESULT_PATH'] = current_abs_path
os.makedirs(os.environ['EVAL_RESULT_PATH'], exist_ok = True)

"""
训练主脚本：用于训练嵌入模型（Embedding Model），并加载到下游模型中进行后续任务。
"""

import os
import argparse
import json
import time
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
# 自定义模块
from my_dataset import BaseDataset, TrainDataset, EmbeddingDataset, ValidDataset
from split_embedding_model import ADEmbeddingLayer as EmbeddingModel
# from split_embedding_model import BaselineModel as DownstreamModel
from baseline_model import BaselineModel as DownstreamModel
from transformers import get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
from pathlib import Path
import json
import os
import time
from tqdm import tqdm
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
def format_time(seconds):
    """将秒转换为 HH:MM:SS 格式"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
def get_args():
    parser = argparse.ArgumentParser()

    # ================== 数据相关参数 ==================
    parser.add_argument('--train_data_size', type=int, default=-1, help='训练数据大小（-1 表示全量）')
    parser.add_argument('--batch_size', type=int, default=128, help='训练/验证批大小')
    parser.add_argument('--embedding_batch_size', type=int, default=64, help='训练 embedding 的批大小')
    parser.add_argument('--maxlen', type=int, default=101, help='序列最大长度')
    parser.add_argument('--num_worker', type=int, default=5, help='序列最大长度')

    # ================== 模型结构参数 ==================
    parser.add_argument('--hidden_units', type=int, default=32, help='隐藏层维度')
    parser.add_argument('--num_blocks', type=int, default=1, help='Transformer 块数')
    parser.add_argument('--num_heads', type=int, default=4, help='多头注意力头数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout 比例')
    parser.add_argument('--l2_emb', type=float, default=0.0, help='嵌入层 L2 正则强度')
    parser.add_argument('--norm_first', action='store_true', help='是否在 Transformer 中先归一化（Pre-LN）')
    parser.add_argument('--mm_emb_id', nargs='+', type=str, default=['81'],
                        choices=[str(s) for s in range(81, 87)],
                        help='多模态嵌入特征 ID 列表')

    # ================== 训练优化参数 ==================
    parser.add_argument('--num_epochs', type=int, default=25, help='训练总轮数')
    parser.add_argument('--embedding_task_lr', type=float, default=1e-3, help='embedding 任务学习率')
    parser.add_argument('--downstream_task_lr', type=float, default=1e-2, help='下游任务学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减（AdamW 等优化器使用）')
    parser.add_argument('--warm_up_rate', default=0.05, type=float, help='学习率 warm-up 占比')

    # ================== 学习率调度 ==================
    parser.add_argument('--use_lr_scheduler_in_embedding_task', action='store_true',
                        help='是否在 embedding 任务中使用学习率调度')
    parser.add_argument('--use_lr_scheduler_in_downstream', action='store_true',
                        help='是否在下游任务中使用学习率调度')

    # ================== 模型加载与推理 ==================
    parser.add_argument('--use_embedding_model', action='store_true',
                        help='是否使用预训练的 embedding 模型权重')
    parser.add_argument('--state_dict_path', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--inference_only', action='store_true', help='仅推理模式')

    # ================== 运行环境 ==================
    parser.add_argument('--device', type=str, default='', help='运行设备: cpu 或 cuda')

    # 解析参数
    args = parser.parse_args()

    # 自动设置 device
    if args.device == '':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args


def initialize_model_weights(model: nn.Module):
    """递归初始化模型权重"""

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.Embedding):
            # 使用小正态分布初始化嵌入层
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        else:
            nn.init.xavier_normal_(m.weight.data)

    model.apply(init_weights)

    # 特殊处理：padding index (0) 的嵌入置零
    # if hasattr(model, 'pos_emb') and model.pos_emb.weight.data is not None:
    #     model.pos_emb.weight.data[0, :] = 0
    # if hasattr(model, 'item_emb') and model.item_emb.weight.data is not None:
    #     model.item_emb.weight.data[0, :] = 0
    # if hasattr(model, 'user_emb') and model.user_emb.weight.data is not None:
    #     model.user_emb.weight.data[0, :] = 0
    # if hasattr(model, 'sparse_emb'):
    #     for emb_layer in model.sparse_emb.values():
    #         emb_layer.weight.data[0, :] = 0



def train_downstream_model(model, train_loader, valid_loader, args, writer, test_dataset = None, test_dataset_2=None):
    global_step = 0

    print("开始下游任务训练...")

    # ✅ 添加：学习率调度器（可选：ReduceLROnPlateau）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.downstream_task_lr, betas=(0.9, 0.98), weight_decay=getattr(args, 'weight_decay', 0.0))
    scheduler = None
    if args.use_lr_scheduler_in_downstream:
        total_steps = args.num_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps*0.01,           # 例如：1000 步 warm-up
            num_training_steps=total_steps   # 总训练步数
        )
    best_hitrate = float('inf')  # ✅ 记录最佳验证损失
    no_improve_count = 0

    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss_epoch = 0.0

        start_time = round(time.time())
        print("start time",start_time)
        for step, batch in enumerate(train_loader):
            # 解包数据
            seq, pos, neg, token_type, next_token_type, next_action_type, \
            seq_feat, pos_feat, neg_feat = batch
            # 移动到设备
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            token_type = token_type.to(args.device)
            next_token_type = next_token_type.to(args.device)
            next_action_type = next_action_type.to(args.device)

            # 前向传播
            output = model(
                seq, pos, neg, token_type, next_token_type, next_action_type,
                seq_feat, pos_feat, neg_feat
            )
            total_loss = output['total_loss'] 

            # L2 正则化（仅 item_emb）
            if args.l2_emb > 0:
                l2_reg = 0.0
                for param in model.item_emb.parameters():
                    l2_reg += torch.norm(param)
                total_loss += args.l2_emb * l2_reg

            # ✅ 检查损失是否为 NaN 或 Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"❌ 跳过 batch，损失异常（NaN/Inf） at epoch {epoch}, step {step}")
                continue

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()

            # ✅ 梯度裁剪（防止梯度爆炸）
            # if epoch>30:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 输出所有层的梯度到writer
            # for name, param in model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         writer.add_scalar(name + '_grad', param.grad.norm(), global_step)
            optimizer.step()
            if scheduler:
                scheduler.step()

            if global_step%100==0 or os.environ.get('DEBUG_MODE', "")=="True":

                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_step)
                if hasattr(model,'train_record'):
                    for record_key in model.train_record:
                        writer.add_scalar(f'Train/{record_key}', output[record_key], global_step)
            if global_step%100==0 and step!=0:
                end_time = round(time.time())
                use_time = format_time(end_time - start_time)
                remain_time = format_time((end_time - start_time)/step * len(train_loader))
                msg = f"[{use_time}/{remain_time}]"
                msg += f"[{step}/{len(train_loader)}]"
                msg += f"global_step:{global_step} "
                msg += f"epoch{epoch} "
                msg += f"total_loss:{total_loss.item():.5f} "
                msg += f"lr:{optimizer.param_groups[0]['lr']:0.5f}"
                if hasattr(model,'train_record'):
                    for record_key in model.train_record:
                        msg += f" {record_key}:{output[record_key]:0.5f}"
                print(msg + '\n')
            total_loss_epoch += total_loss.item()
            global_step += 1

        avg_train_loss = total_loss_epoch / len(train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Time: {time.time() - t0:.2f}s")

        # ========== 验证阶段 ==========
        model.eval()
        record = defaultdict(float)
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation", leave=False):
                seq, pos, neg, token_type, next_token_type, next_action_type, \
                seq_feat, pos_feat, neg_feat = batch

                seq, pos, neg, next_token_type, next_action_type = \
                    seq.to(args.device), pos.to(args.device), neg.to(args.device), next_token_type.to(args.device), next_action_type.to(args.device)
                
                output = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type,
                    seq_feat, pos_feat, neg_feat
                )
                total_loss = output['total_loss']
                record['total_loss'] = total_loss.item()
                if hasattr(model,'eval_record'):
                    for record_key in model.eval_record:
                        record[record_key] += output[record_key]
                val_batches += 1

        for key in record:
            record[key] /= val_batches
            writer.add_scalar(f'Eval_Loss/{key}', record[key], global_step)

        # ✅ 保存最佳模型

        if test_dataset:
            eval_candidate_path = os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_eval.json')
            infer = Infer(args, model, eval_dataset=test_dataset, candidate_path=eval_candidate_path)
            hitrate_eval = infer.infer()
            # 输出结果
            print("✅ 评估结果")
            print("eval:", hitrate_eval)
            writer.add_scalar('HitRat/eval', hitrate_eval, global_step)

        ckpt_dir = Path(os.environ.get('TRAIN_CKPT_PATH')) / f"global_step{global_step}.hitrate={hitrate_eval:.4f}"
        if hitrate_eval < best_hitrate:
            best_hitrate = record['total_loss']
            no_improve_count = 0
            user_dir = Path(os.environ.get('USER_CACHE_PATH')) / f"global_step{global_step}.hitrate={hitrate_eval:.4f}"
            for save_dir in [ckpt_dir, user_dir]:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir / "model.pt")
                print(f"✅ 最佳模型已保存至: {save_dir / 'model.pt'}")
        else:
            ckpt_dir = Path(os.environ.get('TRAIN_CKPT_PATH')) / f"global_step{global_step}.hitrate={hitrate_eval:.4f}"
            save_dir.mkdir(parents=True, exist_ok=True)
            no_improve_count += 1
            torch.save(model.state_dict(), save_dir / "model.pt")
            print(f"✅ 模型已保存至: {save_dir / 'model.pt'}")

        if test_dataset_2 and os.environ.get("DEBUG_MODE","")==True:
            train_candidate_path = os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_train.json')
            infer = Infer(args, model, eval_dataset=test_dataset_2, candidate_path=train_candidate_path)
            hitrate_train = infer.infer()
            print("train:", hitrate_train)
            # 写入writer
            writer.add_scalar('HitRat/train', hitrate_train, global_step)
    return model
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

    writer = SummaryWriter(os.environ['TRAIN_TF_EVENTS_PATH'])

    args = get_args()

    # 数据加载
    data_path = os.environ['TRAIN_DATA_PATH']
    base_dataset = BaseDataset(data_path, args)
    if args.train_data_size==-1:
        args.train_data_size=None
    train_idx, valid_idx = base_dataset.split_index([0.9, 0.1], args.train_data_size)

    # 构建嵌入训练与验证数据集
    usernum, itemnum = base_dataset.usernum, base_dataset.itemnum
    feat_stats, feat_types = base_dataset.feat_statistics, base_dataset.feature_types
    
    # ========== 推理 ==========
    test_dataset = ValidDataset(base_dataset, sample_index=valid_idx)  # 可替换为独立测试集
    test_dataset_2 = ValidDataset(base_dataset, sample_index=train_idx)  # 可替换为独立测试集
    eval_candidate_path = os.path.join(os.environ['TRAIN_DATA_PATH'], 'item_feat_dict.json')
    with open(eval_candidate_path, 'r', encoding='utf-8') as f:
        condidate_data = json.load(f)
    # 可以提前预知哪一个输出
    eval_candidate_index = []
    # 获取测试集的标签
    print("获取测试集的标签...")
    
    for item in test_dataset:
        seq, token_type, seq_feat, user_id, label = item
        eval_candidate_index.append(label[0])
    res_eval_condidate_data = {}
    for item in eval_candidate_index:
        res_eval_condidate_data[str(item)] = condidate_data[str(item)]
    json.dump(res_eval_condidate_data, fp=open(os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_eval.json'),'w',encoding='utf-8'),indent=4, ensure_ascii=False)
    json.dump(res_eval_condidate_data, fp=open(os.path.join(os.environ['USER_CACHE_PATH'], 'item_feat_dict_eval.json'),'w',encoding='utf-8'),indent=4, ensure_ascii=False)
    eval_candidate_path = os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_eval.json')
    print("获取测试集候选数据完成")
    train_candidate_path = None
    print("开始获取训练集候选数据")
    eval_candidate_index = []
    for item in test_dataset_2:
        seq, token_type, seq_feat, user_id, label = item
        eval_candidate_index.append(label[0])
    res_eval_condidate_data = {}
    for item in eval_candidate_index:
        res_eval_condidate_data[str(item)] = condidate_data[str(item)]
    json.dump(res_eval_condidate_data, fp=open(os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_train.json'),'w',encoding='utf-8'),indent=4, ensure_ascii=False)
    json.dump(res_eval_condidate_data, fp=open(os.path.join(os.environ['USER_CACHE_PATH'], 'item_feat_dict_train.json'),'w',encoding='utf-8'),indent=4, ensure_ascii=False)
    train_candidate_path = os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_train.json')
    print("获取训练集候选数据完成")

    print("数据集准备完成，继续下游任务训练...")

    train_dataset = TrainDataset(base_dataset, sample_index=train_idx)  # 全量训练
    valid_dataset = TrainDataset(base_dataset, sample_index=valid_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        num_workers=args.num_worker,
        persistent_workers = True,
        prefetch_factor=5,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
        pin_memory=True,
        num_workers=args.num_worker,
        persistent_workers = True,
        prefetch_factor=5,
    )

    # 模型初始化
    usernum, itemnum = base_dataset.usernum, base_dataset.itemnum
    feat_stats, feat_types = base_dataset.feat_statistics, base_dataset.feature_types

    downstream_model = DownstreamModel(usernum, itemnum, feat_stats, feat_types, args).to(args.device)
    if args.state_dict_path:
        try:
            downstream_model.load_state_dict(torch.load(args.state_dict_path, map_location=args.device))
            print(f"✅ 已加载预训练权重: {args.state_dict_path}")
        except Exception as e:
            print(f"⚠️ 权重加载失败: {e}")

    # 开始训练
    train_downstream_model(downstream_model, train_loader, valid_loader, args, writer,test_dataset, test_dataset_2)
    eval_candidate_path = os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_eval.json')
    infer = Infer(args, downstream_model, eval_dataset=test_dataset, candidate_path=eval_candidate_path)
    hitrate_eval = infer.infer()
    print("✅ 推理完成")
    train_candidate_path = os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_train.json')
    infer = Infer(args, downstream_model, eval_dataset=test_dataset_2, candidate_path=train_candidate_path)
    hitrate_train = infer.infer()

    # 输出结果
    print("✅ 评估结果")
    print("eval:", hitrate_eval)
    print("train:", hitrate_train)
    # 清理资源
    writer.close()
    
if __name__ == '__main__':
    main()


# eval: 0.47674418604651164
# train: 0.3856041131105398
# Epoch 9 | Train Loss: 3.5280 | Time: 15.93s


# -105  特征
# eval: 0.5116279069767442
# train: 0.794344473007712
# Epoch 13 | Train Loss: 2.5730 | Time: 16.76s