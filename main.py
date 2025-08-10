import os

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
    parser.add_argument('--batch_size', type=int, default=30, help='训练/验证批大小')
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

    return parser.parse_args()


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
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

    model.apply(init_weights)

    # 特殊处理：padding index (0) 的嵌入置零
    if hasattr(model, 'pos_emb') and model.pos_emb.weight.data is not None:
        model.pos_emb.weight.data[0, :] = 0
    if hasattr(model, 'item_emb') and model.item_emb.weight.data is not None:
        model.item_emb.weight.data[0, :] = 0
    if hasattr(model, 'user_emb') and model.user_emb.weight.data is not None:
        model.user_emb.weight.data[0, :] = 0
    if hasattr(model, 'sparse_emb'):
        for emb_layer in model.sparse_emb.values():
            emb_layer.weight.data[0, :] = 0


def train_embedding_model(embedding_model, train_loader, valid_loader, optimizer, args, writer, log_file, global_step):
    """训练嵌入模型主循环"""
    print("开始训练嵌入模型...")
    bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        embedding_model.train()
        if args.inference_only:
            print("inference_only 模式开启，跳过训练。")
            break

        t0 = time.time()
        total_loss_epoch = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch}", leave=False) as pbar:
            for step, batch in enumerate(pbar):
                user, input_item, pos_item, neg_item, \
                user_feat, input_item_feat, pos_item_feat, neg_item_feat = batch

                # 前向传播
                info_nce_loss = embedding_model(
                    user, input_item, pos_item, neg_item,
                    user_feat, input_item_feat, pos_item_feat, neg_item_feat
                )
                loss = info_nce_loss

                # L2 正则化（仅 item_emb）
                for param in embedding_model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 日志记录
                total_loss_epoch += loss.item()
                log_json = json.dumps({
                    'global_step': global_step,
                    'total_loss': loss.item(),
                    'epoch': epoch,
                    'time': time.time()
                })
                log_file.write(log_json + '\n')
                log_file.flush()
                writer.add_scalar('Loss/Train', loss.item(), global_step)

                global_step += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss_epoch / len(train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Time: {time.time() - t0:.2f}s")

        # 验证阶段
        embedding_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation", leave=False):
                user, input_item, pos_item, neg_item, \
                user_feat, input_item_feat, pos_item_feat, neg_item_feat = batch

                info_nce_loss = embedding_model(
                    user, input_item, pos_item, neg_item,
                    user_feat, input_item_feat, pos_item_feat, neg_item_feat
                )
                val_loss += info_nce_loss.item()

        val_loss /= len(valid_loader)
        writer.add_scalar('Loss/Valid', val_loss, global_step)
        print(f"Epoch {epoch} | Valid Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = Path(os.environ.get('USER_CACHE_PATH')) / "best_embedding_model"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(embedding_model.state_dict(), save_dir / "model.pt")
            print(f"✅ 最佳模型已保存至: {save_dir / 'model.pt'}")

    return global_step

def train_downstream_model(model, train_loader, valid_loader, optimizer, bce_criterion, args, writer, log_file):
    """
    下游模型训练主循环
    """
    global_step = 0
    print("开始下游任务训练...")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            print("inference_only 模式开启，跳过训练。")
            break

        t0 = time.time()
        total_loss_epoch = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch}", leave=False) as pbar:
            for step, batch in enumerate(pbar):
                # 解包数据
                seq, pos, neg, token_type, next_token_type, next_action_type, \
                seq_feat, pos_feat, neg_feat = batch

                # 移动到设备
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)

                # 前向传播
                pos_logits, neg_logits, sim_loss = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type,
                    seq_feat, pos_feat, neg_feat
                )

                # 只对 next_token_type == 1 的位置计算损失（即目标动作）
                indices = (next_token_type == 1)  # [B, L] bool mask

                # 构造标签
                pos_labels = torch.ones_like(pos_logits, device=args.device)
                neg_labels = torch.zeros_like(neg_logits, device=args.device)

                # BCE 损失：正样本接近 1，负样本接近 0
                loss_pos = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss_neg = bce_criterion(neg_logits[indices], neg_labels[indices])
                bce_loss = loss_pos + loss_neg

                # 相似性损失（如有）
                sim_loss = sim_loss.mean() * 10.0  # 缩放系数可调

                # 总损失
                total_loss = bce_loss + sim_loss

                # L2 正则化（仅 item_emb）
                if args.l2_emb > 0:
                    for param in model.item_emb.parameters():
                        total_loss += args.l2_emb * torch.norm(param)

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # 日志记录
                log_entry = {
                    'global_step': global_step,
                    'bce_loss': bce_loss.item(),
                    'sim_loss': sim_loss.item(),
                    'total_loss': total_loss.item(),
                    'epoch': epoch,
                    'time': time.time()
                }
                log_json = json.dumps(log_entry)
                log_file.write(log_json + '\n')
                log_file.flush()

                # TensorBoard 记录
                writer.add_scalar('Loss/BCE', bce_loss.item(), global_step)
                writer.add_scalar('Loss/Sim', sim_loss.item(), global_step)
                writer.add_scalar('Loss/Total', total_loss.item(), global_step)

                total_loss_epoch += total_loss.item()
                global_step += 1

                pbar.set_postfix({
                    'bce': f'{bce_loss.item():.4f}',
                    'sim': f'{sim_loss.item():.4f}'
                })

        avg_train_loss = total_loss_epoch / len(train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Time: {time.time() - t0:.2f}s")

        # ========== 验证阶段 ==========
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation", leave=False):
                seq, pos, neg, token_type, next_token_type, next_action_type, \
                seq_feat, pos_feat, neg_feat = batch

                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)

                pos_logits, neg_logits, sim_loss = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type,
                    seq_feat, pos_feat, neg_feat
                )

                indices = (next_token_type == 1)
                pos_labels = torch.ones_like(pos_logits, device=args.device)
                neg_labels = torch.zeros_like(neg_logits, device=args.device)

                loss_pos = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss_neg = bce_criterion(neg_logits[indices], neg_labels[indices])
                bce_loss = loss_pos + loss_neg

                val_loss_sum += bce_loss.item()

        avg_val_loss = val_loss_sum / len(valid_loader)
        writer.add_scalar('Loss/Valid', avg_val_loss, global_step)
        print(f"Epoch {epoch} | Valid BCE Loss: {avg_val_loss:.4f}")

        # ========== 模型保存 ==========
        ckpt_dir = Path(os.environ.get('TRAIN_CKPT_PATH')) / f"global_step{global_step}.valid_loss={avg_val_loss:.4f}"
        cache_dir = Path(os.environ.get('USER_CACHE_PATH')) / f"global_step{global_step}_sim_model.valid_loss={avg_val_loss:.4f}"

        for save_dir in [ckpt_dir, cache_dir]:
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")
            print(f"✅ 模型已保存至: {save_dir / 'model.pt'}")

    return global_step

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

    # 构建嵌入训练与验证数据集
    train_emb_dataset = EmbeddingDataset(base_dataset, train_idx)
    valid_emb_dataset = EmbeddingDataset(base_dataset, valid_idx)

    train_loader = DataLoader(
        train_emb_dataset,
        batch_size=128,
        num_workers=0,
        collate_fn=train_emb_dataset.collate_fn,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_emb_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=valid_emb_dataset.collate_fn,
        shuffle=False
    )

    # 模型初始化
    usernum, itemnum = base_dataset.usernum, base_dataset.itemnum
    feat_stats, feat_types = base_dataset.feat_statistics, base_dataset.feature_types

    embedding_model = EmbeddingModel(usernum, itemnum, feat_stats, feat_types, args).to(device)
    print("Embedding Model:\n", embedding_model)

    initialize_model_weights(embedding_model)

    # 加载预训练权重（如有）
    if args.state_dict_path:
        try:
            embedding_model.load_state_dict(torch.load(args.state_dict_path, map_location=device))
            print(f"✅ 成功加载预训练权重: {args.state_dict_path}")
        except Exception as e:
            print(f"❌ 加载权重失败: {args.state_dict_path}")
            raise RuntimeError(f"权重加载失败: {e}")

    # 优化器
    optimizer = torch.optim.Adam(
        embedding_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98)
    )

    # 开始训练
    global_step = 0
    global_step = train_embedding_model(
        embedding_model, train_loader, valid_loader, optimizer,
        args, writer, log_file, global_step
    )

    # 保存最终模型
    final_save_dir = Path(os.environ['USER_CACHE_PATH']) / "embedding_model_final"
    final_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embedding_model.state_dict(), final_save_dir / "model.pt")
    print(f"✅ 最终嵌入模型已保存至: {final_save_dir / 'model.pt'}")

    # 加载嵌入权重到下游模型（示例）
    downstream_model = DownstreamModel(usernum, itemnum, feat_stats, feat_types, args).to(device)
    print("Downstream Model:\n", downstream_model)

    try:
        downstream_model.load_state_dict(torch.load(final_save_dir / "model.pt", map_location=device))
        print("✅ 下游模型成功加载嵌入权重")
    except Exception as e:
        print(f"❌ 下游模型加载权重失败: {e}")



    print("数据集准备完成，继续下游任务训练...")

    train_dataset = TrainDataset(base_dataset, sample_index=train_idx)  # 全量训练
    valid_dataset = TrainDataset(base_dataset, sample_index=valid_idx)
    test_dataset = ValidDataset(base_dataset, sample_index=valid_idx)  # 可替换为独立测试集

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=train_dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=valid_dataset.collate_fn
    )

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

    # 优化器与损失
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # 开始训练
    train_downstream_model(model, train_loader, valid_loader, optimizer, bce_criterion, args, writer, log_file)

    # ========== 推理 ==========
    candidate_path = os.path.join(os.environ['TRAIN_DATA_PATH'], 'item_feat_dict.json')
    infer = Infer(args, model, eval_dataset=test_dataset, candidate_path=candidate_path)
    infer.infer()
    print("✅ 推理完成")

    # 清理资源
    writer.close()
    log_file.close()
    
    # 清理资源
    log_file.close()
    writer.close()
if __name__ == '__main__':
    main()
    
    