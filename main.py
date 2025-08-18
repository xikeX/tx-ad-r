# main.py
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
from split_embedding_model import BaselineModel as DownstreamModel
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


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练序列推荐模型（含特征嵌入）")

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='训练/验证批大小')
    # 训练emmbedding的batch_size
    parser.add_argument('--embedding_batch_size', type=int, default=64, help='训练embedding的批大小')
    parser.add_argument('--train_data_size', type=int, default=-1, help='训练数据大小')
    parser.add_argument('--embedding_task_lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--downstream_task_lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--maxlen', type=int, default=101, help='序列最大长度')

    # 模型结构参数
    parser.add_argument('--hidden_units', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--num_blocks', type=int, default=4, help='Transformer 块数')
    parser.add_argument('--num_epochs', type=int, default=25, help='训练总轮数')
    parser.add_argument('--num_heads', type=int, default=4, help='多头注意力头数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout 比例')
    parser.add_argument('--l2_emb', type=float, default=0.0, help='嵌入层L2正则强度')
    parser.add_argument('--device', type=str, default='', help='运行设备: cpu 或 cuda')
    parser.add_argument('--inference_only', action='store_true', help='仅推理模式')
    parser.add_argument('--state_dict_path', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--norm_first', action='store_true', help='是否在Transformer中先归一化')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--mm_emb_id', nargs='+', type=str, default=['81'],
                        choices=[str(s) for s in range(81, 87)],
                        help='多模态嵌入特征ID列表')
    parser.add_argument("--warm_up_rate", default=0.05, type=float, help="warm up 的比例")
    
    # 模型模块/训练模块参数
    parser.add_argument("--use_embedding_model", action="store_true", help="是否使用预训练的模型权重")
    # 使用在下游任务中使用学习率策略
    parser.add_argument("--use_lr_scheduler_in_downstream", action="store_true", help="是否使用在下游任务中使用学习率策略")

    parser.add_argument("--use_lr_scheduler_in_embedding_task", action="store_true", help="是否使用在嵌入任务中使用学习率策略")
    args = parser.parse_args()
    if args.device =='':
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


def train_embedding_model(embedding_model:EmbeddingModel, train_loader, valid_loader, optimizer, args, writer, log_file, global_step,verbose = False):
    """训练嵌入模型主循环"""
    print("开始训练嵌入模型...")
    bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    best_val_loss = float('inf')
    # 设置策略warmup 和 cosine decay策略
    scheduler=None
    if args.use_lr_scheduler_in_embedding_task:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warm_up_rate * len(train_loader)),
                                                num_training_steps=args.num_epochs * len(train_loader))
    for epoch in range(1, args.num_epochs + 1):
        embedding_model.train()
        if args.inference_only:
            print("inference_only 模式开启，跳过训练。")
            break

        t0 = time.time()
        total_loss_epoch = 0.0

        # 启用梯度缩放（用于混合精度或防止溢出）
        scaler = torch.cuda.amp.GradScaler() if args.device == 'cuda' else None
        local_step = 0
        with tqdm(train_loader,total=-1, desc=f"Epoch {epoch}", leave=False) as pbar:
            for step, batch in enumerate(pbar):
                try:
                    # 解包数据
                    user, input_item, pos_item, neg_item, \
                    user_feat, input_item_feat, pos_item_feat, neg_item_feat = batch

                    # ======== 前向传播（使用自动混合精度可选）========
                    if scaler:
                        with torch.cuda.amp.autocast():
                            user_embedding = embedding_model(user,user_feat).squeeze(0)
                            item_1_embedding = embedding_model(input_item,input_item_feat).squeeze(0)
                            pos_item_embedding = embedding_model(pos_item,pos_item_feat).squeeze(0)
                            neg_item_1_embedding = embedding_model(neg_item,neg_item_feat).squeeze(0)
                            neg_item_2_embedding = embedding_model(neg_item,neg_item_feat).squeeze(0)
                            u2i_loss,u2i_pos_sim,u2i_neg_sim = embedding_model.embedding_loss(user_embedding,item_1_embedding,neg_item_1_embedding)
                            i2i_loss,i2i_pos_sim,i2i_neg_sim = embedding_model.embedding_loss(item_1_embedding,pos_item_embedding,neg_item_2_embedding)
                            self_i2i_loss,i2i_pos_sim,i2i_neg_sim = embedding_model.embedding_loss(neg_item_1_embedding.reshape(-1,neg_item_1_embedding.shape[-1]),neg_item_1_embedding.reshape(-1,neg_item_1_embedding.shape[-1]))
                            loss = u2i_loss + i2i_loss 
                    else:
                        user_embedding = embedding_model(user,user_feat).squeeze(0)
                        item_1_embedding = embedding_model(input_item,input_item_feat).squeeze(0)
                        pos_item_embedding = embedding_model(pos_item,pos_item_feat).squeeze(0)
                        neg_item_1_embedding = embedding_model(neg_item,neg_item_feat).squeeze(0)
                        # neg_item_2_embedding = embedding_model(neg_item,neg_item_feat).squeeze(0)
                        u2i_loss,u2i_pos_sim,u2i_neg_sim = embedding_model.embedding_loss(user_embedding,item_1_embedding,neg_item_1_embedding)
                        i2i_loss,i2i_pos_sim,i2i_neg_sim = embedding_model.embedding_loss(item_1_embedding,pos_item_embedding,neg_item_1_embedding)
                        self_i2i_loss,i2i_pos_sim,i2i_neg_sim = embedding_model.embedding_loss(neg_item_1_embedding.reshape(-1,neg_item_1_embedding.shape[-1]),neg_item_1_embedding.reshape(-1,neg_item_1_embedding.shape[-1]))
                        loss = u2i_loss + i2i_loss 

                        # L2 正则化：仅作用于 item_emb，避免过强惩罚
                        if args.l2_emb > 0:
                            l2_reg = 0.0
                            for param in embedding_model.item_emb.parameters():
                                l2_reg += torch.norm(param)
                            loss += args.l2_emb * l2_reg

                    # ======== 反向传播 ========
                    optimizer.zero_grad()
                    local_step +=1
                    if scaler is not None:
                        # 混合精度反向传播
                        scaler.scale(loss).backward()
                        # 梯度裁剪（在 unscale 之后）
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(embedding_model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # 普通精度
                        loss.backward()
                        clip_grad_norm_(embedding_model.parameters(), max_norm=1.0)
                        optimizer.step()

                    # 更新学习率调度器
                    if scheduler is not None:
                        scheduler.step()

                    # ======== 日志记录 ========
                    total_loss_epoch += loss.item()

                    # 当前学习率
                    current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.embedding_task_lr

                    log_json = json.dumps({
                        'global_step': global_step,
                        'total_loss': round(loss.item(), 6),
                        'u2i_loss': round(u2i_loss.item(), 6),
                        "i2i_loss": round(i2i_loss.item(), 6),
                        'learning_rate': round(current_lr, 8),
                        'epoch': epoch,
                        'time': time.time()
                    })
                    log_file.write(log_json + '\n')
                    if verbose:
                        print(log_json)
                    log_file.flush()

                    writer.add_scalar('Emd_Loss/total_loss', loss.item(), global_step)
                    writer.add_scalar('Emd_Loss/u2i_loss', u2i_loss.item(), global_step)
                    writer.add_scalar('Emd_Loss/i2i_loss', i2i_loss.item(), global_step)
                    writer.add_scalar('Emd_Loss/self_i2i_loss', self_i2i_loss.item(), global_step)
                    writer.add_scalar('Emd_Loss/Learning_rate', current_lr, global_step)

                    global_step += 1
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})
                except Exception as e:
                    print(f"\n❌ Error at epoch {epoch}, step {step}: {str(e)}")
                    # 可选：保存当前 batch 数据用于 debug
                    # torch.save(batch, "debug_batch.pt")
                    raise  #    


        avg_train_loss = total_loss_epoch / local_step
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Time: {time.time() - t0:.2f}s")

        # 验证阶段
        embedding_model.eval()
        val_loss = 0.0
        val_ui2_loss = 0.0
        val_i2i_loss = 0.0
        val_self_i2i_loss = 0.0
        vel_local_step = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, total=-1, desc="Validation", leave=True):
                user, input_item, pos_item, neg_item, \
                user_feat, input_item_feat, pos_item_feat, neg_item_feat = batch

                user_embedding = embedding_model(user,user_feat).squeeze(0)
                item_1_embedding = embedding_model(input_item,input_item_feat).squeeze(0)
                pos_item_embedding = embedding_model(pos_item,pos_item_feat).squeeze(0)
                neg_item_1_embedding = embedding_model(neg_item,neg_item_feat).squeeze(0)
                neg_item_2_embedding = embedding_model(neg_item,neg_item_feat).squeeze(0)
                u2i_loss,u2i_pos_sim,u2i_neg_sim = embedding_model.embedding_loss(user_embedding,item_1_embedding,neg_item_1_embedding)
                i2i_loss,i2i_pos_sim,i2i_neg_sim = embedding_model.embedding_loss(item_1_embedding,pos_item_embedding,neg_item_1_embedding)
                self_i2i_loss,i2i_pos_sim,i2i_neg_sim = embedding_model.embedding_loss(neg_item_1_embedding.reshape(-1,neg_item_1_embedding.shape[-1]),neg_item_1_embedding.reshape(-1,neg_item_1_embedding.shape[-1]))
                loss = u2i_loss + i2i_loss 

                val_loss += loss.item()
                val_ui2_loss += u2i_loss.item()
                val_i2i_loss += i2i_loss.item()
                val_self_i2i_loss += self_i2i_loss.item()
                vel_local_step += 1

        val_loss /= vel_local_step
        val_ui2_loss /= vel_local_step
        val_i2i_loss /= vel_local_step
        val_self_i2i_loss /= vel_local_step
        writer.add_scalar('Emb_Valid/Loss', val_loss, global_step)
        writer.add_scalar('Emb_Valid/u2i_loss', val_ui2_loss, global_step)
        writer.add_scalar('Emb_Valid/i2i_loss', val_i2i_loss, global_step)
        writer.add_scalar('Emb_Valid/self_i2i_loss', val_self_i2i_loss, global_step)
        print(f"Epoch {epoch} | Emb_Valid Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = Path(os.environ.get('USER_CACHE_PATH')) / "global_best_embedding_model"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(embedding_model.state_dict(), save_dir / "model.pt")
            print(f"✅ 最佳模型已保存至: {save_dir / 'model.pt'}")

    return global_step


def run_embedding_training(
        args,
        train_emb_dataset,
        valid_emb_dataset,
        writer: SummaryWriter,
        log_file,
        final_save_dir,
        device=None
    ):
    """
    封装嵌入模型的完整训练流程：初始化、加载权重、训练、保存。

    Args:
        args: 训练参数命名空间，需包含以下字段：
            - batch_size
            - lr
            - state_dict_path (可选)
            - USER_CACHE_PATH
            - 其他模型相关参数
        train_emb_dataset: 训练数据集，需有 collate_fn
        valid_emb_dataset: 验证数据集，需有 collate_fn
        writer: TensorBoard SummaryWriter 实例
        log_file: 日志文件对象（已打开）
        device: 训练设备 (如 'cuda' 或 'cpu')

    Returns:
        embedding_model: 训练完成的模型
        global_step: 总训练步数
    """
    print("🚀 开始构建嵌入模型训练流程...")

    # ==========================
    # 1. 数据加载器
    # ==========================
    train_loader = DataLoader(
        train_emb_dataset,
        batch_size=args.embedding_batch_size,
        collate_fn=train_emb_dataset.collate_fn,
        shuffle=False,  # 确保训练集打乱
    )

    valid_loader = DataLoader(
        valid_emb_dataset,
        batch_size=args.batch_size,
        collate_fn=valid_emb_dataset.collate_fn,
        shuffle=False,
    )

    print(f"📊 训练集 batch 数: {len(train_loader.dataset.sample_index)}, 验证集 batch 数: {len(valid_loader.dataset.sample_index)}")

    # ==========================
    # 2. 模型初始化
    # ==========================
    embedding_model = EmbeddingModel(
        user_num=train_emb_dataset.base_dataset.usernum,
        item_num=train_emb_dataset.base_dataset.itemnum,
        feat_statistics=train_emb_dataset.base_dataset.feat_statistics,
        feat_types=train_emb_dataset.base_dataset.feature_types,
        args=args
    ).to(args.device)

    # 自定义初始化
    initialize_model_weights(embedding_model)
    print("✅ 模型权重已初始化")

    # ==========================
    # 3. 加载预训练权重（可选）
    # ==========================
    if args.state_dict_path:
        try:
            state_dict = torch.load(args.state_dict_path, map_location=args.device)
            embedding_model.load_state_dict(state_dict)
            print(f"✅ 成功加载预训练权重: {args.state_dict_path}")
            log_file.write(f"INFO: Loaded pretrained weights from {args.state_dict_path}\n")
        except Exception as e:
            msg = f"❌ 加载预训练权重失败: {args.state_dict_path}, 错误: {str(e)}"
            print(msg)
            log_file.write(f"ERROR: {msg}\n")
            raise RuntimeError(msg)

    # ==========================
    # 4. 优化器
    # ==========================
    optimizer = torch.optim.Adam(
        embedding_model.parameters(),
        lr=args.embedding_task_lr,
        betas=(0.9, 0.98),
        weight_decay=getattr(args, 'weight_decay', 0.0)  # 可选 weight decay
    )

    # ==========================
    # 5. 开始训练
    # ==========================
    global_step = 0
    print("🔥 开始训练...")
    # try:
    global_step = train_embedding_model(
        embedding_model=embedding_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        args=args,
        writer=writer,
        log_file=log_file,
        global_step=global_step
    )
    print("🎉 训练完成！")
    # except Exception as e:
    #     print(f"❌ 训练过程中发生错误: {e}")
    #     raise

    # ==========================
    # 6. 保存最终模型
    # ==========================
    
    final_save_dir.mkdir(parents=True, exist_ok=True)

    save_path = final_save_dir / "model.pt"
    torch.save(embedding_model.state_dict(), save_path)
    print(f"✅ 最终嵌入模型已保存至: {save_path}")
    log_file.write(f"INFO: Final model saved to {save_path}\n")

    return embedding_model, global_step



def train_downstream_model(model, train_loader, valid_loader, optimizer, args, log_file, writer,test_dataset = None, test_dataset_2=None):
    global_step = 0
    print("开始下游任务训练...")

    # ✅ 添加：学习率调度器（可选：ReduceLROnPlateau）
    scheduler = None
    if args.use_lr_scheduler_in_downstream:
        total_steps = args.num_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps*0.01,           # 例如：1000 步 warm-up
            num_training_steps=total_steps   # 总训练步数
        )
    best_val_loss = float('inf')  # ✅ 记录最佳验证损失
    patience = args.early_stop_patience if hasattr(args, 'early_stop_patience') else 5
    no_improve_count = 0


    for epoch in range(1, args.num_epochs + 1):
        model.train()
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
                if scheduler:scheduler.step()

                # 日志记录
                log_entry = {
                    'global_step': global_step,
                    'total_loss': total_loss.item(),
                    'epoch': epoch,
                    'time': time.time()
                }
                log_file.write(json.dumps(log_entry) + '\n')
                log_file.flush()

                # TensorBoard
                # writer.add_scalar('Loss/BCE', bce_loss.item(), global_step)
                if global_step%100==0:
                    msg = f"lr:{optimizer.param_groups[0]['lr']:0.5f}"
                    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_step)
                    if hasattr(model,'train_record'):
                        for record_key in model.train_record:
                            writer.add_scalar(f'Train/{record_key}', output[record_key], global_step)
                            msg += f" {record_key}:{output[record_key]:0.5f}"
                    print(msg + '\n')
                total_loss_epoch += total_loss.item()
                global_step += 1
                pbar.set_postfix({
                    'infoNce_loss': f'{infoNCE_loss.item():.4f}',
                })

        avg_train_loss = total_loss_epoch / len(train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Time: {time.time() - t0:.2f}s")

        # ========== 验证阶段 ==========
        model.eval()
        val_bce_loss_sum = 0.0
        val_infoNCE_loss_sum = 0.0
        val_pos_sim_sum = 0.0
        val_neg_sim_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation", leave=False):
                try:
                    seq, pos, neg, token_type, next_token_type, next_action_type, \
                    seq_feat, pos_feat, neg_feat = batch

                    seq = seq.to(args.device)
                    pos = pos.to(args.device)
                    neg = neg.to(args.device)
                    next_token_type = next_token_type.to(args.device)
                    next_action_type = next_action_type.to(args.device)

                    output = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type,
                        seq_feat, pos_feat, neg_feat
                    )
                    total_loss = output['total_loss']

                    val_infoNCE_loss_sum += infoNCE_loss.item()
                    val_pos_sim_sum += pos_sim.mean().item()
                    val_neg_sim_sum += neg_sim.mean().item()


                    # val_bce_loss_sum += bce_loss.item()
                    val_batches += 1

                except Exception as e:
                    print(f"❌ 验证异常（跳过 batch）: {e}")
                    continue

        # avg_bce_val_loss = val_bce_loss_sum / val_batches if val_batches > 0 else float('inf')
        # writer.add_scalar('Loss/BCE_loss', avg_bce_val_loss, global_step)
        avg_infoNCE_loss = val_infoNCE_loss_sum / val_batches if val_batches > 0 else float('inf') 
        avg_pos_sim = val_pos_sim_sum / val_batches if val_batches > 0 else float('inf')
        avg_neg_sim = val_neg_sim_sum / val_batches if val_batches > 0 else float('inf')
        writer.add_scalar('Eval_Loss/InfoNCE_loss', avg_infoNCE_loss, global_step)
        writer.add_scalar('Eval_Loss/Pos_sim', avg_pos_sim, global_step)
        writer.add_scalar('Eval_Loss/Neg_sim', avg_neg_sim, global_step)
        # print(f"Epoch {epoch} | Valid BCE Loss: {avg_bce_val_loss:.4f}")

        # ✅ 学习率调度
        if scheduler is not None:
            scheduler.step()

        # ✅ 保存最佳模型
        ckpt_dir = Path(os.environ.get('TRAIN_CKPT_PATH')) / f"global_step{global_step}.valid_loss={avg_infoNCE_loss:.4f}"
        # cache_dir = Path(os.environ.get('USER_CACHE_PATH')) / f"global_step{global_step}.valid_loss={avg_infoNCE_loss:.4f}"

        if avg_infoNCE_loss < best_val_loss:
            best_val_loss = avg_infoNCE_loss
            no_improve_count = 0

            for save_dir in [ckpt_dir]:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir / "model.pt")
                print(f"✅ 最佳模型已保存至: {save_dir / 'model.pt'}")
        else:
            save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={avg_infoNCE_loss:.4f}")
            save_dir.mkdir(parents=True, exist_ok=True)
            no_improve_count += 1
            torch.save(model.state_dict(), save_dir / "model.pt")
            print(f"✅ 模型已保存至: {save_dir / 'model.pt'}")
        if test_dataset:
            eval_candidate_path = os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_eval.json')
            infer = Infer(args, model, eval_dataset=test_dataset, candidate_path=eval_candidate_path)
            hitrate_eval = infer.infer()
            # train_candidate_path = os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_train.json')
            # infer = Infer(args, model, eval_dataset=test_dataset_2, candidate_path=train_candidate_path)
            # hitrate_train = infer.infer()

            # 输出结果
            print("✅ 评估结果")
            print("eval:", hitrate_eval)
            # print("train:", hitrate_train)
            # 写入writer
            writer.add_scalar('HitRat/eval', hitrate_eval, global_step)
            # writer.add_scalar('HitRat/train', hitrate_train, global_step)
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

    log_file = open(Path(os.environ['TRAIN_LOG_PATH']) / 'train.log', 'w')
    writer = SummaryWriter(os.environ['TRAIN_TF_EVENTS_PATH'])

    args = get_args()

    # 数据加载
    data_path = os.environ['TRAIN_DATA_PATH']
    base_dataset = BaseDataset(data_path, args)
    if args.train_data_size==-1:
        args.train_data_size=None
    train_idx, valid_idx = base_dataset.split_index([0.9, 0.1],args.train_data_size)

    # 构建嵌入训练与验证数据集
    train_emb_dataset = EmbeddingDataset(base_dataset, train_idx)
    valid_emb_dataset = EmbeddingDataset(base_dataset, valid_idx)
    usernum, itemnum = base_dataset.usernum, base_dataset.itemnum
    feat_stats, feat_types = base_dataset.feat_statistics, base_dataset.feature_types
    final_save_dir = Path(os.environ['USER_CACHE_PATH']) / "embedding_model_final"
    
    # ========== 推理 ==========
    test_dataset = ValidDataset(base_dataset, sample_index=valid_idx)  # 可替换为独立测试集
    test_dataset_2 = ValidDataset(base_dataset, sample_index=train_idx)  # 可替换为独立测试集
    eval_candidate_path = os.path.join(os.environ['TRAIN_DATA_PATH'], 'item_feat_dict.json')
    with open(eval_candidate_path, 'r', encoding='utf-8') as f:
        condidate_data = json.load(f)
    # 可以提前预知哪一个输出
    eval_candidate_index = []
    for item in test_dataset:
        seq, token_type, seq_feat, user_id, label = item
        eval_candidate_index.append(label[0])

    res_eval_condidate_data = {}
    for item in eval_candidate_index:
        res_eval_condidate_data[str(item)] = condidate_data[str(item)]
    json.dump(res_eval_condidate_data, fp=open(os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_eval.json'),'w',encoding='utf-8'),indent=4, ensure_ascii=False)

    eval_candidate_index = []
    for item in test_dataset_2:
        seq, token_type, seq_feat, user_id, label = item
        eval_candidate_index.append(label[0])
    eval_candidate_path = os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_eval.json')

    res_eval_condidate_data = {}
    for item in eval_candidate_index:
        res_eval_condidate_data[str(item)] = condidate_data[str(item)]
    json.dump(res_eval_condidate_data, fp=open(os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_train.json'),'w',encoding='utf-8'),indent=4, ensure_ascii=False)
    train_candidate_path = os.path.join(os.environ['TEMP_PATH'], 'item_feat_dict_train.json')
    
    if args.use_embedding_model:
        model, final_step = run_embedding_training(
            args=args,
            train_emb_dataset=train_emb_dataset,
            valid_emb_dataset=valid_emb_dataset,
            writer=writer,
            log_file=log_file,
            final_save_dir=final_save_dir,
            device=args.device
        )




    print("数据集准备完成，继续下游任务训练...")

    train_dataset = TrainDataset(base_dataset, sample_index=train_idx)  # 全量训练
    valid_dataset = TrainDataset(base_dataset, sample_index=valid_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
    )

    # 模型初始化
    usernum, itemnum = base_dataset.usernum, base_dataset.itemnum
    feat_stats, feat_types = base_dataset.feat_statistics, base_dataset.feature_types

    # 加载嵌入权重到下游模型（示例）
    downstream_model = DownstreamModel(usernum, itemnum, feat_stats, feat_types, args).to(args.device)
    # 加载预训练 embedding（可选）
    if args.state_dict_path:
        try:
            downstream_model.load_state_dict(torch.load(args.state_dict_path, map_location=args.device))
            print(f"✅ 已加载预训练权重: {args.state_dict_path}")
        except Exception as e:
            print(f"⚠️ 权重加载失败: {e}")

    if args.use_embedding_model:
        try:
            downstream_model.embedding_layer.load_state_dict(torch.load(final_save_dir / "model.pt", map_location=args.device))
            # 冻结预训练 embedding
            for name, param in downstream_model.embedding_layer.named_parameters():
                param.requires_grad = False
            print("✅ 下游模型成功加载嵌入权重")
        except Exception as e:
            print(f"❌ 下游模型加载权重失败: {e}")

    # 优化器与损失
    optimizer = torch.optim.Adam(downstream_model.parameters(), lr=args.downstream_task_lr, betas=(0.9, 0.98), weight_decay=getattr(args, 'weight_decay', 0.0))
    bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # 开始训练
    train_downstream_model(downstream_model, train_loader, valid_loader, optimizer, args, log_file, writer,test_dataset, test_dataset_2)

   
    
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
    log_file.close()
    
if __name__ == '__main__':
    main()


# eval: 0.47674418604651164
# train: 0.3856041131105398
# Epoch 9 | Train Loss: 3.5280 | Time: 15.93s


# -105  特征
# eval: 0.5116279069767442
# train: 0.794344473007712
# Epoch 13 | Train Loss: 2.5730 | Time: 16.76s