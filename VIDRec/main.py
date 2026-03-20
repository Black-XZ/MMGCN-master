"""
VIDRec-MMGCN 主训练脚本
基于 MicroLens-50K 数据集的 MMGCN (ID + Video, Frozen Encoder) 模型

使用方法:
    # 1. 数据预处理
    python main.py --mode preprocess

    # 2. 训练模型（使用默认超参数）
    python main.py --mode train

    # 3. 训练模型（自定义超参数）
    python main.py --mode train --dim_id 128 --num_layers 3 --lr 0.0005

    # 4. 评估模型
    python main.py --mode eval
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PATHS, MODEL_CONFIG, TRAIN_CONFIG, DEVICE_CONFIG, FEATURE_CONFIG, LOG_CONFIG
from utils.data_loader import MicroLens50kDataset, load_processed_data
from utils.train_eval import BPRTrainDataset, train_epoch, evaluate, EarlyStopping
from models.mmgcn_vidrec import build_mmgcn_vidrec


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def preprocess_data(args):
    """
    数据预处理
    """
    print("\n" + "="*60)
    print("Step 1: Data Preprocessing")
    print("="*60 + "\n")

    dataset = MicroLens50kDataset(
        data_root=str(PATHS['data_root']),
        pairs_file=str(PATHS['pairs_file']),
        titles_file=str(PATHS['titles_file']),
        covers_dir=str(PATHS['covers_dir']),
        output_dir=str(PROJECT_ROOT / PATHS['processed_data']),
        features_dir=str(PROJECT_ROOT / PATHS['features_dir']),
        seed=TRAIN_CONFIG['seed'],
        feature_config=FEATURE_CONFIG
    )

    stats = dataset.process_all(
        text_feat_dim=MODEL_CONFIG['dim_text_feat'],
        image_feat_dim=MODEL_CONFIG['dim_video_feat'],
        val_ratio=0.1,
        test_ratio=0.1
    )

    print("\nPreprocessing completed!")
    print("Dataset Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats


def train_model(args):
    """
    训练模型
    """
    print("\n" + "="*60)
    print("Step 2: Model Training")
    print("="*60 + "\n")

    # 设置随机种子
    set_seed(TRAIN_CONFIG['seed'])

    # 设置设备
    device = torch.device(
        f"cuda:{DEVICE_CONFIG['cuda_device']}" if torch.cuda.is_available() and DEVICE_CONFIG['use_cuda'] else "cpu"
    )
    print(f"Using device: {device}")

    # 加载数据
    print("\nLoading processed data...")
    data = load_processed_data(str(PROJECT_ROOT / PATHS['processed_data']))

    print(f"  - Users: {data['num_users']}")
    print(f"  - Items: {data['num_items']}")
    print(f"  - Train edges: {len(data['train_edges'])}")
    print(f"  - Val edges: {len(data['val_edges'])}")
    print(f"  - Test edges: {len(data['test_edges'])}")
    print(f"  - Text features: {data['text_features'].shape}")
    print(f"  - Image features: {data['image_features'].shape}")

    # 准备模型数据
    model_data = {
        'num_users': data['num_users'],
        'num_items': data['num_items'],
        'train_edges': data['train_edges'],
        'v_feat': data['image_features'],  # 视觉特征（封面图像）
        't_feat': data['text_features'],    # 文本特征（标题）
    }

    # 构建模型
    print("\nBuilding MMGCN-VIDRec model...")
    model = build_mmgcn_vidrec(MODEL_CONFIG, model_data)
    model = model.to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 冻结特征检查
    if MODEL_CONFIG['use_frozen_encoder']:
        print("  Frozen Encoder: ENABLED (content features are NOT trainable)")
        # 确保特征参数不更新
        for name, param in model.named_parameters():
            if 'v_feat' in name or 't_feat' in name:
                param.requires_grad = False
    else:
        print("  Frozen Encoder: DISABLED (content features are trainable)")

    # 创建优化器
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    # 创建数据加载器
    train_dataset = BPRTrainDataset(
        num_users=data['num_users'],
        num_items=data['num_items'],
        train_edges=data['train_edges'],
        user_item_dict=data['user_item_dict'],
        neg_sample_ratio=TRAIN_CONFIG['neg_sample_ratio']
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=DEVICE_CONFIG['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 早停
    early_stopping = EarlyStopping(
        patience=TRAIN_CONFIG['patience'],
        mode='max'
    )

    # 训练循环
    print("\nStarting training...")
    best_metric = 0.0
    best_epoch = 0

    for epoch in range(1, TRAIN_CONFIG['num_epochs'] + 1):
        epoch_start_time = time.time()

        # 训练
        train_metrics = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            gradient_clip=TRAIN_CONFIG['gradient_clip'],
            print_freq=LOG_CONFIG['print_freq']
        )

        epoch_time = time.time() - epoch_start_time

        print(f"\nEpoch {epoch}/{TRAIN_CONFIG['num_epochs']} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Train BPR Loss: {train_metrics['bpr_loss']:.4f}")

        # 评估
        if epoch % TRAIN_CONFIG['eval_steps'] == 0:
            val_results = evaluate(
                model=model,
                test_data=data['val_full'],
                user_item_dict_train=data['user_item_dict'],
                num_users=data['num_users'],
                num_items=data['num_items'],
                topk=TRAIN_CONFIG['topk'],
                batch_size=TRAIN_CONFIG['test_batch_size']
            )

            test_results = evaluate(
                model=model,
                test_data=data['test_full'],
                user_item_dict_train=data['user_item_dict'],
                num_users=data['num_users'],
                num_items=data['num_items'],
                topk=TRAIN_CONFIG['topk'],
                batch_size=TRAIN_CONFIG['test_batch_size']
            )

            print("\n  Validation Results:")
            for metric, value in val_results.items():
                print(f"    {metric}: {value:.4f}")

            print("\n  Test Results:")
            for metric, value in test_results.items():
                print(f"    {metric}: {value:.4f}")

            # 使用 HR@10 作为早停指标
            current_metric = val_results.get('HR@10', val_results.get('HR@20', 0))

            # 学习率调度
            scheduler.step(current_metric)

            # 早停检查
            if early_stopping(current_metric):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

            # 保存最佳模型
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch

                checkpoint_dir = PROJECT_ROOT / PATHS['checkpoint_dir']
                checkpoint_dir.mkdir(exist_ok=True)

                checkpoint_path = checkpoint_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric': best_metric,
                    'config': {
                        'model': MODEL_CONFIG,
                        'train': TRAIN_CONFIG,
                    }
                }, checkpoint_path)

                print(f"\n  Best model saved! (HR@10: {best_metric:.4f})")

    print(f"\nTraining completed!")
    print(f"Best epoch: {best_epoch}, Best HR@10: {best_metric:.4f}")

    return model, best_metric


def evaluate_model(args):
    """
    评估模型
    """
    print("\n" + "="*60)
    print("Step 3: Model Evaluation")
    print("="*60 + "\n")

    # 设置设备
    device = torch.device(
        f"cuda:{DEVICE_CONFIG['cuda_device']}" if torch.cuda.is_available() and DEVICE_CONFIG['use_cuda'] else "cpu"
    )
    print(f"Using device: {device}")

    # 加载数据
    print("\nLoading data...")
    data = load_processed_data(str(PROJECT_ROOT / PATHS['processed_data']))

    # 加载模型
    checkpoint_path = PROJECT_ROOT / PATHS['checkpoint_dir'] / 'best_model.pt'

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python main.py --mode train")
        return

    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 构建模型
    model_data = {
        'num_users': data['num_users'],
        'num_items': data['num_items'],
        'train_edges': data['train_edges'],
        'v_feat': data['image_features'],
        't_feat': data['text_features'],
    }

    model = build_mmgcn_vidrec(checkpoint['config']['model'], model_data)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best metric: {checkpoint['best_metric']:.4f}")

    # 评估
    print("\nEvaluating on test set...")

    test_results = evaluate(
        model=model,
        test_data=data['test_full'],
        user_item_dict_train=data['user_item_dict'],
        num_users=data['num_users'],
        num_items=data['num_items'],
        topk=TRAIN_CONFIG['topk'],
        batch_size=TRAIN_CONFIG['test_batch_size']
    )

    print("\nTest Results:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.4f}")

    # 保存结果
    results_dir = PROJECT_ROOT / PATHS['results_dir']
    results_dir.mkdir(exist_ok=True)

    import json
    results_file = results_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    return test_results


def parse_override_args():
    """
    解析命令行中的超参数覆盖
    这些参数会覆盖 config.py 中的默认配置
    """
    parser = argparse.ArgumentParser(add_help=False)

    # ============ 模型超参数 ============
    parser.add_argument('--dim_id', type=int, default=None)
    parser.add_argument('--dim_latent', type=int, default=None)
    parser.add_argument('--dim_video_feat', type=int, default=None)
    parser.add_argument('--dim_text_feat', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--aggr_mode', type=str, default=None)
    parser.add_argument('--concat', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--modal_fusion', type=str, default=None)
    parser.add_argument('--use_frozen_encoder', type=lambda x: x.lower() == 'true', default=None)

    # ============ 训练超参数 ============
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', '--learning_rate', dest='learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--neg_sample_ratio', type=int, default=None)

    # ============ 评估超参数 ============
    parser.add_argument('--topk', type=str, default=None)  # 格式: "10,20"
    parser.add_argument('--eval_steps', type=int, default=None)

    # ============ 设备配置 ============
    parser.add_argument('--device', type=str, default=None)  # 'cuda' 或 'cpu'
    parser.add_argument('--cuda_device', type=int, default=None)

    # ============ 特征提取配置 ============
    parser.add_argument('--text_model', type=str, default=None)
    parser.add_argument('--image_model', type=str, default=None)
    parser.add_argument('--text_device', type=str, default=None)
    parser.add_argument('--image_device', type=str, default=None)

    return parser


def merge_config_with_args(config: dict, args) -> dict:
    """
    将命令行参数合并到配置字典中
    只覆盖明确指定的参数（不为 None 的参数）
    """
    result = {k: v for k, v in config.items()}  # 深拷贝

    # 模型配置
    if hasattr(args, 'dim_id') and args.dim_id is not None:
        result['dim_id'] = args.dim_id
    if hasattr(args, 'dim_latent') and args.dim_latent is not None:
        result['dim_latent'] = args.dim_latent
    if hasattr(args, 'dim_video_feat') and args.dim_video_feat is not None:
        result['dim_video_feat'] = args.dim_video_feat
    if hasattr(args, 'dim_text_feat') and args.dim_text_feat is not None:
        result['dim_text_feat'] = args.dim_text_feat
    if hasattr(args, 'num_layers') and args.num_layers is not None:
        result['num_layers'] = args.num_layers
    if hasattr(args, 'aggr_mode') and args.aggr_mode is not None:
        result['aggr_mode'] = args.aggr_mode
    if hasattr(args, 'concat') and args.concat is not None:
        result['concat'] = args.concat
    if hasattr(args, 'modal_fusion') and args.modal_fusion is not None:
        result['modal_fusion'] = args.modal_fusion
    if hasattr(args, 'use_frozen_encoder') and args.use_frozen_encoder is not None:
        result['use_frozen_encoder'] = args.use_frozen_encoder

    return result


def merge_train_config_with_args(config: dict, args) -> dict:
    """
    将命令行参数合并到训练配置字典中
    """
    result = {k: v for k, v in config.items()}  # 深拷贝

    if hasattr(args, 'batch_size') and args.batch_size is not None:
        result['batch_size'] = args.batch_size
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        result['learning_rate'] = args.learning_rate
    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        result['weight_decay'] = args.weight_decay
    if hasattr(args, 'num_epochs') and args.num_epochs is not None:
        result['num_epochs'] = args.num_epochs
    if hasattr(args, 'patience') and args.patience is not None:
        result['patience'] = args.patience
    if hasattr(args, 'gradient_clip') and args.gradient_clip is not None:
        result['gradient_clip'] = args.gradient_clip
    if hasattr(args, 'neg_sample_ratio') and args.neg_sample_ratio is not None:
        result['neg_sample_ratio'] = args.neg_sample_ratio
    if hasattr(args, 'topk') and args.topk is not None:
        result['topk'] = [int(k) for k in args.topk.split(',')]
    if hasattr(args, 'eval_steps') and args.eval_steps is not None:
        result['eval_steps'] = args.eval_steps

    return result


def merge_device_config_with_args(config: dict, args) -> dict:
    """
    将命令行参数合并到设备配置字典中
    """
    result = {k: v for k, v in config.items()}

    if hasattr(args, 'device') and args.device is not None:
        result['use_cuda'] = (args.device.lower() == 'cuda')
    if hasattr(args, 'cuda_device') and args.cuda_device is not None:
        result['cuda_device'] = args.cuda_device

    return result


def merge_feature_config_with_args(config: dict, args) -> dict:
    """
    将命令行参数合并到特征提取配置字典中
    """
    result = {k: v for k, v in config.items()}

    if hasattr(args, 'text_model') and args.text_model is not None:
        result['text_model'] = args.text_model
    if hasattr(args, 'image_model') and args.image_model is not None:
        result['image_model'] = args.image_model
    if hasattr(args, 'text_device') and args.text_device is not None:
        result['text_device'] = args.text_device
    if hasattr(args, 'image_device') and args.image_device is not None:
        result['image_device'] = args.image_device

    return result


def main():
    """主入口函数"""
    # 解析已知参数
    known_parser = parse_override_args()
    known_args, unknown_args = known_parser.parse_known_args()

    # 解析主参数
    parser = argparse.ArgumentParser(
        description='VIDRec-MMGCN: Multi-modal Graph Convolution Network for Video Recommendation',
        parents=[known_parser]
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['preprocess', 'train', 'eval'],
        help='Running mode: preprocess data, train model, or evaluate model'
    )

    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Data root directory'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file for evaluation'
    )

    args = parser.parse_args()

    # 打印使用的配置
    print("\n" + "="*60)
    print("VIDRec-MMGCN Configuration")
    print("="*60)

    # 显示命令行覆盖的配置
    overrides = []
    for attr in dir(args):
        if not attr.startswith('_') and getattr(args, attr) is not None:
            if attr in ['mode', 'data_root', 'checkpoint']:
                continue
            overrides.append(f"  --{attr} {getattr(args, attr)}")

    if overrides:
        print("\n[Override Parameters]:")
        for o in overrides:
            print(o)
    else:
        print("\n[Using default config.py settings]")

    # 全局更新配置
    global MODEL_CONFIG, TRAIN_CONFIG, DEVICE_CONFIG, FEATURE_CONFIG, PATHS

    MODEL_CONFIG = merge_config_with_args(MODEL_CONFIG, args)
    TRAIN_CONFIG = merge_train_config_with_args(TRAIN_CONFIG, args)
    DEVICE_CONFIG = merge_device_config_with_args(DEVICE_CONFIG, args)
    FEATURE_CONFIG = merge_feature_config_with_args(FEATURE_CONFIG, args)

    print("\n[Final Model Config]:")
    for k, v in MODEL_CONFIG.items():
        print(f"  {k}: {v}")

    print("\n[Final Train Config]:")
    for k, v in TRAIN_CONFIG.items():
        print(f"  {k}: {v}")

    print("\n[Final Device Config]:")
    for k, v in DEVICE_CONFIG.items():
        print(f"  {k}: {v}")

    print("\n[Final Feature Config]:")
    for k, v in FEATURE_CONFIG.items():
        print(f"  {k}: {v}")
    print("="*60 + "\n")

    # 确保在正确的工作目录下
    os.chdir(PROJECT_ROOT)

    if args.mode == 'preprocess':
        preprocess_data(args)
    elif args.mode == 'train':
        train_model(args)
    elif args.mode == 'eval':
        evaluate_model(args)


if __name__ == '__main__':
    main()
