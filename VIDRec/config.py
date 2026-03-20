"""
VIDRec-MMGCN 配置文件
基于 MicroLens-50K 数据集的 MMGCN (ID + Video, Frozen Encoder) 实现
"""

# ============== 路径配置 ==============
PATHS = {
    # 原始数据路径
    'data_root': '../MicroLens-50k',
    'pairs_file': '../MicroLens-50k/MicroLens-50k_pairs.csv',
    'titles_file': '../MicroLens-50k/MicroLens-50k_titles.csv',
    'covers_dir': '../MicroLens-50k/MicroLens-50k_covers',

    # 预处理后数据保存路径
    'processed_data': './data/processed',
    'features_dir': './data/features',
    'checkpoint_dir': './checkpoints',
    'logs_dir': './logs',
    'results_dir': './results',
}

# ============== 数据集统计（预处理后填充） ==============
DATASET_STATS = {
    'num_users': None,
    'num_items': None,
    'num_interactions': None,
    'train_edges': None,
    'val_edges': None,
    'test_edges': None,
    'user_item_dict': None,
}

# ============== 模型超参数 ==============
MODEL_CONFIG = {
    # ID embedding 配置
    'dim_id': 64,           # ID embedding 维度

    # 多模态特征配置
    'dim_video_feat': 256,   # 视频封面特征维度 (ViT/SCNN 提取后)
    'dim_text_feat': 256,    # 文本特征维度 (BGE/Sentence-BERT 提取后)

    # MMGCN 配置
    'num_layers': 2,         # GCN 层数
    'aggr_mode': 'mean',     # 聚合方式: 'mean', 'max', 'add'
    'concat': False,         # 是否拼接融合: True/False

    # Frozen Encoder 配置
    'use_frozen_encoder': True,  # 冻结内容编码器
    'modal_fusion': 'mean',      # 模态融合方式: 'mean', 'weighted', 'gating'
}

# ============== 训练超参数 ==============
TRAIN_CONFIG = {
    'seed': 42,
    'batch_size': 1024,
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'lr_scheduler': 'plateau',  # 'plateau', 'step', 'cosine'
    'patience': 20,            # 早停耐心值
    'gradient_clip': 5.0,      # 梯度裁剪

    # 负采样配置
    'neg_sample_ratio': 1,     # 正负样本比例 1:neg_sample_ratio

    # 评估配置
    'eval_steps': 5,          # 每隔多少 epoch 评估一次
    'topk': [10, 20],          # 评估的 Top-K 值
    'test_batch_size': 2048,   # 测试时的 batch size
}

# ============== 特征提取配置 ==============
FEATURE_CONFIG = {
    # 文本特征提取
    'text_model': 'sentence-transformers/all-MiniLM-L6-v2',  # HuggingFace 模型
    'text_max_length': 128,
    'text_device': 'cpu',  # 或 'cuda' 如果有 GPU

    # 图像特征提取
    'image_model': 'ViT-B/32',  # OpenCLIP 模型
    'image_device': 'cpu',      # 或 'cuda' 如果有 GPU
}

# ============== 设备配置 ==============
DEVICE_CONFIG = {
    'use_cuda': True,
    'cuda_device': 0,
    'num_workers': 4,
}

# ============== 日志配置 ==============
LOG_CONFIG = {
    'log_level': 'INFO',
    'tensorboard': True,
    'save_model': True,
    'print_freq': 100,        # 每隔多少 batch 打印一次
}

def update_dataset_stats(stats: dict):
    """更新数据集统计信息"""
    DATASET_STATS.update(stats)

def get_config():
    """获取完整配置"""
    return {
        'paths': PATHS,
        'model': MODEL_CONFIG,
        'train': TRAIN_CONFIG,
        'features': FEATURE_CONFIG,
        'device': DEVICE_CONFIG,
        'log': LOG_CONFIG,
    }
