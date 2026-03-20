"""
数据预处理模块
负责 MicroLens-50K 数据集的加载、划分和特征提取
"""

import os
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

# 尝试导入特征提取依赖（可选）
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("[WARNING] sentence-transformers not installed. Text features will use random initialization.")

try:
    from PIL import Image
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("[WARNING] CLIP not installed. Image features will use random initialization.")


class MicroLens50kDataset:
    """
    MicroLens-50K 数据集处理类
    """

    def __init__(
        self,
        data_root: str,
        pairs_file: str,
        titles_file: str,
        covers_dir: str,
        output_dir: str,
        features_dir: str,
        seed: int = 42,
        feature_config: dict = None
    ):
        self.data_root = Path(data_root)
        self.pairs_file = Path(pairs_file)
        self.titles_file = Path(titles_file)
        self.covers_dir = Path(covers_dir)
        self.output_dir = Path(output_dir)
        self.features_dir = Path(features_dir)
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

        self.num_users = None
        self.num_items = None
        self.user_item_dict = defaultdict(set)
        self.all_items = set()

        # 特征提取配置
        self.feature_config = feature_config or {}

    def load_interactions(self) -> pd.DataFrame:
        """
        加载用户-视频交互数据
        格式: user, item, timestamp
        """
        print("[1/5] Loading interaction data...")
        df = pd.read_csv(self.pairs_file)

        print(f"  - Total interactions: {len(df)}")
        print(f"  - Unique users: {df['user'].nunique()}")
        print(f"  - Unique items: {df['item'].nunique()}")

        # 按时间排序
        df = df.sort_values('timestamp')

        # 重新映射 ID（确保连续）
        self.all_users = sorted(df['user'].unique())
        self.all_items = sorted(df['item'].unique())

        self.user_id_map = {u: i for i, u in enumerate(self.all_users)}
        self.item_id_map = {v: i for i, v in enumerate(self.all_items)}
        self.id_user_map = {i: u for u, i in self.user_id_map.items()}
        self.id_item_map = {i: v for v, i in self.item_id_map.items()}

        df['user_idx'] = df['user'].map(self.user_id_map)
        df['item_idx'] = df['item'].map(self.item_id_map)

        self.num_users = len(self.user_id_map)
        self.num_items = len(self.item_id_map)

        # 构建 user-item 交互字典
        for _, row in df.iterrows():
            self.user_item_dict[row['user_idx']].add(row['item_idx'])

        print(f"  - Mapped users: {self.num_users}")
        print(f"  - Mapped items: {self.num_items}")

        return df[['user_idx', 'item_idx', 'timestamp']]

    def load_titles(self) -> Dict[int, str]:
        """
        加载视频标题数据
        """
        print("[2/5] Loading title data...")
        titles_df = pd.read_csv(self.titles_file)

        titles = {}
        for _, row in titles_df.iterrows():
            item_id = row['item']
            if item_id in self.item_id_map:
                titles[self.item_id_map[item_id]] = str(row['title'])

        # 对于缺失标题的 item，使用空字符串
        for i in range(self.num_items):
            if i not in titles:
                titles[i] = ""

        print(f"  - Loaded {len(titles)} titles")
        return titles

    def extract_text_features(
        self,
        titles: Dict[int, str],
        feat_dim: int = 256,
    ) -> np.ndarray:
        """
        提取文本特征（标题）
        使用 Sentence-BERT 或随机初始化
        """
        print("[3/5] Extracting text features...")

        feat_path = self.features_dir / 'text_features.npy'

        if feat_path.exists():
            print(f"  - Loading cached text features from {feat_path}")
            features = np.load(feat_path)
            return features

        # 获取配置
        model_name = self.feature_config.get('text_model', 'sentence-transformers/all-MiniLM-L6-v2')
        device = self.feature_config.get('text_device', 'cpu')

        if HAS_SENTENCE_TRANSFORMERS:
            print(f"  - Using Sentence-BERT: {model_name}")
            model = SentenceTransformer(model_name, device=device)

            texts = [titles.get(i, "") for i in range(self.num_items)]
            features = model.encode(
                texts,
                batch_size=256,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            # 投影到目标维度
            if features.shape[1] != feat_dim:
                proj = np.random.randn(features.shape[1], feat_dim).astype(np.float32)
                proj = proj / np.linalg.norm(proj, axis=0)
                features = features @ proj

        else:
            print("  - [FALLBACK] Using random text features (install sentence-transformers for real features)")
            features = np.random.randn(self.num_items, feat_dim).astype(np.float32)
            features = features / np.linalg.norm(features, axis=1, keepdims=True)

        np.save(feat_path, features)
        print(f"  - Text features shape: {features.shape}")
        return features

    def extract_image_features(
        self,
        feat_dim: int = 256,
    ) -> np.ndarray:
        """
        提取图像特征（视频封面）
        使用 CLIP 或随机初始化
        """
        print("[4/5] Extracting image features...")

        feat_path = self.features_dir / 'image_features.npy'

        if feat_path.exists():
            print(f"  - Loading cached image features from {feat_path}")
            features = np.load(feat_path)
            return features

        # 获取配置
        model_name = self.feature_config.get('image_model', 'ViT-B/32')
        device = self.feature_config.get('image_device', 'cpu')

        if HAS_CLIP:
            print(f"  - Using CLIP for image features: {model_name}")
            model, preprocess = clip.load(model_name, device=device)
            model.eval()

            features = np.zeros((self.num_items, 512), dtype=np.float32)

            for item_idx in range(self.num_items):
                orig_id = self.id_item_map.get(item_idx)
                if orig_id is None:
                    continue

                img_path = self.covers_dir / f"{orig_id}.jpg"
                if not img_path.exists():
                    continue

                try:
                    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = model.encode_image(image)
                        feat = feat.cpu().numpy().squeeze()

                    features[item_idx] = feat
                except Exception:
                    continue

                if item_idx % 5000 == 0:
                    print(f"  - Processed {item_idx}/{self.num_items} images")

            # 投影到目标维度
            if features.shape[1] != feat_dim:
                proj = np.random.randn(features.shape[1], feat_dim).astype(np.float32)
                proj = proj / np.linalg.norm(proj, axis=0)
                features = features @ proj

            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

        else:
            print("  - [FALLBACK] Using random image features (install CLIP for real features)")
            features = np.random.randn(self.num_items, feat_dim).astype(np.float32)
            features = features / np.linalg.norm(features, axis=1, keepdims=True)

        np.save(feat_path, features)
        print(f"  - Image features shape: {features.shape}")
        return features

    def split_data(
        self,
        df: pd.DataFrame,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        按时间划分训练集、验证集、测试集
        """
        print("[5/5] Splitting data (temporal split)...")

        # 每个用户按时间排序，取最后 val_ratio + test_ratio 作为测试
        train_edges = []
        val_edges = []
        test_edges = []

        for user_idx in range(self.num_users):
            user_items = sorted(df[df['user_idx'] == user_idx]['item_idx'].tolist())

            if len(user_items) < 3:
                # 交互太少的用户，全部作为训练集
                train_edges.extend([(user_idx, item) for item in user_items])
                continue

            n_items = len(user_items)
            n_val = max(1, int(n_items * val_ratio))
            n_test = max(1, int(n_items * test_ratio))

            # 按时间顺序：train -> val -> test
            train_items = user_items[:-(n_val + n_test)]
            val_items = user_items[-(n_val + n_test):-n_test] if n_test > 0 else user_items[-n_val:]
            test_items = user_items[-n_test:]

            train_edges.extend([(user_idx, item) for item in train_items])
            val_edges.extend([(user_idx, item) for item in val_items])
            test_edges.extend([(user_idx, item) for item in test_items])

        train_edges = np.array(train_edges)
        val_edges = np.array(val_edges)
        test_edges = np.array(test_edges)

        print(f"  - Train edges: {len(train_edges)}")
        print(f"  - Val edges: {len(val_edges)}")
        print(f"  - Test edges: {len(test_edges)}")

        # 更新 user_item_dict 为训练集
        self.user_item_dict_train = defaultdict(set)
        for u, i in train_edges:
            self.user_item_dict_train[u].add(i)

        return train_edges, val_edges, test_edges

    def save_processed_data(
        self,
        train_edges: np.ndarray,
        val_edges: np.ndarray,
        test_edges: np.ndarray,
        text_features: np.ndarray,
        image_features: np.ndarray
    ):
        """保存预处理后的数据"""
        print("Saving processed data...")

        # 保存边
        np.save(self.output_dir / 'train_edges.npy', train_edges)
        np.save(self.output_dir / 'val_edges.npy', val_edges)
        np.save(self.output_dir / 'test_edges.npy', test_edges)

        # 保存特征
        np.save(self.features_dir / 'text_features.npy', text_features)
        np.save(self.features_dir / 'image_features.npy', image_features)

        # 保存映射关系
        with open(self.output_dir / 'id_mappings.pkl', 'wb') as f:
            pickle.dump({
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'id_user_map': self.id_user_map,
                'id_item_map': self.id_item_map,
                'num_users': self.num_users,
                'num_items': self.num_items,
            }, f)

        # 保存 user_item_dict
        with open(self.output_dir / 'user_item_dict.pkl', 'wb') as f:
            pickle.dump(dict(self.user_item_dict_train), f)

        # 保存 val/test 的完整交互用于评估
        val_full = self._build_full_edges(val_edges)
        test_full = self._build_full_edges(test_edges)
        np.save(self.output_dir / 'val_full.npy', val_full)
        np.save(self.output_dir / 'test_full.npy', test_full)

        print(f"  - Saved to {self.output_dir}")

    def _build_full_edges(self, edges: np.ndarray) -> np.ndarray:
        """
        构建评估用的完整边列表
        每行: [user_idx, pos_item_1, pos_item_2, ...]
        """
        user_pos_items = defaultdict(list)
        for u, i in edges:
            user_pos_items[u].append(i)

        result = []
        for u in sorted(user_pos_items.keys()):
            items = [u] + user_pos_items[u]
            result.append(np.array(items))

        return np.array(result, dtype=object)

    def process_all(
        self,
        text_feat_dim: int = 256,
        image_feat_dim: int = 256,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict:
        """
        执行完整的数据处理流程
        """
        print("\n" + "="*60)
        print("MicroLens-50K Data Processing Pipeline")
        print("="*60 + "\n")

        start_time = time.time()

        # 1. 加载交互数据
        df = self.load_interactions()

        # 2. 加载标题数据
        titles = self.load_titles()

        # 3. 提取文本特征
        text_features = self.extract_text_features(
            titles, feat_dim=text_feat_dim
        )

        # 4. 提取图像特征
        image_features = self.extract_image_features(feat_dim=image_feat_dim)

        # 5. 划分数据集
        train_edges, val_edges, test_edges = self.split_data(
            df, val_ratio, test_ratio
        )

        # 6. 保存处理后的数据
        self.save_processed_data(
            train_edges, val_edges, test_edges,
            text_features, image_features
        )

        elapsed = time.time() - start_time
        print(f"\n[Done] Data processing completed in {elapsed:.2f}s")

        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_train': len(train_edges),
            'num_val': len(val_edges),
            'num_test': len(test_edges),
            'text_features_shape': text_features.shape,
            'image_features_shape': image_features.shape,
        }


class VIDRecDataset(Dataset):
    """
    VIDRec 训练数据集
    """

    def __init__(
        self,
        edges: np.ndarray,
        num_users: int,
        num_items: int,
        user_item_dict: Dict[int, set],
        all_items: set,
        neg_sample_ratio: int = 1
    ):
        self.edges = edges
        self.num_users = num_users
        self.num_items = num_items
        self.user_item_dict = user_item_dict
        self.all_items = all_items
        self.neg_sample_ratio = neg_sample_ratio

    def __len__(self):
        return len(self.edges) * (1 + self.neg_sample_ratio)

    def __getitem__(self, idx):
        edge_idx = idx // (1 + self.neg_sample_ratio)
        is_pos = (idx % (1 + self.neg_sample_ratio)) == 0

        user, pos_item = self.edges[edge_idx]

        if is_pos:
            item = pos_item
        else:
            # 负采样：不在用户交互历史中的 item
            while True:
                neg_item = random.randint(0, self.num_items - 1)
                if neg_item not in self.user_item_dict.get(user, set()):
                    break
            item = neg_item

        return torch.LongTensor([user]), torch.LongTensor([item])


def load_processed_data(data_dir: str) -> Dict:
    """
    加载预处理后的数据
    """
    data_dir = Path(data_dir)
    features_dir = data_dir.parent / 'features'

    # 加载映射
    with open(data_dir / 'id_mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)

    # 加载边
    train_edges = np.load(data_dir / 'train_edges.npy')
    val_edges = np.load(data_dir / 'val_edges.npy')
    test_edges = np.load(data_dir / 'test_edges.npy')

    # 加载特征
    text_features = np.load(features_dir / 'text_features.npy')
    image_features = np.load(features_dir / 'image_features.npy')

    # 加载 user_item_dict
    with open(data_dir / 'user_item_dict.pkl', 'rb') as f:
        user_item_dict = pickle.load(f)

    # 加载 val/test 完整边
    val_full = np.load(data_dir / 'val_full.npy', allow_pickle=True)
    test_full = np.load(data_dir / 'test_full.npy', allow_pickle=True)

    return {
        'num_users': mappings['num_users'],
        'num_items': mappings['num_items'],
        'train_edges': train_edges,
        'val_edges': val_edges,
        'test_edges': test_edges,
        'val_full': val_full,
        'test_full': test_full,
        'text_features': text_features,
        'image_features': image_features,
        'user_item_dict': user_item_dict,
    }


if __name__ == '__main__':
    # 示例用法
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config import PATHS

    dataset = MicroLens50kDataset(
        data_root=PATHS['data_root'],
        pairs_file=PATHS['pairs_file'],
        titles_file=PATHS['titles_file'],
        covers_dir=PATHS['covers_dir'],
        output_dir=PATHS['processed_data'],
        features_dir=PATHS['features_dir'],
        seed=42
    )

    stats = dataset.process_all(
        text_feat_dim=256,
        image_feat_dim=256,
        val_ratio=0.1,
        test_ratio=0.1
    )

    print("\nDataset Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
