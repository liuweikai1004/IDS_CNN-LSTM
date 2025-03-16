import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import psutil
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split


# 1. 数据加载与预处理
def load_data_csv(path):
    # 添加CSV格式特有参数
    data = pd.read_csv(path,
                       sep=',',  # 分隔符，根据实际数据调整
                       encoding='utf-8')  # 编码方式，中文数据可尝试 'gbk'

    # 分类特征设置（请根据实际数据字段确认）
    categorical_cols = ['proto', 'service', 'state']  # UNSW-NB15典型分类特征
    label_encoders = {}

    # 分类特征编码
    for col in categorical_cols:
        if col in data.columns:
            # 填充缺失值为 'unknown'，并转换为字符串
            data[col] = data[col].fillna('unknown').astype(str)
            # 初始化新的编码器
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le  # 保存编码器
        else:
            print(f"Warning: 分类特征列 {col} 不存在于数据集中，已跳过")

    # 数值型缺失值处理（排除分类特征）
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in categorical_cols]
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    # 定义攻击类型到数值的映射
    attack_cat_mapping = {
        'Normal': 0,
        'Fuzzers': 1,
        'Analysis': 2,
        'Backdoors': 3,
        'Dos': 4,
        'Exploits': 5,
        'Generic': 6,
        'Reconnaissance': 7,
        'Shellcode': 8,
        'Worms': 9
    }

    # 如果 attack_cat 列存在
    if 'attack_cat' in data.columns:
        # 填充缺失值为 'unknown'（如果有）
        data['attack_cat'] = data['attack_cat'].fillna('unknown')
        # 将 attack_cat 列的值映射为数值
        data['attack_cat'] = data['attack_cat'].map(attack_cat_mapping).fillna(-1).astype(int)  # 未知类别标记为 -1
    else:
        print("Warning: 'attack_cat' 列不存在于数据集中")

    # 添加数据打印
    print("\n=== 数据加载报告 ===")
    print(f"数据集形状: {data.shape}")
    print("前3行样本:")
    print(data.head(3))
    print("\n特征类型分布:")
    print(data.dtypes.value_counts())
    print("\n标签分布:")
    print(data['label'].value_counts(normalize=True))
    print("缺失值统计:")
    print(data.isnull().sum().sort_values(ascending=False).head(5))
    print("=" * 40)
    return data

# 特征选择
def optimized_feature_engineering(data):
    """优化后的特征工程函数"""
    exclude_cols = ['label', 'id']  # 非特征列
    X = data.drop(columns=[col for col in exclude_cols if col in data.columns])
    y = data['label']

    # 训练随机森林模型
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # 获取特征重要性
    importances = model.feature_importances_
    important_features = np.argsort(importances)[-20:]  # 选择前 20 个重要特征

    # 获取特征名称
    selected_feature_names = X.columns[important_features].tolist()

    return selected_feature_names

# 3. 数据预处理主函数
def preprocess_data(train_path, split_rate):
    # 加载数据
    print("加载数据...")
    data = load_data_csv(train_path)

    # 特征选择
    print("进行特征选择...")
    selected_features = optimized_feature_engineering(data)
    print(f"Selected features: {selected_features}")

    # 筛选特征并处理
    X = data[selected_features]
    y = data['label']

    # 划分训练集和验证集（先划分再预处理，避免数据泄漏）
    print("划分训练集和验证集...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y,
        test_size=split_rate,
        random_state=42,
        stratify=y  # 保持类别分布一致（适用于分类任务）
    )

    # 标准化和归一化处理（仅在训练集上拟合）
    print("标准化和归一化处理...")

    # 第一步：归一化处理
    minmax_scaler = MinMaxScaler()
    X_train_normalized = minmax_scaler.fit_transform(X_train_raw)
    X_val_normalized = minmax_scaler.transform(X_val_raw)

    # 第二步：标准化处理
    standard_scaler = StandardScaler()
    X_train_processed = standard_scaler.fit_transform(X_train_normalized)
    X_val_processed = standard_scaler.transform(X_val_normalized)

    # 将数据重塑为适合 CNN 的 4D 格式 (samples, height, width, channels)
    def reshape_to_cnn_format(data):
        """将数据重塑为适合 CNN 的 4D 格式"""
        n_samples = data.shape[0]
        n_features = data.shape[1]

        # 计算 height 和 width
        height = int(np.sqrt(n_features))  # 假设将特征重塑为正方形
        width = height
        if height * width < n_features:
            width += 1  # 调整宽度以容纳所有特征

        # 填充零值
        padded_data = np.zeros((n_samples, height * width))
        padded_data[:, :n_features] = data

        # 重塑为 4D 格式 (samples, height, width, channels)
        return padded_data.reshape(n_samples, height, width, 1)

    X_train_cnn = reshape_to_cnn_format(X_train_processed)
    X_val_cnn = reshape_to_cnn_format(X_val_processed)

    return X_train_cnn, y_train, X_val_cnn, y_val, selected_features