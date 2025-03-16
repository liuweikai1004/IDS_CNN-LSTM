import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import psutil
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_classif
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

# 修改后的代码
def optimized_feature_engineering(data):
    """优化后的特征工程函数"""
    exclude_cols = ['label', 'id']  # 非特征列
    X = data.drop(columns=[col for col in exclude_cols if col in data.columns])
    y = data['label']

    # 第一阶段：快速特征预筛选
    print("正在进行快速特征预筛选...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    selected_idx = np.where(mi_scores > np.quantile(mi_scores, 0.5))[0]  # 保留前50%特征
    X_prefiltered = X.iloc[:, selected_idx]

    # 第二阶段：改进的离散化方法
    def efficient_discretize(df):
        """基于分布特征的智能分箱"""
        discretized = pd.DataFrame()
        for col in df.columns:
            if df[col].nunique() > 10:  # 仅对高基数特征离散化
                try:
                    q = min(5, len(pd.qcut(df[col], 5, duplicates='drop').cat.categories))
                    discretized[col] = pd.qcut(df[col], q=q, labels=False, duplicates='drop')
                except ValueError:  # 处理分箱失败的情况
                    discretized[col] = df[col]
            else:
                discretized[col] = df[col]
        return discretized

    X_discrete = efficient_discretize(X_prefiltered)

    # 第三阶段：优化关联规则挖掘
    print("正在执行优化版关联规则挖掘...")

    # 生成事务数据
    transactions = [
        [f"{col}_{val}" for col, val in row.items()] + [f"label_{y.iloc[_]}"]
        for _, row in tqdm(X_discrete.iterrows(), desc="Encoding transactions")
    ]

    # 使用 TransactionEncoder 编码事务数据
    te = TransactionEncoder()
    te.fit(transactions)  # 先调用 fit 生成列名
    te_ary = te.transform(transactions)  # 再调用 transform 编码数据

    # 动态调整支持度阈值
    n_samples = len(X_discrete)
    min_support = max(0.05, 100 / n_samples)  # 提高支持度阈值

    # 使用稀疏矩阵
    te_ary_sparse = csr_matrix(te_ary)

    # 并行化挖掘频繁项集
    def run_fpgrowth(subset, min_support):
        return fpgrowth(subset, min_support=min_support, use_colnames=True, max_len=2)

    print("正在挖掘频繁项集...")
    subsets = np.array_split(pd.DataFrame(te_ary_sparse.toarray(), columns=te.columns_), 4)
    results = Parallel(n_jobs=4)(delayed(run_fpgrowth)(subset, min_support) for subset in subsets)
    frequent_itemsets = pd.concat(results).drop_duplicates().reset_index(drop=True)

    # 特征选择策略
    selected_features = set()
    if not frequent_itemsets.empty:
        print("正在生成关联规则...")
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

        # 基于规则置信度和提升度筛选
        valid_rules = rules[(rules['confidence'] > 0.6) & (rules['lift'] > 1.2)]

        # 特征提取优化
        for itemset in valid_rules['antecedents']:
            for item in itemset:
                feature = item.split('_')[0]
                if feature in X.columns and feature != 'label':
                    selected_features.add(feature)

    return list(selected_features) if selected_features else X_prefiltered.columns.tolist()

# 3. 数据预处理主函数
def preprocess_data(test_path, selected_features, timesteps):
    # 加载数据
    print("加载数据...")
    data = load_data_csv(test_path)

    # 筛选特征并处理
    X = data[selected_features]
    y_test = data['label']

    # 标准化和归一化处理
    print("标准化和归一化处理...")

    # 第一步：归一化处理
    minmax_scaler = MinMaxScaler()
    X_test_normalized = minmax_scaler.fit_transform(X)

    # 第二步：标准化处理
    standard_scaler = StandardScaler()
    X_test_processed = standard_scaler.fit_transform(X_test_normalized)

    # 转换为3D格式 (samples, timesteps, features)
    X_test_3d = create_sliding_windows(X_test_processed, timesteps)

    y_test_cropped = y_test[timesteps - 1:]

    return X_test_3d,  y_test_cropped

def create_sliding_windows(data, timesteps):
    X = []
    for i in range(len(data) - timesteps + 1):
        X.append(data[i:i+timesteps])
    return np.array(X)
