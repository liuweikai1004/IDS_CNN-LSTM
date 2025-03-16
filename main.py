import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.losses import binary_crossentropy
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import TrainSet_Preprocessing_CNN_LSTM
import TestSet_Preprocessing_CNN_LSTM
import TestSet_Preprocessing_CNN
import TrainSet_Preprocessing_CNN
import TestSet_Preprocessing_LSTM
import TrainSet_Preprocessing_LSTM
# 设置字体为SimHei，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



def model_train(x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size, epochs, model_type='cnn_lstm'):
    """
    模型训练函数
    :param x_train: 训练集数据
    :param y_train: 训练集标签
    :param x_valid: 验证集数据
    :param y_valid: 验证集标签
    :param x_test: 测试集数据
    :param y_test: 测试集标签
    :param batch_size: 批大小
    :param epochs: 训练轮数
    :param model_type: 模型类型 ('cnn_lstm', 'cnn', 'lstm')
    :return: 训练历史、模型、混淆矩阵
    """
    if model_type == 'cnn':
        input_shape = ((x_train.shape[1], x_train.shape[2], 1))
    else:
        input_shape = (x_train.shape[1], x_train.shape[2])  # (window_size, n_features)

    # 定义模型
    if model_type == 'cnn_lstm':
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)),
            MaxPooling1D(pool_size=2),
            LSTM(units=100, return_sequences=False),
            Dropout(0.5),
            Dense(100, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == 'cnn':
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == 'lstm':
        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            Dropout(0.2),
            Dense(units=16, activation='relu'),
            Dropout(0.2),
            Dense(units=1, activation='sigmoid')
        ])
    else:
        raise ValueError("Invalid model_type. Choose from 'cnn_lstm', 'cnn', or 'lstm'.")

    # 计算类别权重
    def calculate_class_weights(y_train):
        class_counts = np.bincount(y_train.astype(int))
        total_samples = np.sum(class_counts)
        num_classes = len(class_counts)
        class_weights = total_samples / (num_classes * class_counts)
        return class_weights

    # 计算训练数据的类别权重
    class_weights = calculate_class_weights(y_train)
    weight_positive = class_weights[1]  # 正类权重
    weight_negative = class_weights[0]  # 负类权重

    # 定义加权损失函数
    def weighted_binary_crossentropy(y_true, y_pred):
        weights = tf.where(y_true == 1, weight_positive, weight_negative)
        weights = tf.cast(weights, tf.float32)  # 将 weights 转换为 float32
        return tf.reduce_mean(weights * tf.keras.losses.binary_crossentropy(y_true, y_pred))

    # 编译模型
    model.compile(optimizer=Adam(), loss=weighted_binary_crossentropy, metrics=['accuracy'])

    # 打印模型摘要
    model.summary()

    # 使用早停法
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # 使用模型对测试集进行预测
    y_pred = model.predict(x_test)

    # 将预测概率转换为类别标签（二分类问题）
    y_pred_labels = np.where(y_pred > 0.5, 1, 0)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred_labels)

    # 计算 F1 值、Precision 和 Recall
    f1 = f1_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels)
    recall = recall_score(y_test, y_pred_labels)

    # 输出结果
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 保存模型
    if model_type == 'cnn':
        model.save(f'{model_type}_binary_model_CNN.h5')
    elif model_type == 'lstm':
        model.save(f'{model_type}_binary_model_LSTM.h5')
    else:
        model.save(f'{model_type}_binary_model_CNN_LSTM.h5')
    return history, model, conf_matrix

def acc_loss_line(history, conf_matrix):
    print("绘制准确率和损失值曲线")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # 绘制accuracy曲线
    plt.plot(epochs, acc, 'r', linestyle='-.')
    plt.plot(epochs, val_acc, 'b', linestyle='dashdot')
    plt.title('训练集和验证集准确率曲线')
    plt.xlabel("训练轮次")
    plt.ylabel("准确率")
    plt.legend(["训练集准确率", "验证集准确率"])

    plt.figure()

    # 绘制loss曲线
    plt.plot(epochs, loss, 'r', linestyle='-.')
    plt.plot(epochs, val_loss, 'b', linestyle='dashdot')
    plt.title('训练集和验证集损失值曲线')
    plt.xlabel("训练轮次")
    plt.ylabel("损失值")
    plt.legend(["训练集损失值", "验证集损失值"])

    # 可视化混淆矩阵
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵')

    plt.show()

if __name__ == '__main__':
    trainSet_file_path = './Dataset/UNSW-NB15/UNSW_NB15_training-set.csv'
    testSet_file_path = './Dataset/UNSW-NB15/UNSW_NB15_testing-set.csv'
    split_rate = 0.3  # 训练集、验证集的划分比例
    window_size = 10
    batch_size = 64  # 训练批次
    epochs = 1000  # 训练轮次
    model_type = 'cnn '  # 选择模型类型 ('cnn_lstm', 'cnn', 'lstm')

    # 1. 数据预处理
    if model_type == 'cnn_lstm':
        X_train_seq_res, y_train_seq_res, X_val_seq, y_val_seq, selected_features = TrainSet_Preprocessing_CNN_LSTM.preprocess_data(trainSet_file_path, split_rate, window_size)
        X_test_seq, y_test_seq = TestSet_Preprocessing_CNN_LSTM.preprocess_data(testSet_file_path, selected_features, window_size)
    elif model_type == 'cnn':
        X_train_seq_res, y_train_seq_res, X_val_seq, y_val_seq, selected_features = TrainSet_Preprocessing_CNN.preprocess_data(trainSet_file_path, split_rate)
        X_test_seq, y_test_seq = TestSet_Preprocessing_CNN.preprocess_data(testSet_file_path, selected_features)
    elif model_type == 'lstm':
        X_train_seq_res, y_train_seq_res, X_val_seq, y_val_seq, selected_features = TrainSet_Preprocessing_LSTM.preprocess_data(trainSet_file_path, split_rate, window_size)
        X_test_seq, y_test_seq = TestSet_Preprocessing_LSTM.preprocess_data(testSet_file_path, window_size, selected_features)
    else:
        raise ValueError("Invalid model_type. Choose from 'cnn_lstm', 'cnn', or 'lstm'.")

    # 2. 模型训练
    history, model, conf_matrix = model_train(X_train_seq_res, y_train_seq_res, X_val_seq, y_val_seq, X_test_seq, y_test_seq, batch_size, epochs, model_type)

    # 3. 准确率与损失值曲线展示
    acc_loss_line(history, conf_matrix)