import numpy as np
import os
import math
from scipy.signal import butter, lfilter
from tqdm import tqdm
import warnings

# =============================================================================
# 1. 滤波器函数 (保持不变)
# =============================================================================

def butter_bandpass(lowcut, highcut, fs, order=5):
    """巴特沃斯带通滤波器系数"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """带通滤波函数"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# =============================================================================
# 2. DE 特征计算 (保持不变)
# =============================================================================

def compute_DE(signal):
    """计算DE特征 (对数方差)"""
    # 确保信号长度大于1以计算方差
    if len(signal) <= 1:
        return 0.0, -np.inf 
        
    variance = np.var(signal, ddof=1)
    
    # 避免对0或负数取对数
    if variance <= 0:
        return variance, -np.inf
        
    return variance, math.log(2 * math.pi * math.e * variance) / 2

# =============================================================================
# 3. [修改] 提取并保存 DE 特征 (适配 _1s.npz 数据)
# =============================================================================

def extract_de_from_1s_windows(data_root, save_root):
    """
    修改版：
    读取 S{participant}_Dataset_1s.npz 文件，
    并对每个 1s 窗口计算 DE 特征。
    """
    
    frequency = 128  # 128 采样点 / 1 秒 = 128 Hz
    
    # 确保保存目录存在
    os.makedirs(save_root, exist_ok=True)
    
    # 遍历每个被试（1到18号被试）
    for participant in range(1, 19): 
        print(f"--- 正在处理第 {participant} 个被试的数据 ---")

        # --- 1. [修改] 加载新的 _1s.npz 数据 ---
        file_name = f'S{participant}_Dataset_1s.npz'
        file_path = os.path.join(data_root, file_name)
        
        if not os.path.exists(file_path):
            warnings.warn(f"找不到文件: {file_path}, 跳过被试 {participant}")
            continue
            
        try:
            data = np.load(file_path, allow_pickle=True)
            # 转换为 numpy 以便 scipy 处理
            EEG_data = data['eeg_slices'].astype(np.float64) 
            # (N_samples, 66, 128)
            
            # [修改] 提取标签并转换为 (0, 1)
            labels = np.array([int(item[0]) for item in data['event_slices']]) - 1
            # (N_samples,)
            
        except Exception as e:
            warnings.warn(f"加载文件 {file_path} 出错: {e}, 跳过被试 {participant}")
            continue

        # --- 2. [修改] 初始化 DE 存储数组 (66 通道) ---
        num_samples = EEG_data.shape[0]
        num_channels = 66 # [修改]
        num_bands = 5
        
        decomposed_de = np.empty([num_samples, num_channels, num_bands])
        
        # 临时数组
        de = np.empty([1, num_channels, num_bands])
        variances_temp = np.empty([1, num_channels, num_bands]) # (可选)

        print(f"已加载 {num_samples} 个 1s 窗口, {num_channels} 个通道。开始计算 DE...")

        # --- 3. [修改] 遍历该被试的每个 1s 样本 ---
        for sample in tqdm(range(num_samples)):
            trial_signal = EEG_data[sample] # (66, 128)
            
            # [修改] 遍历 66 个通道
            for channel in range(num_channels): 
                signal_1s = trial_signal[channel] # (128,)
                
                # 各频段滤波
                delta_data = butter_bandpass_filter(signal_1s, 0.1, 4, frequency, order=5)
                theta_data = butter_bandpass_filter(signal_1s, 4, 8, frequency, order=5)
                alpha_data = butter_bandpass_filter(signal_1s, 8, 14, frequency, order=5)
                beta_data = butter_bandpass_filter(signal_1s, 14, 31, frequency, order=5)
                gamma_data = butter_bandpass_filter(signal_1s, 31, 50, frequency, order=5)

                # 计算DE特征
                variances_temp[0, channel, 0], de[0, channel, 0] = compute_DE(delta_data)
                variances_temp[0, channel, 1], de[0, channel, 1] = compute_DE(theta_data)
                variances_temp[0, channel, 2], de[0, channel, 2] = compute_DE(alpha_data)
                variances_temp[0, channel, 3], de[0, channel, 3] = compute_DE(beta_data)
                variances_temp[0, channel, 4], de[0, channel, 4] = compute_DE(gamma_data)
            
            # 将当前样本的DE特征添加到结果数组中
            decomposed_de[sample] = de

        # --- 4. [修改] 保存该被试的 DE 特征和标签 ---
        save_file_path = os.path.join(save_root, f'S{participant}_DE_Features_1s.npz')
        np.savez(save_file_path, DE=decomposed_de, labels=labels)
        print(f"已保存: {save_file_path}")

    print("="*40)
    print("所有被试的DE特征已保存完毕 (基于 1s 非重叠窗口)。")

# =============================================================================
# 4. 主函数
# =============================================================================
if __name__ == "__main__":
    
    # [修改] 设置为您的 _1s.npz 文件所在的目录
    # 我根据您旧代码的路径猜测了新路径
    data_root = '/share/workspace/shared_datasets/DTU'
    
    # [修改] 设置为您希望保存 DE 特征的新目录
    save_root = '/share/home/yuan/LY/SNN_DBformer/DE_DTU/DE_1s'
    
    # 检查输入路径是否存在
    if not os.path.exists(data_root):
        print(f"!!! 警告: 输入路径不存在: {data_root}")
        print("!!! 请确保 'data_root' 变量指向 'S1_Dataset_1s.npz' 等文件所在的目录。")
    else:
        # 调用函数
        extract_de_from_1s_windows(data_root, save_root)