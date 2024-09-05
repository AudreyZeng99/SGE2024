import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 假设文件名为 data.json
# filename = '../exp/wn18rr/explanations/shifted_triplet_set_wn18rr_with_shifts.json'
filename = '../exp/fb15k-237/explanations/shifted_triplet_set_fb15k-237_with_shifts.json'
# 读取JSON文件
with open(filename, 'r') as file:
    data = json.load(file)

# 初始化向量维度总值的变量
h_shift_total = np.zeros(50)
r_shift_total = np.zeros(50)
t_shift_total = np.zeros(50)

# 记录h_shift_vector的数量
h_shift_vector_count = 0

# 存储每个triplet的hsft_emb, rsft_emb, tsft_emb的总和值
hsft_emb_sums = []
rsft_emb_sums = []
tsft_emb_sums = []

# 遍历数据计算总值
for key, value in data.items():
    if 'shifted' in value:
        h_shift_total += np.array(value['h_shift_vector'])
        r_shift_total += np.array(value['r_shift_vector'])
        t_shift_total += np.array(value['t_shift_vector'])
        h_shift_vector_count += 1

    # 计算每个triplet的总和值
    hsft_emb_sum = sum(value['shifted']['hsft_emb'])
    rsft_emb_sum = sum(value['shifted']['rsft_emb'])
    tsft_emb_sum = sum(value['shifted']['tsft_emb'])

    hsft_emb_sums.append(hsft_emb_sum)
    rsft_emb_sums.append(rsft_emb_sum)
    tsft_emb_sums.append(tsft_emb_sum)

# 对总值进行归一化
scaler = MinMaxScaler()

h_shift_total_normalized = scaler.fit_transform(h_shift_total.reshape(-1, 1)).flatten()
r_shift_total_normalized = scaler.fit_transform(r_shift_total.reshape(-1, 1)).flatten()
t_shift_total_normalized = scaler.fit_transform(t_shift_total.reshape(-1, 1)).flatten()

# 分别对 hsft_emb_sums, rsft_emb_sums, tsft_emb_sums 进行归一化
hsft_emb_sums_normalized = scaler.fit_transform(np.array(hsft_emb_sums).reshape(-1, 1)).flatten()
rsft_emb_sums_normalized = scaler.fit_transform(np.array(rsft_emb_sums).reshape(-1, 1)).flatten()
tsft_emb_sums_normalized = scaler.fit_transform(np.array(tsft_emb_sums).reshape(-1, 1)).flatten()

# 将结果保留小数点后3位
h_shift_total_normalized = np.round(h_shift_total_normalized, 3)
r_shift_total_normalized = np.round(r_shift_total_normalized, 3)
t_shift_total_normalized = np.round(t_shift_total_normalized, 3)
hsft_emb_sums_normalized = np.round(hsft_emb_sums_normalized, 3)
rsft_emb_sums_normalized = np.round(rsft_emb_sums_normalized, 3)
tsft_emb_sums_normalized = np.round(tsft_emb_sums_normalized, 3)

# 找到最大值和索引
h_list = h_shift_total_normalized.tolist()
h_max = max(h_list)
h_max_index = h_list.index(h_max) + 1

r_list = r_shift_total_normalized.tolist()
r_max = max(r_list)
r_max_index = r_list.index(r_max) + 1

t_list = t_shift_total_normalized.tolist()
t_max = max(t_list)
t_max_index = t_list.index(t_max) + 1

print(h_max, h_max_index)
print(r_max, r_max_index)
print(t_max, t_max_index)

# 打印输出结果
print("h_shift_vector归一化结果: ", h_shift_total_normalized.tolist())
print("r_shift_vector归一化结果: ", r_shift_total_normalized.tolist())
print("t_shift_vector归一化结果: ", t_shift_total_normalized.tolist())

# 打印前三个和最后一个triplet的归一化结果
print("前三个triplet的hsft_emb归一化结果: ", hsft_emb_sums_normalized.tolist()[:3])
print("前三个triplet的rsft_emb归一化结果: ", rsft_emb_sums_normalized.tolist()[:3])
print("前三个triplet的tsft_emb归一化结果: ", tsft_emb_sums_normalized.tolist()[:3])
print("最后一个triplet的hsft_emb归一化结果: ", hsft_emb_sums_normalized.tolist()[-1])
print("最后一个triplet的rsft_emb归一化结果: ", rsft_emb_sums_normalized.tolist()[-1])
print("最后一个triplet的tsft_emb归一化结果: ", tsft_emb_sums_normalized.tolist()[-1])
print("len", len(hsft_emb_sums))
