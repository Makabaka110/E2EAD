#filter some 0 out

import pandas as pd
import sys
sys.path.append("..")


# 加载Excel文件到一个DataFrame中
file_path = 'your_excel_file.xlsx'
df = pd.read_excel(file_path, header=None)
print(df.shape)
print(df.columns)

# 初始化变量以跟踪连续的零值和连续零值段的起始索引
consecutive_zeros = 0
start_index = None

# 用于存储要保留的行的索引
rows_to_keep = []

# 遍历DataFrame的行
for index, row in df.iterrows():
    if row[3] == 0:  # 假设第四列是0-based
        if consecutive_zeros == 0:
            start_index = index
        consecutive_zeros += 1
    else:
        if consecutive_zeros > 10:
            rows_to_keep.extend([start_index, start_index + 1, index - 2 ,index - 1, index])
        elif consecutive_zeros > 0 and consecutive_zeros <= 10:
            rows_to_keep.extend(range(start_index, index + 1))
        else:
            rows_to_keep.extend([index])  # 包含非零行
        consecutive_zeros = 0


# 检查是否有尾随的连续零值段需要包含
if consecutive_zeros > 10:
    rows_to_keep.extend([start_index, start_index + 1])

if consecutive_zeros > 0 and consecutive_zeros <= 10:
    rows_to_keep.extend(range(start_index, index + 1))

# 创建一个新的DataFrame，其中包含所选的行
filtered_df = df.loc[rows_to_keep]

# 将修改后的DataFrame保存到新的Excel文件中
output_file_path = 'modified_excel_file.xlsx'
filtered_df.to_excel(output_file_path, index=False, header=False, engine='openpyxl')

print(f"已保存修改后的Excel文件至 {output_file_path}")