import numpy as np

# 加载 .npz 文件
data = np.load('path/to/your/file.npz', allow_pickle=True)

# 查看文件中包含的所有数组名称
print(data.keys())  # 例如，可能输出 ['arr_0', 'arr_1', ...] 或 ['train_data', 'labels', ...]

# 通过键名访问具体的数组
array1 = data['arr_0']
array2 = data['arr_1']

# 使用数据...
print(array1.shape)

# （可选）使用完后关闭文件，以释放资源
data.close()