import os, sys
import numpy as np
import torch
import pandas as pd

# ========== 路径 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PARENT = os.path.join(BASE_DIR, 'DHPA')
sys.path.insert(0, SRC_PARENT)

# 确保 src/__init__.py 存在
src_init = os.path.join(SRC_PARENT, 'src', '__init__.py')
if not os.path.exists(src_init):
    open(src_init, 'w').close()

from src.models.DeepPA import DeepPA

# ========== 超参数 ==========
params = dict(
    num_nodes=1687, seq_len=12, horizon=12, output_dim=1, input_dim=12,
    n_hidden=64, end_channels=512, n_blocks=2, n_heads=2, mlp_expansion=2,
    dropout=0.3, covar_dim=10, GCO_Thre=0.5,
    spatial_flag=True, temporal_flag=True,
    spatial_encoding=True, temporal_encoding=True, temporal_PE=True,
    GCO=True, CLUSTER=True, temporal_causal=True,
    spatial_op='gco', spatial_heads=2, afno_keep_ratio=0.5,
    ts_decompose=True, decompose_kernel=5, season_periods=(12, 24, 48),
    device=torch.device('cpu')
)

# ========== 加载测试数据 ==========
data_path = os.path.join(BASE_DIR, 'data', 'val.npz')
data = np.load(data_path, mmap_mode='r')
x_all = data['x']          # (1217, 12, 1687, 12)
y_true_all = data['y']     # (1217, 12, 1687, 1)

sample_idx = 0
x_np = x_all[sample_idx:sample_idx+1].astype(np.float32)
y_true_np = y_true_all[sample_idx:sample_idx+1].squeeze(-1).astype(np.float32)

# ========== 公共函数 ==========
def load_model(weight_name):
    """加载 end 或 start 模型"""
    model = DeepPA(
        dropout=params['dropout'], spatial_flag=params['spatial_flag'],
        temporal_flag=params['temporal_flag'], spatial_encoding=params['spatial_encoding'],
        temporal_encoding=params['temporal_encoding'], temporal_PE=params['temporal_PE'],
        GCO=params['GCO'], CLUSTER=params['CLUSTER'], n_hidden=params['n_hidden'],
        end_channels=params['end_channels'], n_blocks=params['n_blocks'],
        n_heads=params['n_heads'], mlp_expansion=params['mlp_expansion'],
        covar_dim=params['covar_dim'], GCO_Thre=params['GCO_Thre'],
        temporal_causal=params['temporal_causal'], spatial_op=params['spatial_op'],
        spatial_heads=params['spatial_heads'], afno_keep_ratio=params['afno_keep_ratio'],
        ts_decompose=params['ts_decompose'], decompose_kernel=params['decompose_kernel'],
        season_periods=params['season_periods'],
        num_nodes=params['num_nodes'], seq_len=params['seq_len'],
        horizon=params['horizon'], output_dim=params['output_dim'],
        input_dim=params['input_dim'], device=params['device'],
        name='inference', dataset='val'
    )
    pt_path = os.path.join(BASE_DIR, 'data', f'final_model_{weight_name}.pt')
    state_dict = torch.load(pt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f'✅ 模型 {weight_name} 加载成功')
    return model

# 加载两个模型
model_start = load_model('start')
model_end = load_model('end')

# ========== 推理 ==========
x_tensor = torch.tensor(x_np, dtype=torch.float32)
with torch.no_grad():
    pred_start = model_start(x_tensor).squeeze(-1).numpy()   # (1, 12, 1687)
    pred_end   = model_end(x_tensor).squeeze(-1).numpy()

# 提取未来6步
H = 6
history_out = x_np[0, :, :, 0].T          # (1687, 12) 历史可用车位（取第0特征）
true_out    = y_true_np[0, :H, :].T       # (1687, 6)  真实未���
pred_start_out = pred_start[0, :H, :].T   # (1687, 6)  优化前预测
pred_end_out   = pred_end[0, :H, :].T     # (1687, 6)  优化后预测

# ========== 坐标加载 ==========
import csv   # 用标准库，无需 pandas

def load_locations():
    """从 CSV 加载经纬度，若无则降级陆地坐标"""
    csv_path = os.path.join(BASE_DIR, 'DHPA', 'aux_data', 'lots_location.csv')
    num = params['num_nodes']

    if os.path.exists(csv_path):
        try:
            lats, lngs = [], []
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                # 支持 'Latitude,Longitude' 或仅有数值两列
                for row in reader:
                    if len(row) >= 2:
                        lats.append(float(row[0]))
                        lngs.append(float(row[1]))
            if len(lats) > 0:
                N = min(len(lats), num)
                ids = [f'lot_{i}' for i in range(N)]
                names = [f'停车场 {i+1}' for i in range(N)]
                print(f'✅ 从 CSV 加载 {N} 个坐标')
                return ids, names, np.array(lats[:N]), np.array(lngs[:N])
        except Exception as e:
            print(f'⚠️ CSV 读取失败：{e}')

    # 回退 NPZ 或陆地随机（保证不落海）
    if 'lat' in data and 'lng' in data:
        print('✅ 使用 NPZ 中的坐标')
        return data['ids'], data['names'], data['lat'], data['lng']
    # 最终降级：严格新加坡陆地区域
    ids = [f'lot_{i}' for i in range(num)]
    names = [f'停车场 {i+1}' for i in range(num)]
    lats = np.random.uniform(1.28, 1.40, num)
    lngs = np.random.uniform(103.6, 103.9, num)
    print('⚠️ 无真实坐标，已生成新加坡陆地随机坐标（避免海上）')
    return ids, names, lats, lngs

ids, names, lats, lngs = load_locations()

pred_start_out = np.clip(pred_start_out, 0, None)
pred_end_out = np.clip(pred_end_out, 0, None)

# ========== 保存结果 ==========
save_dict = {
    'history': history_out,
    'true': true_out,
    'pred_start': pred_start_out,
    'pred_end': pred_end_out,
    'ids': np.array(ids),
    'names': np.array(names),
    'lat': np.array(lats),
    'lng': np.array(lngs),
}
np.savez_compressed(os.path.join(BASE_DIR, 'data', 'predictions.npz'), **save_dict)
print('✅ 推理结果已保存至 data/predictions.npz')