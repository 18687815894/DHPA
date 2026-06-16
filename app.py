import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# ==================== 配置 ====================
DATA_DIR = 'data'
PRED_NPZ = os.path.join(DATA_DIR, 'predictions.npz')

# ==================== 全局变量 ====================
parking_ids = []
parking_names = []
lats = []
lngs = []
history_data = None      # (N, 12)
true_data = None         # (N, 6)
pred_start_data = None   # (N, 6)
pred_end_data = None     # (N, 6)
total_slots_estimates = None  # 预先算好的总车位数组

def load_data():
    global parking_ids, parking_names, lats, lngs
    global history_data, true_data, pred_start_data, pred_end_data
    global total_slots_estimates

    try:
        data = np.load(PRED_NPZ, allow_pickle=True)
        history_data = data['history']          # (N, 12)
        true_data = data['true']                # (N, 6)
        pred_start_data = data['pred_start']    # (N, 6)
        pred_end_data = data['pred_end']        # (N, 6)
        parking_ids = list(data['ids'])
        parking_names = list(data['names'])
        lats = data['lat']
        lngs = data['lng']

        N = len(parking_ids)
        print(f'✅ 从 {PRED_NPZ} 加载数据成功，{N} 个停车场')

        # ---------- 估算总车位 ----------
        total_slots_estimates = []
        for i in range(N):
            hist = history_data[i]                    # 历史12个时间点
            max_hist = np.max(hist)
            if max_hist == 0:
                total = 50
            else:
                # 假设最低占用率为15%（即可用车位最多占85%）
                total = int(max_hist / 0.85)
                total = max(total, max_hist + 5)      # 确保总车位至少比历史峰值多5
            total_slots_estimates.append(total)
        total_slots_estimates = np.array(total_slots_estimates)

    except Exception as e:
        print(f'❌ 加载失败: {e}')
        generate_mock_data()

def generate_mock_data():
    global parking_ids, parking_names, lats, lngs, history_data, true_data, pred_start_data, pred_end_data, total_slots_estimates
    NUM_LOTS = 100
    parking_ids = [f'lot_{i}' for i in range(NUM_LOTS)]
    parking_names = [f'模拟停车场 {i+1}' for i in range(NUM_LOTS)]
    lats = np.random.uniform(1.27, 1.47, NUM_LOTS)
    lngs = np.random.uniform(103.6, 104.0, NUM_LOTS)
    base_avail = np.random.randint(50, 300, NUM_LOTS)
    history_data = np.zeros((NUM_LOTS, 12), dtype=np.float32)
    true_data = np.zeros((NUM_LOTS, 6), dtype=np.float32)
    pred_start_data = np.zeros((NUM_LOTS, 6), dtype=np.float32)
    pred_end_data = np.zeros((NUM_LOTS, 6), dtype=np.float32)

    for i in range(NUM_LOTS):
        history_data[i] = base_avail[i] + np.random.randint(-10, 10, 12)
        true_data[i] = base_avail[i] + np.random.randint(-5, 5, 6)
        pred_start_data[i] = true_data[i] + np.random.randint(-3, 3, 6)
        pred_end_data[i] = true_data[i] + np.random.randint(-2, 2, 6)

    history_data = np.clip(history_data, 0, None)
    true_data = np.clip(true_data, 0, None)
    pred_start_data = np.clip(pred_start_data, 0, None)
    pred_end_data = np.clip(pred_end_data, 0, None)

    # 估算总车位
    total_slots_estimates = []
    for i in range(NUM_LOTS):
        max_hist = np.max(history_data[i])
        if max_hist == 0:
            total = 50
        else:
            total = int(max_hist / 0.85)
            total = max(total, max_hist + 5)
        total_slots_estimates.append(total)
    total_slots_estimates = np.array(total_slots_estimates)

    print('⚠️ 使用模拟数据（100个停车场）')

# 加载数据
load_data()

# ==================== API ====================
@app.route('/api/parking/list', methods=['GET'])
def get_parking_list():
    import time
    start = time.time()

    if history_data is None:
        return jsonify([])

    N = len(parking_ids)
    current_avail = history_data[:, -1].astype(int)   # 最后时刻的可用车位
    total = total_slots_estimates.astype(int)
    occupancy = 1 - (current_avail / total)
    # 避免除零或负占用率
    occupancy = np.clip(occupancy, 0.0, 1.0)

    result = []
    for i in range(N):
        result.append({
            'id': parking_ids[i],
            'name': parking_names[i],
            'lat': float(lats[i]),
            'lng': float(lngs[i]),
            'total': int(total[i]),
            'available': int(current_avail[i]),
            'occupancy': float(occupancy[i])
        })

    elapsed = (time.time() - start) * 1000
    print(f"[PERF] /api/parking/list 耗时: {elapsed:.2f} ms")
    return jsonify(result)

@app.route('/api/parking/<parking_id>/history', methods=['GET'])
def get_parking_history(parking_id):
    import time
    start = time.time()

    if history_data is None:
        return jsonify({'error': '数据未加载'}), 500
    try:
        idx = parking_ids.index(parking_id)
    except ValueError:
        return jsonify({'error': '未找到该停车场'}), 404

    # 钳位负值为0（确保预测结果非负）
    history = np.clip(history_data[idx], 0, None).tolist()
    true_val = np.clip(true_data[idx], 0, None).tolist()
    pred_start = np.clip(pred_start_data[idx], 0, None).tolist()
    pred_end = np.clip(pred_end_data[idx], 0, None).tolist()

    elapsed = (time.time() - start) * 1000
    print(f"[PERF] /api/parking/history({parking_id}) 耗时: {elapsed:.2f} ms")
    
    return jsonify({
        'history': history,
        'true': true_val,
        'pred_start': pred_start,
        'pred_end': pred_end
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)