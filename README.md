# DHPA 智慧停车协同服务平台

基于 DHPA（分解增强预测模型）的智慧停车协同服务平台，提供停车场实时状态可视化与未来可用车位预测。

## 项目结构
DHPA/
├── app.py # Flask 后端服务
├── inference.py # 模型推理脚本，生成 predictions.npz
├── perf_test.py # API 性能测试脚本
├── test_integration.py # 数据集成校验脚本
├── check_data.py # 查看 npz 文件结构的工具
├── load_data.py # npz 加载示例
├── index.html # 前端可视化页面（单页应用）
├── data/ # 数据目录（需提前准备）
│ ├── val.npz # 验证集原始数据（由 DHPA 训练产出）
│ ├── final_model_start.pt
│ ├── final_model_end.pt
│ └── predictions.npz # 推理后生成，供 app.py 加载
├── DHPA/ # 模型源码（需从原项目复制）
│ ├── src/
│ └── aux_data/ # 可选，含 lots_location.csv
└── requirements.txt # Python 依赖



## 环境配置

### 1. Python 环境

推荐使用 Python 3.8+，创建虚拟环境并安装依赖：

```
conda create -n dhpa python=3.8
conda activate dhpa
pip install -r requirements.txt
requirements.txt 内容示例：


flask
flask-cors
numpy
torch
pandas
requests
2. 数据准备
将训练好的模型权重 final_model_start.pt 和 final_model_end.pt 放入 data/ 目录。

将验证集数据 val.npz（含 x, y 等键）放入 data/。

（可选）若需真实停车场经纬度，请准备 DHPA/aux_data/lots_location.csv，格式为两列：Latitude,Longitude（无表头也可）。

3. 运行推理
执行以下命令生成 predictions.npz（包含 1687 个停车场的 12 步历史、6 步真实值、两种模型预测值及坐标）：


python inference.py
若缺少坐标文件，程序会自动生成新加坡陆地区域随机坐标。

启动服务
1. 启动后端
python app.py
Flask 服务将在 http://127.0.0.1:5000 运行，开启 Debug 模式。

2. 打开前端
直接用浏览器打开 index.html，或使用 Live Server 等工具。

页面会自动加载地图并请求 /api/parking/list 获取所有停车场数据。

API 接口说明
接口	方法	说明
/api/parking/list	GET	返回所有停车场的基础信息（id, name, lat, lng, total, available, occupancy）
/api/parking/<parking_id>/history	GET	返回指定停车场的历史（12步）、真实值（6步）、优化前预测（6步）、优化后预测（6步）
返回数据均为 JSON 格式，字段含义详见前端代码或实际调用。

性能测试
执行 perf_test.py 可测试 API 响应时间：


python perf_test.py
注意事项
前端使用 Leaflet 和 ECharts，所有资源通过 CDN 加载，需联网。

若数据量较大（如 1687 个点），建议开启浏览器硬件加速。

地图标记已集成聚类插件，加载流畅。