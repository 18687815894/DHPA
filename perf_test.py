import requests
import time
import random

def test_list():
    times = []
    for _ in range(20):
        start = time.time()
        r = requests.get('http://localhost:5000/api/parking/list')
        times.append((time.time() - start) * 1000)
    print(f"/list 平均耗时: {sum(times)/len(times):.2f} ms")

def test_history():
    # 先获取所有 parking id
    resp = requests.get('http://localhost:5000/api/parking/list').json()
    ids = [lot['id'] for lot in resp]
    times = []
    for _ in range(50):
        lot_id = random.choice(ids)
        start = time.time()
        r = requests.get(f'http://localhost:5000/api/parking/{lot_id}/history')
        times.append((time.time() - start) * 1000)
    print(f"/history 平均耗时: {sum(times)/len(times):.2f} ms")

if __name__ == '__main__':
    test_list()
    test_history()