# UR5e Motion Planning and SAC–IK Control

Project mô phỏng cánh tay UR5e trong MuJoCo bằng ba phương pháp:

- **RRT:** tìm đường đi khả thi và tránh vật cản.
- **RRT\*:** tiếp tục mở rộng, lựa chọn lại nút cha và tái kết nối cây để cải thiện đường đi.
- **SAC–IK:** SAC đưa robot vào vùng chuyển giao, sau đó IK đưa đầu công tác đến pose đích chính xác.

Các phương pháp dùng chung mô hình UR5e, động học thuận/nghịch, điểm bắt đầu và hai vật cản trong `assets/scene.xml`.

## 1. Cấu trúc project

```text
PROJECT_UR5_Final/
├── assets/                 Scene, XML và mesh UR5e
├── models/
│   └── SAC_trained_agent.zip
├── results/                Excel và video kết quả
├── common.py               Tham số chung, FK, IK, kiểm tra va chạm
├── rrt.py                  Thuật toán RRT
├── rrt_star.py             Thuật toán RRT*
├── sac_environment.py      Môi trường SAC
├── train_sac.py            Huấn luyện SAC qua 7 level
├── run_sac_ik.py           Thực thi Actor rồi chuyển sang IK
├── trajectory_viewer.py    Xem Excel trên MuJoCo và quay video
├── requirements.txt        Các thư viện Python
└── README.md
```

## 2. Cài đặt

Mở thư mục project bằng VS Code, sau đó chọn **Terminal → New Terminal**.

Tạo môi trường Python riêng:

```powershell
python -m venv .venv
```

Cài thư viện:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Chỉ cần cài một lần trên mỗi máy tính. Cú pháp đúng là `.\.venv\Scripts\python.exe`; không dùng `..venv` hoặc `..\.venv`.

## 3. Chạy RRT

```powershell
.\.venv\Scripts\python.exe .\rrt.py
```

Kết quả: `results\rrt_path.xlsx`.

Xem quỹ đạo:

```powershell
.\.venv\Scripts\python.exe .\trajectory_viewer.py .\results\rrt_path.xlsx
```

## 4. Chạy RRT\*

```powershell
.\.venv\Scripts\python.exe .\rrt_star.py
```

RRT\* cần nhiều thời gian hơn RRT vì vẫn tiếp tục tối ưu sau khi tìm thấy đường đầu tiên.

Kết quả: `results\rrt_star_path.xlsx`.

```powershell
.\.venv\Scripts\python.exe .\trajectory_viewer.py .\results\rrt_star_path.xlsx
```

## 5. Huấn luyện SAC

Project đã có `models\SAC_trained_agent.zip`. Có thể bỏ qua bước này nếu chỉ muốn chạy model đã huấn luyện.

Để huấn luyện lại hoàn toàn từ đầu, chạy duy nhất:

```powershell
.\.venv\Scripts\python.exe .\train_sac.py
```

Curriculum gồm bảy level:

| Level | Miền thay đổi mỗi khớp | Action |
|---:|---:|---:|
| 1 | ±0.10 rad | 100,000 |
| 2 | ±0.20 rad | 50,000 |
| 3 | ±0.35 rad | 50,000 |
| 4 | ±0.55 rad | 50,000 |
| 5 | ±0.80 rad | 150,000 |
| 6 | ±1.10 rad | 200,000 |
| 7 | ±π/2 rad | 400,000 |

Tổng cộng: **1,000,000 action**.

Sau huấn luyện:

- Model mới ghi đè `models\SAC_trained_agent.zip`.
- Agent được đánh giá deterministic trên 20 goal.
- Kết quả đánh giá lưu tại `results\evaluation.csv`.

## 6. Chạy SAC kết hợp IK

Mở `run_sac_ik.py` và sửa hai biến đầu file:

```python
OUTPUT_FILE = ROOT / "results" / "sac1_ik_path.xlsx"
TARGET_Q = Q_GOAL_REFERENCE.copy()
```

Chạy:

```powershell
.\.venv\Scripts\python.exe .\run_sac_ik.py
```

SAC phát action ở **20 Hz**. Mỗi action được nội suy thành năm lệnh actuator ở **100 Hz**. MuJoCo chạy vật lý ở **500 Hz**.

SAC chuyển một chiều sang IK khi sai số vị trí không quá `0.10 m` và robot không va chạm. Sai số hướng vẫn có trong observation và reward; IK hiệu chỉnh pose cuối đến:

- sai số vị trí `< 1e-5 m`;
- sai số hướng `< 2e-4 rad`.

### Các goal đã kiểm tra

**SAC1**

```python
OUTPUT_FILE = ROOT / "results" / "sac1_ik_path.xlsx"
TARGET_Q = np.array([-2.70, -0.70, 0.70, -1.50, -1.40, -0.50])
```

**SAC2**

```python
OUTPUT_FILE = ROOT / "results" / "sac2_ik_path.xlsx"
TARGET_Q = np.array([-2.88, -0.58, 0.52, -1.55, -1.32, -0.60])
```

**SAC3**

```python
OUTPUT_FILE = ROOT / "results" / "sac3_ik_path.xlsx"
TARGET_Q = np.array([-2.95, -0.52, 0.58, -1.58, -1.35, -0.65])
```

**SAC4 – goal ở cao giữa hai vật cản**

```python
OUTPUT_FILE = ROOT / "results" / "sac4_ik_path.xlsx"
TARGET_Q = np.array([
    -2.296010322,
    -1.331316670,
     1.147140580,
    -2.896024462,
    -0.769984635,
     0.575502169,
])
```

Mỗi lần chạy phải đổi `OUTPUT_FILE` để không ghi đè kết quả trước.

Nếu báo `SAC did not reach the handover region`, Actor không đưa robot vào vùng 10 cm trong số bước cho phép. Hãy kiểm tra lại `TARGET_Q`; một cấu hình không va chạm chưa chắc nằm trong miền mà model hiện tại điều khiển thành công.

## 7. Xem quỹ đạo và quay video

Ví dụ với SAC2:

```powershell
.\.venv\Scripts\python.exe .\trajectory_viewer.py .\results\sac2_ik_path.xlsx
```

Các nút trên giao diện:

- **Browse:** chọn file Excel.
- **Load:** nạp file trong ô đường dẫn.
- **Reset:** trở về đầu quỹ đạo.
- **Pause:** tạm dừng.
- **Play:** chạy quỹ đạo.
- **Record:** tạo video MP4 từ toàn bộ quỹ đạo.

Video giữ hình 2 giây trước khi robot bắt đầu và 2 giây sau khi kết thúc.

## 8. File đầu ra

| Chương trình | Kết quả |
|---|---|
| `rrt.py` | `results\rrt_path.xlsx` |
| `rrt_star.py` | `results\rrt_star_path.xlsx` |
| `train_sac.py` | `models\SAC_trained_agent.zip`, `results\evaluation.csv` |
| `run_sac_ik.py` | File Excel đặt tại `OUTPUT_FILE` |
| `trajectory_viewer.py` | Video MP4 do người dùng chọn |

Excel quỹ đạo chứa thời gian, sáu biến khớp, pha điều khiển, sai số vị trí, sai số hướng và trạng thái va chạm. Chu kỳ giữa hai hàng là `0.01 s`.

## 9. Lỗi thường gặp

### Không tìm thấy `.venv`

Nếu PowerShell không nhận `.\.venv\Scripts\python.exe`, hãy tạo lại `.venv` và cài thư viện theo mục 2.

### Không tìm thấy Excel

Hãy chạy chương trình sinh quỹ đạo trước, rồi truyền đúng file:

```powershell
.\.venv\Scripts\python.exe .\trajectory_viewer.py .\results\rrt_path.xlsx
```

### Goal va chạm

Thông báo `The goal configuration is in collision` nghĩa là `TARGET_Q` làm robot chạm vật cản hoặc môi trường. Cần chọn cấu hình goal hợp lệ.
