# CLAUDE.md

Bài 1 (ArtiGate-YOLO) trong lộ trình 3 bài Q1. Kế hoạch đầy đủ nằm trong sổ tay nghiên cứu trên claude.ai. Xem cấu trúc, quy trình và bảng ablation trong README.md.

## Quy tắc nghiên cứu (không ngoại lệ)

- Chia dữ liệu theo tổn thương (`lesion_id`) hoặc theo bệnh nhân. Mọi manifest phải qua `check_manifest` trước khi xuất.
- Mỗi cấu hình chạy ít nhất 3 seed, cùng lịch train `configs/train/default.yaml`. Không chỉnh tham số trên tập test.
- Báo cáo trung thực, kể cả chỗ mô hình không thắng. Tiêu chí thắng đã đặt trước khi chạy: thắng rõ trên ảnh nhiều artifact, không tệ hơn trên ảnh sạch.
- Quy tắc đánh giá đã khóa ngày 25/10/2026, trước khi chấm bất kỳ ảnh test ISIC 2018 nào:
  - quy tắc mask chính là top-1 (`scripts/eval.py --mask-rule top1`), union ≥ 0,25 chỉ là phân tích độ nhạy;
  - biên không thua kém trên ảnh sạch δ = 0,5 điểm Dice;
  - optimizer cố định AdamW lr 0.002 trong `configs/train/default.yaml`.
  Chạy thử chỉ đánh giá trên val. Tập test chỉ chấm bằng checkpoint của các run chính thức, mỗi checkpoint chấm một lần, và không chỉnh gì sau khi đã xem kết quả test.
- Bảo mật: không dùng nội dung nào từ các bản thảo hội nghị anh Hải đang review (không ghi mã bài ở đây). Không đọc thư mục `D:\PAPER_SPKT\paper_base_ham1000`.
- Không đưa dữ liệu, `runs/`, `weights/`, `DTA/` hay file `*.pt` lên git. Manifest trong `data/splits/` thì có lên git.

## Lệnh

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pytest   # Python 3.9, torch 2.6.0+cu124, Ultralytics 8.4.13
```

Máy cá nhân có RTX 4060 8 GB, chỉ dùng để chạy thử với batch nhỏ. Train thật chạy trên server 12–16 GB.

## Lưu ý kỹ thuật

- Code phải chạy được trên Python 3.9: không dùng `match`, không viết kiểu `X | Y` khi chưa có `from __future__ import annotations`.
- Gọi `register_modules()` trước khi dựng mô hình từ YAML hoặc nạp checkpoint. Hàm này thay `Segment26` của Ultralytics bằng `P2CompatibleSegment26`, vì `parse_model` so sánh class theo identity.
  Hàm này cũng vá `parse_model` (trong bộ nhớ) để `ArtifactHead` và `AGA` tự nhận số kênh đầu vào, kể cả khi có nhiều đầu vào.
- Chỉ nạp trọng số gốc qua `transfer_trainer(...)`, vì `Model.train()` dựng lại mô hình YAML và bỏ trọng số đã nạp trước đó (đây là lỗi của ICEEIS). Module mới chèn vào topology gốc (AGA, đầu phụ artifact) phải được thêm vào `INSERTED_MODULES` trong `transfer.py`.
- Tham số kênh của CBAM trong YAML là độ rộng ở scale n. Scale khác cần sửa lại.
- ArtiGate: `ArtifactHead` chèn ngay sau B2, `AGA` thay CBAM ở B2/B3 và dùng chung class `ChannelAttention`; lúc khởi tạo AGA giống hệt CBAM. `beta` không bị kẹp; cổng là max(0, 1 − βĀ).
- Mọi dòng ablation train qua `ArtiGateTrainer`. Ảnh train của các dòng ArtiGate có 7 kênh (BGR + 4 mask artifact, mã 255/0/114 = có/không/chưa biết) để augmentation của Ultralytics giữ mask khớp ảnh; `preprocess_batch` tách lại. Không cài `albumentations`.
- Checkpoint cùng kiến trúc (resume) phải nạp đủ mọi tensor; chỉ trọng số gốc YOLO26n-seg mới đi qua bản đồ chuyển một phần.
- Ultralytics thay `cv2.imread` toàn cục, ảnh xám trả về (H, W, 1). Đọc mask qua `read_mask`.
- Dữ liệu YOLO là ảnh letterbox 640×640. Mask gốc nằm ở `masks/<split>/<stem>.png`, phép biến đổi nằm trong `metadata/<split>.jsonl`.

## Làm việc giữa web và VS Code

Repo `Hainguyen752004/artigate-yolo` là nơi chứa code duy nhất. Kéo code về (pull) trước khi làm, commit và push khi xong. Claude web không push được: khi viết code, Claude web gửi một file patch (`git am`) để Claude VS Code áp dụng, chạy test trên GPU rồi push. Ghi chú giữa hai bên đặt trong sổ tay nghiên cứu.
