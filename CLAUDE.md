# CLAUDE.md

Bài 1 (ArtiGate-YOLO) trong lộ trình 3 bài Q1. Kế hoạch đầy đủ nằm trong sổ tay nghiên cứu trên claude.ai. Xem cấu trúc, quy trình và bảng ablation trong README.md.

## Quy tắc nghiên cứu (không ngoại lệ)

- Chia dữ liệu theo tổn thương (`lesion_id`) hoặc theo bệnh nhân. Mọi manifest phải qua `check_manifest` trước khi xuất.
- Mỗi cấu hình chạy ít nhất 3 seed, cùng lịch train `configs/train/default.yaml`. Không chỉnh tham số trên tập test.
- Báo cáo trung thực, kể cả chỗ mô hình không thắng. Tiêu chí thắng đã đặt trước khi chạy: thắng rõ trên ảnh nhiều artifact, không tệ hơn trên ảnh sạch.
- Bảo mật: không dùng nội dung nào từ 3 bản thảo RIVF đang review (1571347679, 1571348289, 1571355058). Không đọc thư mục `D:\PAPER_SPKT\paper_base_ham1000`.
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
- Chỉ nạp trọng số gốc qua `transfer_trainer(...)`, vì `Model.train()` dựng lại mô hình YAML và bỏ trọng số đã nạp trước đó (đây là lỗi của ICEEIS). Module mới chèn vào topology gốc (AGA, đầu phụ artifact) phải được thêm vào `INSERTED_MODULES` trong `transfer.py`.
- Tham số kênh của CBAM trong YAML là độ rộng ở scale n. Scale khác cần sửa lại.
- Ultralytics thay `cv2.imread` toàn cục, ảnh xám trả về (H, W, 1). Đọc mask qua `read_mask`.
- Dữ liệu YOLO là ảnh letterbox 640×640. Mask gốc nằm ở `masks/<split>/<stem>.png`, phép biến đổi nằm trong `metadata/<split>.jsonl`.

## Làm việc giữa web và VS Code

Repo riêng tư `Hainguyen752004/artigate-yolo` là nơi chứa code duy nhất. Kéo code về (pull) trước khi làm, commit và push khi xong. Claude web làm việc với file anh Hải tải lên, nên khi cần thì đưa cho Claude web bản zip hoặc từng file từ repo.
