# ArtiGate-YOLO

**ArtiGate-YOLO: Artifact-Supervised Gated Attention for Robust Real-Time Skin Lesion Segmentation**. Đây là Bài 1 trong lộ trình 3 bài Q1, nhắm IEEE Access (dự phòng BSPC).

Repo này là nơi chứa code duy nhất cho cả Claude trên web lẫn Claude trong VS Code. Nền code được viết lại từ bài hội nghị ICEEIS (YOLO26n-seg + P2 + CBAM). Mô hình ICEEIS đóng vai baseline A0/A1 của bài này.

## Trạng thái

- [x] Viết lại phần nền từ ICEEIS: mô hình A0/A1, tiền xử lý hai view (v1/v7), train, đánh giá, bootstrap
- [ ] Script tạo manifest cho ISIC 2018, HAM10000 (chia theo `lesion_id`), PH2. Chờ kết quả `tree -L 2 data/raw`
- [x] Code A2–A7: đầu phụ artifact, AGA, loss artifact (nhãn một phần), dán artifact tổng hợp, một trainer chung cho mọi dòng ablation
- [ ] Chạy thật A0–A7 trên server (3 seed mỗi dòng)

## Cấu trúc thư mục

```
configs/models/   YAML mô hình A0, A1, A2 (cbam-arthead), A4 (artigate), A7 (artigate-p2only)
configs/data/     YAML dữ liệu, `path` tính tương đối từ chính file YAML; `*_art.yaml` có khối `artigate:`
configs/train/    lịch train dùng chung cho mọi mô hình (lấy từ lần chạy cuối của ICEEIS)
src/artigate/     modules.py (CBAM, đầu P2, ArtifactHead, AGA), artifact_loss.py, transfer.py, trainer.py,
                  metrics.py, stats.py, provenance.py
src/artigate/data/  views.py (v1/v7), masks.py, split.py, export.py, dataset_yaml.py,
                  synth_artifacts.py (sinh artifact có mask), artifact_dataset.py (ảnh train 7 kênh)
scripts/          export_yolo.py, train.py, eval.py, bootstrap_ci.py,
                  eval_artifact_head.py, overfit_artifact_head.py, viz_train_batch.py, viz_artifact_maps.py,
                  make_severity_sets.py, label_artifacts.py, make_artifact_flags.py, check_leakage.py, benchmark.py
data/raw/         dữ liệu gốc (không lên git)
data/splits/      manifest chia dữ liệu, CÓ lên git để ai cũng chia giống nhau
data/yolo/        dữ liệu đã xuất sang YOLO-seg (không lên git)
runs/             kết quả train/eval (không lên git)
weights/          trọng số gốc yolo26n-seg.pt (không lên git)
paper/            LaTeX
tests/            pytest
```

## Cài đặt

```bash
pip install -r requirements.txt
```

Trọng số khởi tạo là `weights/yolo26n-seg.pt`, đúng file mà ICEEIS đã dùng, sha256 `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f`. Trên server phải dùng đúng file này. Mỗi lần train đều ghi lại sha256 vào `provenance.json` để đối chiếu.

Trên Windows, gọi thẳng Python của môi trường và bật UTF-8 cho log:

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pytest
```

## Quy trình

**1. Manifest.** Mỗi bộ dữ liệu có một file `data/splits/<bộ dữ liệu>.csv` với các cột `sample_id, split, image, mask, group`. Đường dẫn ảnh và mask tính từ `data/raw`. `group` là đơn vị chống rò rỉ (`lesion_id` với HAM10000); để trống thì lấy bằng `sample_id`. Hàm `artigate.data.split.grouped_split` chia nguyên từng tổn thương vào một tập.

**2. Xuất sang YOLO-seg.** Trước khi ghi, script kiểm tra trùng `sample_id`, tổn thương nằm ở nhiều tập, và ISIC ID trùng giữa các tập.

```bash
python scripts/export_yolo.py --manifest data/splits/isic2018.csv --out data/yolo/isic2018
# baseline hai view của ICEEIS: train có cả v1 và v7, val/test chỉ v1
python scripts/export_yolo.py --manifest data/splits/isic2018.csv --out data/yolo/isic2018_twoview --train-views v1,v7
```

Mỗi ảnh xuất ra kèm `labels/*.txt` (polygon), `masks/*.png` (mask gốc dùng để đánh giá) và `metadata/<split>.jsonl` (phép biến đổi, để sau này quy về độ phân giải gốc).

**3. Train, mỗi cấu hình 3 seed.**

```bash
python scripts/train.py --model configs/models/yolo26n-seg-p2-cbam.yaml --data configs/data/isic2018.yaml --seed 0
```

Mặc định tên run là `<model>_s<seed>`. Muốn chạy thử nhanh thì thêm `--epochs 3 --batch 4 --fraction 0.1`, nhưng không dùng các run đó cho bài báo.

Mọi dòng ablation đều qua `ArtiGateTrainer` (kế thừa `TransferSegmentationTrainer`), nên dữ liệu, augmentation và cách nạp trọng số giống hệt nhau. Các dòng ArtiGate dùng `configs/data/isic2018_art.yaml`:

```bash
python scripts/train.py --model configs/models/yolo26n-seg-p2-artigate.yaml --data configs/data/isic2018_art.yaml --seed 0              # A4
python scripts/train.py --model configs/models/yolo26n-seg-p2-artigate.yaml --data configs/data/isic2018_art.yaml --art-lambda 0 --seed 0  # A3
python scripts/train.py --model configs/models/yolo26n-seg-p2-artigate.yaml --data configs/data/isic2018_art.yaml --p-synth 0 --seed 0     # A5
python scripts/train.py --model configs/models/yolo26n-seg-p2-cbam.yaml --data configs/data/isic2018_art.yaml --name A1s_s0 --seed 0     # A1s
```

Trước khi chạy thật: `viz_train_batch.py` (mask artifact khớp ảnh sau augmentation) và `overfit_artifact_head.py --iters 300` (dữ liệu giả: IoU lông 0,83, góc tối 0,87 sau 250 bước ở 320 px).

**4. Đánh giá.**

```bash
python scripts/eval.py --weights runs/segment/yolo26n-seg-p2-cbam_s0/weights/best.pt --split test
```

Script ghi `eval_summary.json` (mAP của Ultralytics cùng Dice/IoU/HD95/boundary F1 trung bình) và `per_image_metrics.csv`. Quy tắc mặc định `--mask-rule top1` lấy instance có độ tin cậy cao nhất, giống ICEEIS.

Đầu artifact được đánh giá riêng trên artifact tổng hợp giữ lại (seed cố định): `python scripts/eval_artifact_head.py --weights <best.pt> --split val --n 100`. Bộ test theo mức độ artifact: `python scripts/make_severity_sets.py --split test --out data/yolo/isic2018_sev`, rồi chạy `eval.py` với `--data data/yolo/isic2018_sev/<loại>_s<mức>/data.yaml`.

**5. Khoảng tin cậy và so sánh ghép cặp.**

```bash
python scripts/bootstrap_ci.py runs/eval/A4_s0_test/per_image_metrics.csv --baseline runs/eval/A1_s0_test/per_image_metrics.csv
# theo nhóm có/không artifact (cờ cấp ảnh từ scripts/label_artifacts.py)
python scripts/bootstrap_ci.py ... --strata data/splits/isic2018_test_artifacts.csv
```

## Ablation

| Mã | Cấu hình | File | Trạng thái |
|---|---|---|---|
| A0 | Base + P2 | `configs/models/yolo26n-seg-p2.yaml` | sẵn sàng |
| A1 | + CBAM (mô hình ICEEIS) | `configs/models/yolo26n-seg-p2-cbam.yaml` | sẵn sàng |
| A1s | A1 train trên cùng dữ liệu có artifact tổng hợp | `yolo26n-seg-p2-cbam.yaml` + `isic2018_art.yaml` | sẵn sàng |
| A2 | + đầu phụ artifact, không điều khiển | `yolo26n-seg-p2-cbam-arthead.yaml` + `isic2018_art.yaml` | sẵn sàng |
| A3 | + AGA nhưng không có loss artifact | A4 với `--art-lambda 0` | sẵn sàng |
| A4 | Đầy đủ: đầu phụ + AGA | `yolo26n-seg-p2-artigate.yaml` + `isic2018_art.yaml` | sẵn sàng |
| A5 | A4 bỏ dán artifact tổng hợp | A4 với `--p-synth 0` | sẵn sàng |
| A6 | A4 với từng loại artifact riêng | đánh giá: `--strata` và `make_severity_sets.py` | sẵn sàng |
| A7 | AGA chỉ ở P2 so với P2 + P3 | `yolo26n-seg-p2-artigate-p2only.yaml` + `isic2018_art.yaml` | sẵn sàng |

A1s tách hiệu ứng "tăng cường dữ liệu bằng artifact" khỏi hiệu ứng của AGA. Chi phí ở 640×640 (đã fuse): A1 2,738 M tham số / 11,21 GFLOPs, A4 2,759 M / 12,26 GFLOPs.

Baseline YOLO26n-seg thường dùng `--model yolo26n-seg.yaml` (YAML có sẵn của Ultralytics). Baseline hai view và baseline cạo lông (DullRazor) dùng dữ liệu xuất với `--train-views`/`--eval-views`.

## Khác với code ICEEIS

- **Sửa lỗi nạp trọng số.** Trong ICEEIS, trọng số YOLO26n-seg được nạp vào mô hình trước khi gọi `model.train()`. Ultralytics lại dựng lại mô hình từ YAML bên trong `train()`, nên các trọng số đó bị bỏ và mô hình ICEEIS thực chất được train từ đầu (đã kiểm chứng trên Ultralytics 8.4.13). Bản này nạp trọng số trong `TransferSegmentationTrainer.get_model`, và có `tests/test_trainer.py` để lỗi không quay lại.
- **Một lớp "lesion"** (ISIC 2018 Task 1) thay cho 7 lớp chẩn đoán của HAM10000.
- **Biến đổi mask trực tiếp** cùng ảnh, rồi mới tách polygon. Cách này thay cho đoạn cắt polygon khoảng 1.200 dòng (`paired_geometry.py` cùng phần audit). Đánh giá dùng mask gốc chứ không dùng polygon vẽ lại.
- **Thước đo theo từng ảnh** gồm Dice, IoU, HD95 và boundary F1, cộng kiểm định ghép cặp (bootstrap trên hiệu số và Wilcoxon) so với baseline.
- **Bản đồ chuyển trọng số tự sinh từ YAML** (`stock_layer_map`) thay vì viết tay, nên A2–A7 dùng lại được. Test xác nhận bản đồ sinh ra trùng với bản đồ viết tay của ICEEIS.
- **Không mang sang:** augmentation bỏ qua lớp NV (chỉ có nghĩa với 7 lớp), GatedFusion v2B (bản thử, không nằm trong kế hoạch Bài 1), script vẽ hình kiến trúc (sẽ chuyển sang khi viết bài).

## Lưu ý tái lập

- Ultralytics được cố định ở bản 8.4.13. Các script cảnh báo nếu phiên bản cài đặt khác.
- CBAM trước đây dùng `AdaptiveMaxPool2d`, không có bản tính ngược tất định trên CUDA. Giờ max-pool toàn cục tính bằng `amax` (cùng kết quả, có test), nên với `deterministic: true` hai lần chạy cùng seed có thể khớp. Vẫn bắt buộc chạy 3 seed.
- Không cài `albumentations`: Ultralytics chỉ áp nó lên ảnh 3 kênh, nên baseline và các dòng ArtiGate (ảnh 7 kênh) sẽ khác augmentation.
- Khi import, Ultralytics thay hàm `cv2.imread` toàn cục, và ảnh xám trả về có dạng (H, W, 1). `read_mask` đã xử lý trường hợp này.
