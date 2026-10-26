# Gắn nhãn artifact cho 1.000 ảnh test ISIC 2018

Nhãn này chia tập test thành các nhóm (có lông, có thước, có mực, có góc tối, không có artifact) để
so sánh A4 với A1 theo từng loại artifact. **Phải làm xong và commit trước khi chấm bất kỳ mô hình
nào trên tập test.** Bài báo ghi rõ điều này.

## Định nghĩa (đánh dấu "có" nếu thấy ở bất kỳ đâu trong ảnh, ở cỡ hiển thị của công cụ)

| Phím | Loại | Đánh dấu "có" khi | Không tính |
|---|---|---|---|
| `h` | Lông (hair) | Có ít nhất một sợi lông/tóc nhìn thấy được, bất kể màu (đen, nâu, xám, trắng), nằm trên hay ngoài tổn thương, kể cả lông nằm dưới lớp gel | Vân da, mạch máu, cấu trúc sắc tố của tổn thương |
| `r` | Thước (ruler) | Có thước đo, vạch chia, số hoặc dải hiệu chuẩn in sẵn | Chữ hoặc số do phần mềm chèn lên ảnh, viền khung ảnh |
| `m` | Mực đánh dấu (marker) | Có vết bút đánh dấu trên da (tím, xanh, đen): chấm, vòng, nét, kể cả mực mờ | Vạch in trên thước (đã tính là thước), miếng dán màu |
| `d` | Góc tối (dark corner) | Có vùng tối hình cung do khẩu độ máy soi da ở ít nhất một góc | Viền đen thẳng do cắt ảnh, bóng tối không hình cung |

Không gắn nhãn bọt gel, viền gel, miếng dán. Một ảnh có thể có nhiều loại cùng lúc; không có loại nào thì để trống và bấm Enter.

## Quy trình

1. **Người gắn nhãn 1 (anh Hải): toàn bộ 1.000 ảnh**, xem ảnh gốc (rõ lông mảnh hơn ảnh 640):

   ```bash
   python scripts/label_artifacts.py --manifest data/splits/isic2018.csv --split test \
       --out data/splits/isic2018_test_artifacts_a1.csv
   ```

   Phím: `h r m d` bật/tắt, Enter hoặc Space lưu và sang ảnh sau, `b` quay lại, `q` thoát (đã lưu, chạy lại sẽ tiếp tục).
   Khoảng 1–1,5 giờ; nghỉ giữa chừng thoải mái. Không xem kết quả mô hình hay mask khi gắn nhãn.

2. **Người gắn nhãn 2: 200 ảnh của tập con đã chọn trước** (`data/splits/isic2018_test_kappa_subset.csv`, chọn ngẫu nhiên với seed 20261026).
   Trước đó đọc bảng định nghĩa và tập cùng người 1 trên khoảng 20 ảnh **train** (không dùng ảnh test để tập):

   ```bash
   python scripts/label_artifacts.py --manifest data/splits/isic2018.csv --split train \
       --out runs/annotation_practice_a2.csv          # file tập, không commit
   python scripts/label_artifacts.py --manifest data/splits/isic2018.csv --split test \
       --ids data/splits/isic2018_test_kappa_subset.csv --out data/splits/isic2018_test_artifacts_a2.csv
   ```

   Người 2 gắn độc lập, không xem nhãn của người 1.

   Nếu người 2 làm trên máy riêng, không có repo: `python scripts/make_annotator_bundle.py --out <thư mục>`
   tạo sẵn 200 ảnh test và 20 ảnh train để tập, ở đúng cỡ người 1 thấy trong công cụ (cạnh dài 900 px,
   JPEG chất lượng 95), kèm bản sao `label_artifacts.py`. Người 2 chạy công cụ với `--images` trên thư mục
   đó. Người 1 gắn nhãn cùng 20 ảnh tập (`--images <thư mục>/images/practice_20 --out runs/annotation_practice_a1.csv`)
   để hai người so và bàn trước khi người 2 gắn nhãn tập con.

3. **Độ đồng thuận và nhãn cuối:**

   ```bash
   python scripts/artifact_agreement.py --a1 data/splits/isic2018_test_artifacts_a1.csv \
       --a2 data/splits/isic2018_test_artifacts_a2.csv --out runs/agreement/isic2018_test
   ```

   Hai người cùng xem các ảnh trong `runs/agreement/isic2018_test_disagreements.csv`, thống nhất và điền cột `resolved` (0 hoặc 1), rồi:

   ```bash
   python scripts/artifact_agreement.py --a1 data/splits/isic2018_test_artifacts_a1.csv \
       --a2 data/splits/isic2018_test_artifacts_a2.csv --out runs/agreement/isic2018_test \
       --resolved runs/agreement/isic2018_test_disagreements.csv --final data/splits/isic2018_test_artifacts.csv
   ```

4. **Commit** ba file nhãn (`_a1`, `_a2`, nhãn cuối) và báo cáo kappa (`runs/agreement/isic2018_test.md` chép vào mô tả commit). Chỉ sau commit này mới chấm tập test. `scripts/eval.py --split test` kiểm tra điều này: chưa có file nhãn cuối, file sai định dạng, file chưa commit hoặc có sửa chưa commit thì từ chối chấm; commit của file được ghi vào `eval_summary.json` (đó là mã commit điền vào Supplementary S6.1).
