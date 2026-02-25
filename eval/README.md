# Pipeline đánh giá (Evaluation Pipeline)

## Tổng quan

Thư mục `eval/` chứa **4 file Python** dùng để đánh giá chất lượng assert do mô hình sinh ra trên các project Java.

### Đại lượng đánh giá

| Ký hiệu | Ý nghĩa |
|---------|---------|
| **T** | Tổng số test **assertion** (ground-truth khác `exception`) |
| **Tce** | Số test assertion bị loại vì **lỗi biên dịch** (assert bị comment) |
| **Tfp** | Số test assertion **compile được nhưng chạy fail** (Failures + Errors) |
| **SR** | Tỉ lệ thành công tổng thể |

Công thức:

$$
SR = \frac{T - Tce - Tfp}{T}
$$

SR đo tỉ lệ test assertion **compile được** và **chạy pass**.

### Thứ tự chạy (bắt buộc)

| Bước | File | Mô tả |
|------|------|-------|
| **1** | `aggregate_assertions.py` | Chèn assert dự đoán vào test Java, tạo bản project dùng cho đánh giá |
| **2** | `run_compile.py` | Compile project, comment assert lỗi, ghi `compile_results.json` |
| **3** | `run_test.py` | Chạy test, đếm Tfp, tính SR, ghi `test_results.json` |

**Lưu ý**: `comment_incompatible_assertions.py` không chạy trực tiếp; được `run_compile.py` gọi nội bộ khi có lỗi compile.

---

## Chi tiết logic từng file

### 1. `aggregate_assertions.py`

**Mục đích**: Lấy dự đoán từ mô hình (oracle_preds.csv) và chèn/thay thế assert trong file test Java để tạo project đã “gắn assert” phục vụ đánh giá.

**Luồng thực thi khi chạy script**:

1. Parse args: `-i base_dir`, `-o output_dir`, `--oracle_csv`, `--meta_csv` (optional).
2. Nếu `output_dir` tồn tại → xóa và copy lại: `shutil.copytree(base_dir, output_dir)`.
3. Gọi `aggregate_assertions()` → sinh file `.txt` tạm chứa thân test đã thay assert.
4. Gọi `copy_assertions()` → merge nội dung `.txt` vào `.java` và xóa `.txt`.

#### Logic `aggregate_assertions(base_dir, output_dir, oracle_csv, meta_csv)`

- **Đọc oracle_preds.csv** (hoặc `oracle_csv`):
  - Cột bắt buộc: `test_name`, `test_prefix`, `file_path`, `assert_pred`.
- **Đọc meta_llm.csv** (nếu có):
  - Lấy `exception_tests` = tập các `test_name` có `GT_output == 'exception'`.
  - Các test này **không** được inject assert (vì ta chỉ đánh giá test assertion).
- **Với mỗi dòng** `(prefix, file_path, test_name, assert_pred)`:
  - Nếu `test_name in exception_tests` → bỏ qua, tăng `total_skipped_exception`.
  - **Chuẩn hóa path**:
    - Nếu `file_path` chứa `/` → `relative_path = "/".join(fpath.split("/")[1:])` (bỏ segment đầu, thường là tên project).
    - `loc = output_dir + "/" + relative_path`, đổi `.java` → `.txt` → ví dụ `output_dir/src/.../Foo_ESTest.txt`.
  - **Ghi vào file `.txt`**:
    - `count[loc]` đếm số test đã ghi vào file đó. Lần đầu (`count[loc]==0`) → truncate file.
    - Ghi: ` @Test(timeout = 4000)\n` + `prefix` (đã thay assert) + `\n`.
  - **Thay assert trong prefix**:
    - Regex `assert\w*\(.*\)` tìm mọi assert cũ.
    - Nếu `assert_pred` chứa chuỗi `"assert"` → thay bằng `assert_pred`; ngược lại thay bằng rỗng.
    - `re.sub(assert_re, lambda m: new_assertion, str(prefix))`.

#### Logic `copy_assertions(base_dir, output_dir, oracle_csv)`

- Với mỗi `file_path` duy nhất trong oracle_preds:
  - `java_test_file` = đường dẫn file `.java` trong `output_dir`.
  - `aggregated_test_file` = cùng path nhưng `.txt`.
  - Nếu không tồn tại `.txt` → bỏ qua.
  - Đọc toàn bộ nội dung `a_tests` từ `.txt`.
  - Mở `.java` (đã copy từ base), đọc từng dòng:
    - Ghi lại từng dòng cho đến khi gặp dòng chứa `@Test(timeout = 4000)`.
    - Khi gặp → **ghi `a_tests`** thay cho phần test cũ, rồi `break`.
  - Ghi thêm `}\n` để đóng class.
  - Xóa file `.txt`.

**Kết quả**: Project trong `output_dir` có các file test Java đã được thay toàn bộ phần test (từ `@Test` trở đi) bằng thân test mới với assert do mô hình dự đoán.

**CLI**:

```bash
python eval/aggregate_assertions.py -i BASE_DIR -o OUTPUT_DIR \
  [--oracle_csv path/to/oracle_preds.csv] \
  [--meta_csv path/to/meta_llm.csv]
```

---

### 2. `comment_incompatible_assertions.py`

**Mục đích**: Tiện ích nội bộ parse log compile Maven, tìm các dòng assert gây lỗi và comment chúng để project có thể compile thành công.

**Logic `comment_assertions(error_log_path)`**:

1. **Parse log**:
   - Đọc từng dòng, chỉ xử lý dòng có `[ERROR]` và `.java:[`.
   - Mỗi dòng lỗi Maven thường dạng: `[ERROR] /path/to/Foo.java:[123,45] message`
   - Trích:
     - `file_path`: chuỗi từ `[ERROR] ` đến hết `.java`.
     - `line_no`: số trong `java:[lineNo,col]` (chữ số trước dấu phẩy).
   - Gom vào `file_to_error_lines[file_path].add(line_no)`.

2. **Sửa file Java**:
   - Với mỗi `(file_path, line_numbers)`:
     - Nếu file không tồn tại → bỏ qua.
     - Đọc file, duyệt từng dòng với `line_count` (1-based).
     - Nếu `str(line_count) in line_numbers` và `line.strip().startswith("assert")`:
       - Thêm tiền tố `//COMPILE_ERROR ` vào đầu dòng → comment dòng assert đó.
       - Tăng `total_commented`.
     - Ghi lại toàn bộ nội dung file.

3. Trả về `total_commented`.

**Chạy độc lập** (ít dùng):

```bash
python eval/comment_incompatible_assertions.py --error_log path/to/compilation_error.txt
```

Trong pipeline, `run_compile.py` gọi `comment_assertions()` mỗi khi phát hiện lỗi compile.

---

### 3. `run_compile.py`

**Mục đích**: Compile từng project trong `input_dir`, tự động comment assert gây lỗi, lặp cho đến khi BUILD SUCCESS hoặc không sửa được nữa; ghi tổng số assert bị comment (**Tce**) vào `compile_results.json`.

**Logic `run_maven_compile(project_path)`**:

- Chạy `mvn test-compile -fae -B -Drat.skip=true` trong thư mục project.
- Gộp stdout + stderr, ghi vào `project_path/compilation_error.txt`.
- Trả về chuỗi log.

**Logic `process_project(project_path)`**:

```
total_commented = 0
while True:
    log = run_maven_compile(project_path)
    if "BUILD SUCCESS" in log:
        return (True, total_commented)
    if ".java:[" not in log:
        # Lỗi không phải compile Java (vd: dependency, plugin)
        return (False, total_commented)
    commented = comment_assertions(compilation_error.txt)
    total_commented += commented
    if commented == 0:
        # Không comment thêm được → lỗi không phải do assert
        return (False, total_commented)
```

**Logic `main()`**:

- Lấy danh sách project: mọi thư mục con của `input_dir` trừ `results`.
- Với mỗi project: `success, commented = process_project(...)`.
- Cộng dồn `total_Tce += commented`.
- Ghi `compile_results.json`:
  ```json
  {
    "total_Tce": <tổng>,
    "projects": {
      "<tên_project>": { "success": bool, "Tce": int }
    }
  }
  ```

Compile thực hiện **trực tiếp trong từng project** (in-place), không copy sang thư mục khác.

**CLI**:

```bash
python eval/run_compile.py --input_dir PATH [--timeout 300] [--projects p1 p2 ...]
```

---

### 4. `run_test.py`

**Mục đích**: Chạy `mvn test` cho từng project, đếm Tfp (Failures + Errors), lấy T và Tce từ file khác, rồi tính SR và ghi `test_results.json`.

**Logic**:

1. **T** = `count_total_tests_from_meta(meta_csv)`:
   - Đọc `meta_llm.csv`, lọc `GT_output != 'exception'`.
   - Số dòng còn lại = tổng test assertion toàn benchmark.

2. **Tce** = `load_compile_results(input_dir)`:
   - Đọc `input_dir/compile_results.json`, lấy `total_Tce`.

3. **Danh sách project**:
   - Loại trừ: `results`, `compiled`, `infer_input`, `toga_output`, `.github`, `travis`.

4. **Với mỗi project**:
   - Chạy `mvn test -B -Drat.skip=true`.
   - Parse thư mục `**/surefire-reports/*.txt`, tìm dòng dạng `Tests run: X, Failures: Y, Errors: Z`.
   - Cộng `Failures + Errors` vào `total_Tfp`.

5. **SR** = `(T - Tce - total_Tfp) / T` nếu T > 0, else 0.

6. Ghi `test_results.json`:
   ```json
   {
     "T": int, "Tce": int, "Tfp": int, "SR": float,
     "projects": { "<tên>": { "Tfp": int } }
   }
   ```

**CLI**:

```bash
python eval/run_test.py --input_dir PATH --meta_csv path/to/meta_llm.csv [--output path/to/test_results.json]
```

---

## Thứ tự chạy đầy đủ (ví dụ)

Giả sử:

- `data/RQ1_raw`: project gốc (có cấu trúc thư mục, chưa inject assert).
- `data/RQ1_raw/outputs/oracle_preds.csv`: dự đoán từ mô hình.
- `data/RQ1_raw/meta_llm.csv`: metadata (cột `test_name`, `GT_output`).

**Bước 1** – Chèn assert, tạo bản đánh giá:

```bash
python eval/aggregate_assertions.py \
  -i data/RQ1_raw \
  -o data/RQ1 \
  --oracle_csv data/RQ1_raw/outputs/oracle_preds.csv \
  --meta_csv data/RQ1_raw/meta_llm.csv
```

→ `data/RQ1` chứa project đã inject assert.

**Bước 2** – Compile và ghi Tce:

```bash
python eval/run_compile.py --input_dir data/RQ1 --timeout 300
```

→ Sinh `data/RQ1/compile_results.json`.

**Bước 3** – Chạy test và tính SR:

```bash
python eval/run_test.py \
  --input_dir data/RQ1 \
  --meta_csv data/RQ1_raw/meta_llm.csv \
  --output data/RQ1/test_results.json
```

→ Sinh `data/RQ1/test_results.json`.

---

## Cấu trúc file output

| File | Vị trí | Nội dung chính |
|------|--------|-----------------|
| `compile_results.json` | `input_dir/` | `total_Tce`, `projects[name].{success, Tce}` |
| `test_results.json` | Theo `--output` | `T`, `Tce`, `Tfp`, `SR`, `projects[name].Tfp` |

---

## Yêu cầu môi trường

- Python 3.7+
- pandas, tqdm
- Maven 3.6+
- JDK 1.8+

Chạy các script từ **thư mục gốc project** (chứa thư mục `eval/`).
