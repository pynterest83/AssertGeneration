# Kiến trúc Multi-Agent: Hệ thống Tự động Sinh Test Assertion


## 1. Tổng quan (Overview)

Hệ thống này giống như một nhóm lập trình viên phối hợp với nhau để tự động viết ra các câu lệnh kiểm tra (assertion) cho mã nguồn. Thay vì nhắm mắt tin rằng code hiện tại đang đúng, nhóm này sẽ đọc code để hiểu mục đích thực sự của hàm, từ đó sinh ra các assertion chuẩn xác về mặt cú pháp.

**Mục tiêu:** Sinh ra assertion có chất lượng cao nhờ suy luận nhiều bước, đánh giá công bằng bằng cùng eval pipeline với solution_2.

**So sánh với solution_2:** Solution_2 gọi LLM 1 lần duy nhất (single-shot) và nhồi tất cả context vào prompt bất kể cần hay không. Solution_3 chia thành 3 agent, mỗi agent có **tools** để chủ động tìm thêm thông tin khi cần, suy luận từng bước trước khi sinh code.

**Framework:** Sử dụng **LangGraph** để quản lý luồng agent, state, và tool calling. Mỗi agent là một node trong graph, output được ghi vào shared state, agent tiếp theo đọc từ state đó.

---

## 2. LangGraph State

```python
class AssertionState(TypedDict):
    # Input (không đổi qua các bước)
    focal_method: str
    docstring: str
    test_prefix: str
    return_type: str
    test_name: str
    file_path: str

    # Agent 1 output
    analysis: str           # Bản phân tích cấu trúc hàm

    # Agent 2 output
    prediction: str         # Dự đoán trạng thái cuối

    # Agent 3 output
    assertion: str          # Dòng code assert
```

---

## 3. Tools

Nguyên tắc: Tool chỉ cần khi cung cấp thông tin **nằm ngoài prompt hiện tại** — code ở file khác, type ở class khác, v.v. LLM tự đọc được focal_method và test_prefix trong prompt, không cần tool parse lại. Tất cả tools chạy local trên `methods.jsonl`, **không tốn LLM call**.

### Tools cho Agent 1 (Code Analyzer)

Agent 1 nhận `focal_method` trong prompt — đọc code nó làm được. Nhưng nó **không biết** project xung quanh trông thế nào. Tools giúp nó tìm thêm context khi cần:

| Tool | Mô tả | Nguồn gốc |
|---|---|---|
| `search_method_usages` | Tìm các method khác trong project gọi tới focal method. Trả về tên + body (truncated). | Chuyển từ `prompt_builder.find_external_usages()` + `rank_by_similarity()` |
| `get_class_methods` | Liệt kê các method cùng class với focal method (sibling methods). Giúp hiểu ngữ cảnh class. | Mới — query từ `methods.jsonl` theo class name |
| `lookup_type_info` | Tra cứu một class/type trong project: có những method nào, field nào. Hữu ích khi focal method trả về hoặc nhận tham số kiểu custom. | Mới — query từ `methods.jsonl` theo class name |

**Khi nào Agent 1 dùng tool:** LLM tự quyết định. Với method đơn giản (getter/setter), nó không cần gọi tool nào. Với method phức tạp gọi tới nhiều class khác, nó sẽ dùng `lookup_type_info` để hiểu dependency. Đây là điểm khác biệt so với solution_2 (luôn nhồi top-K external context bất kể cần hay không).

### Tools cho Agent 2 (State Predictor)

Agent 2 chủ yếu **suy luận** — đọc analysis + test_prefix rồi trace logic. Phần lớn không cần tool. Ngoại trừ:

| Tool | Mô tả | Nguồn gốc |
|---|---|---|
| `lookup_type_info` | Giống Agent 1. Khi test_prefix tạo object kiểu lạ, Agent 2 cần biết constructor/method của type đó làm gì. | Dùng chung tool với Agent 1 |

### Tools cho Agent 3 (Assertion Generator)

Agent 3 viết code — cần biết pattern nào hợp lệ:

| Tool | Mô tả | Nguồn gốc |
|---|---|---|
| `get_assertion_examples` | Trả về 3–5 ví dụ assertion tương tự (few-shot) dựa trên similarity với prediction hiện tại. | Mới — dùng embedding từ `bow_embedding.py` hoặc `semantic_embedding.py` |

---

## 4. Nhiệm vụ của từng Agent

### Agent 1: Người Đọc Code (Code Analyzer)

* **Vai trò:** Phân tích tĩnh hàm cần kiểm tra — chỉ đọc code, chưa quan tâm test_prefix.
* **Đọc từ state:** `focal_method`, `docstring`, `return_type`
* **Tools có thể dùng:** `search_method_usages`, `get_class_methods`, `lookup_type_info`
* **Nhiệm vụ:**
    * Xác định: hàm nhận vào gì, trả về kiểu gì.
    * Liệt kê các nhánh logic chính: `if/else`, `try/catch`, các điều kiện trả về.
    * Nếu hàm gọi tới method/type lạ → dùng tool để tra cứu.
    * Ghi chú khi nào hàm trả null, khi nào ném exception, khi nào trả giá trị cụ thể.
* **Ghi vào state:** `analysis`. Ví dụ:
    ```
    Method: get() returns Object
    Branch 1: if delegate is null → throws NullPointerException
    Branch 2: if delegate.isDone() → returns delegate.get()
    Branch 3: else → blocks and returns delegate.get()
    Return type: Object (nullable)
    ```

### Agent 2: Người Dự Đoán (State Predictor)

* **Vai trò:** Đọc test_prefix + bản phân tích → dự đoán trạng thái cuối.
* **Đọc từ state:** `analysis`, `test_prefix`, `focal_method`
* **Tools có thể dùng:** `lookup_type_info` (khi test_prefix dùng type không quen)
* **Nhiệm vụ:**
    * Đọc test_prefix: biến nào được tạo, giá trị gì truyền vào hàm.
    * Dựa vào `analysis`, xác định test_prefix rơi vào nhánh logic nào.
    * Chốt dự đoán: biến kết quả có trạng thái gì?
* **Ghi vào state:** `prediction`. Ví dụ:
    ```
    test_prefix creates ListenableFutureAdapter with a mock delegate.
    delegate.isDone() returns false → get() will block.
    Result: listenableFuture0 != null (constructor succeeded).
    ```

### Agent 3: Người Viết Test (Assertion Generator)

* **Vai trò:** Chuyển dự đoán thành code assertion.
* **Đọc từ state:** `prediction`, `test_prefix`
* **Tools có thể dùng:** `get_assertion_examples` (few-shot reference)
* **Nhiệm vụ:**
    * Dựa vào `prediction`, viết câu lệnh assert JUnit 4.
    * Chỉ dùng biến tồn tại trong `test_prefix`.
    * Assertion hợp lệ: `assertEquals`, `assertTrue`, `assertFalse`, `assertNull`, `assertNotNull`, `assertSame`, `assertNotSame`, `fail`.
* **Ghi vào state:** `assertion`
* **Hậu xử lý:** `clean_prediction()` + `fix_assertion()` (tái sử dụng từ solution_2).

**Routing logic trong LangGraph:**
```
Agent 1 → Agent 2 → Agent 3 → Output (oracle_preds.csv)
```

---

## 5. Mapping: solution_2 → tools & components

### Chuyển thành Tool

| Code solution_2 | Thành tool | Agent dùng |
|---|---|---|
| `prompt_builder.find_external_usages()` + `rank_by_similarity()` | `search_method_usages` | Agent 1 |

### Tái sử dụng nguyên (không đổi)

| Code solution_2 | Vai trò | Giai đoạn |
|---|---|---|
| `extract_project_elements.py` | Trích method info → `methods.jsonl` | Phase A (chuẩn bị) |
| `vector_project_elements.py` | Tạo embeddings | Phase A (chuẩn bị) |
| `utils/bow_embedding.py` | Backing cho similarity search | Trong tools |
| `utils/semantic_embedding.py` | Backing cho similarity search | Trong tools |
| `utils/api_inference.py` | Client gọi LLM | Agent 1, 2, 3 |
| `clean_prediction()` + `fix_assertion()` | Hậu xử lý | Agent 3 output |
| `copy_test_prefix.py` (logic) | Test_prefix TOGA | Trước eval |
| `eval/` toàn bộ | Đánh giá SR | Phase C |

### Cần viết mới

| Thành phần | Mô tả |
|---|---|
| Tool `get_class_methods` | Query `methods.jsonl` theo class name |
| Tool `lookup_type_info` | Query `methods.jsonl` theo type/class → methods + fields |
| Tool `get_assertion_examples` | Few-shot retrieval dùng embedding similarity |
| Prompt template × 3 | Agent 1 (phân tích), Agent 2 (dự đoán), Agent 3 (sinh assertion) |
| LangGraph orchestrator | Graph definition: nodes, edges |

---

## 6. Quy trình chạy thực tế (Workflow)

### Phase A: Chuẩn bị (1 lần / project)

```
1. ExtractProjectElements     → methods.jsonl
2. VectorProjectElements      → methods_embeddings_*.jsonl (nếu dùng external)
3. Đọc inputs.csv + meta_llm.csv → danh sách test case
4. Load methods.jsonl vào memory → backing data cho tools
```

### Phase B: Sinh assertion (LangGraph — mỗi test case)

```
Với mỗi test case:
  State khởi tạo: focal_method, docstring, test_prefix, return_type

  Agent 1: đọc focal_method
           → [tùy chọn] gọi search_method_usages, get_class_methods, lookup_type_info
           → ghi analysis

  Agent 2: đọc analysis + test_prefix
           → [tùy chọn] gọi lookup_type_info
           → ghi prediction

  Agent 3: đọc prediction + test_prefix
           → [tùy chọn] gọi get_assertion_examples
           → ghi assertion
           → hậu xử lý: clean_prediction() + fix_assertion()
  
  Ghi kết quả: test_name, test_prefix, file_path, assertion → oracle_preds.csv
```

### Phase C: Eval (không đổi, dùng nguyên eval/)

```
1. copy_test_prefix           → thay test_prefix bằng TOGA (nếu cần)
2. aggregate_assertions.py    → inject assert vào project
3. run_compile.py             → compile + comment assert lỗi → Tce
4. run_test.py                → chạy test → Tfp, tính SR
```

Eval pipeline **giống hệt** solution_2: cùng Tce, Tfp, SR → so sánh công bằng.

---

## 7. Đầu vào / Đầu ra

### Đầu vào (giống solution_2)

```
input_dir/
└── {project}/
    ├── infer_input/
    │   ├── inputs.csv       # focal_method, docstring
    │   └── meta_llm.csv     # test_name, test_prefix, file_path, GT_output
    └── src/...              # Mã nguồn Java
```

### Đầu ra

```
output_dir/
└── {project}/
    ├── methods.jsonl                        # ExtractProjectElements
    ├── methods_embeddings_*.jsonl           # VectorProjectElements (--external)
    ├── analysis_*.jsonl                     # [MỚI] Output Agent 1
    ├── predictions_*.jsonl                  # [MỚI] Output Agent 2
    └── oracle_preds_{feature}_{model}.csv   # Kết quả cuối cùng
         Cột: test_name, test_prefix, file_path, assert_pred
```

Format `oracle_preds_*.csv` **giống hệt solution_2** → eval/ dùng được ngay.

---

## 8. Ước lượng chi phí LLM

| Thành phần | Số lần gọi LLM | Ghi chú |
|---|---|---|
| Agent 1 (phân tích) | 1 / test case | Tool calls chạy local, không tốn LLM call |
| Agent 2 (dự đoán) | 1 / test case | |
| Agent 3 (sinh assertion) | 1 / test case | |
| **Tổng** | **3 / test case** | Solution_2: 1 / test case |

Với ~500 test cases: ~1500 LLM calls (solution_2: 500 calls). Chi phí tăng 3x nhưng mỗi bước suy luận sâu hơn, prompt ngắn và tập trung hơn.

---

## 9. Tóm tắt điểm khác biệt so với solution_2

| | Solution 2 | Solution 3 |
|---|---|---|
| Kiến trúc | Single-shot (1 LLM call) | Multi-Agent + Tools (3 LLM calls) |
| Framework | Script Python thuần | LangGraph (state, routing, tool calling) |
| Context gathering | Luôn nhồi top-K external context | Agent tự quyết định gọi tool khi cần |
| Phân tích code | Không | Agent 1 phân tích có cấu trúc |
| Suy luận | Không | Agent 2 truy vết logic từng bước |
| Xử lý compile error | eval/ (comment + đếm Tce) | eval/ (giống hệt — so sánh công bằng) |
| Output format | oracle_preds.csv | oracle_preds.csv (giống hệt) |
| Eval pipeline | eval/ | eval/ (không đổi) |
