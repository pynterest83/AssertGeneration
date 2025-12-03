# Simple Evaluation Pipeline

Các script Python đơn giản và practical để chạy evaluation cho tất cả RQ (RQ1-RQ5).

## 🎯 Mục đích

Pipeline này cho phép bạn:
- Benchmark oracle generation solution của bạn trên tất cả RQ
- So sánh với baselines (TOGLL, TOGA, EvoSuite)
- Chạy từng RQ riêng lẻ hoặc tất cả cùng lúc
- Tích hợp dễ dàng với bất kỳ solution nào (LLM, rule-based, hybrid)

## 📁 Cấu trúc

```
eval_pipeline/
├── config.py              # Paths to datasets (auto-configured)
├── utils.py              # Helper functions (Maven, metrics, I/O)
├── eval_rq1.py           # RQ1: Intrinsic accuracy
├── eval_rq2.py           # RQ2: Generalization
├── eval_rq4.py           # RQ4: Test execution + mutation
├── eval_rq5.py           # RQ5: Bug detection
├── run_all.py            # Run all RQs at once
├── example_usage.py      # Examples
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd eval_pipeline
pip install -r requirements.txt
```

### 2. Implement your oracle generation function

Trong `run_all.py` (hoặc file riêng), implement function:

```python
def your_generate_oracle(test_prefix, focal_method=None, docstring=None):
    """
    Your oracle generation logic here
    
    Args:
        test_prefix: Test code without oracle
        focal_method: Method being tested (optional)
        docstring: Documentation (optional)
        
    Returns:
        oracle: String like "assertEquals(5, result);" or "exception"
    """
    # Your implementation
    # Can be: LLM, rule-based, search-based, etc.
    
    return oracle_statement
```

### 3. Run evaluation

```bash
# Quick test (small subset)
python run_all.py --quick

# Full evaluation
python run_all.py

# Run specific RQ
python eval_rq1.py --subset 100
python eval_rq2.py --projects async-http-client
python eval_rq4.py --projects async-http-client --versions togll
python eval_rq5.py --subset 100
```

## 📊 What Each RQ Tests

| RQ | Tests | Output | Time |
|----|-------|--------|------|
| **RQ1** | Accuracy on SF110 | Exact match | Fast (minutes) |
| **RQ2** | Generalization to new projects | Exact match per project | Medium (hours) |
| **RQ4** | Real test execution + mutation | Compile/pass rate, mutation score | Slow (hours-days) |
| **RQ5** | Real bug detection | Bugs found, precision/recall | Medium (hours) |

## 🔧 Key Features

### ✅ RQ2 và RQ4 có thể chạy Java code

- **RQ2:** Chỉ generate predictions (Python only)
- **RQ4:** Chạy Maven compile + test + PITest mutation testing
  - Tự động gọi `mvn` commands
  - Parse test results và mutation scores
  - Không cần manual setup!

### ✅ Pluggable architecture

Bạn chỉ cần implement 1 function duy nhất:

```python
def generate_oracle(test_prefix, focal_method, docstring) -> str
```

Không cần biết internal pipeline - chỉ input/output!

### ✅ Automatic metric computation

Pipeline tự động tính:
- Exact match accuracy
- Compilation rates
- Test pass rates
- Mutation scores
- Bug detection metrics
- Precision/Recall

## 📚 Examples

Xem `example_usage.py` cho examples đầy đủ:

```bash
# Example 1: Single RQ
python example_usage.py --example 1

# Example 2: Multiple RQs
python example_usage.py --example 2

# Example 3: Compare approaches
python example_usage.py --example 3

# Example 4: RQ4 với Maven
python example_usage.py --example 4
```

## 🎓 Detailed Usage

### RQ1: Intrinsic Evaluation

```bash
python eval_rq1.py --output results/rq1 --subset 100
```

Đánh giá accuracy trên SF110 validation set.

### RQ2: Generalization

```bash
# All projects
python eval_rq2.py --output results/rq2

# Specific projects
python eval_rq2.py --projects async-http-client commons-beanutils-1.9.4

# Quick test
python eval_rq2.py --projects async-http-client --subset 50
```

### RQ4: Test Execution

```bash
# Test compilation and execution
python eval_rq4.py --projects async-http-client --versions togll

# Include mutation testing (slow!)
python eval_rq4.py --projects async-http-client --versions togll --mutation

# Compare all versions
python eval_rq4.py --projects async-http-client --versions evosuite togll no_oracle
```

**Note:** RQ4 cần Maven và Java. Script tự động chạy:
- `mvn clean compile` - Compile
- `mvn test` - Run tests
- `mvn pitest:mutationCoverage` - Mutation testing (nếu --mutation)

### RQ5: Bug Detection

```bash
# Generate oracles và classify
python eval_rq5.py --output results/rq5 --subset 100

# Analyze existing Docker results
python eval_rq5.py --analyze-only --togll-results /path/to/results --toga-results /path/to/toga
```

**Note:** Full bug detection cần Docker (xem RQ5 README). Script này chỉ generate oracles và classify types.

## 📈 Output Structure

```
results/
├── rq1/
│   ├── rq1_predictions.json     # All predictions
│   └── rq1_metrics.json         # Accuracy metrics
├── rq2/
│   ├── project1/
│   │   ├── predictions.json
│   │   └── metrics.json
│   ├── project2/...
│   ├── rq2_overall_metrics.json
│   └── rq2_summary.csv
├── rq4/
│   ├── project1/
│   │   ├── togll_metrics.json
│   │   ├── togll_test.log
│   │   └── togll_mutation.log
│   ├── rq4_overall_metrics.json
│   └── rq4_comparison.csv
├── rq5/
│   ├── oracle_predictions.csv
│   ├── rq5_metrics.json
│   └── classification_report.json
└── all_results.json             # Summary of all RQs
```

## 🔌 Integration với solution của bạn

### Option 1: Direct implementation

Edit `run_all.py` hoặc các eval scripts:

```python
def your_generate_oracle(test_prefix, focal_method=None, docstring=None):
    # Load your model
    # model = YourModel.load()
    
    # Generate
    # oracle = model.generate(test_prefix, focal_method, docstring)
    
    return oracle
```

### Option 2: External module

```python
# your_model.py
class YourModel:
    def generate_oracle(self, test_prefix, focal_method, docstring):
        # Your implementation
        return oracle

# run_all.py
from your_model import YourModel

model = YourModel()
run_all_evaluations(
    generate_oracle_fn=model.generate_oracle,
    ...
)
```

### Option 3: API-based

```python
import requests

def api_generate_oracle(test_prefix, focal_method, docstring):
    response = requests.post('http://your-api/generate', json={
        'test_prefix': test_prefix,
        'focal_method': focal_method,
        'docstring': docstring
    })
    return response.json()['oracle']
```

## ⚙️ Configuration

Edit `config.py` nếu paths khác:

```python
# Paths to datasets
RQ1_DATASET = Path("path/to/rq1/dataset.pickle")
RQ2_INFERENCE_DATA = Path("path/to/rq2/data")
# ...
```

## 🐛 Troubleshooting

### Maven not found

```bash
sudo apt install maven
```

### Java version issues

```bash
# Check Java version
java -version

# RQ4 projects cần Java 8 hoặc 11
sudo update-alternatives --config java
```

### Import errors

```bash
pip install -r requirements.txt
```

### RQ4 tests fail

- Check logs trong `results/rq4/project_name/*.log`
- Có thể do oracle syntax errors hoặc project dependencies

## 📝 Notes

- **RQ1:** Cần tokenizer để decode pickle data (simplified trong script)
- **RQ2:** Chỉ string matching, không run code
- **RQ4:** Chạy code thật, cần Maven/Java
- **RQ5:** Full bug detection cần Docker + Defects4J

## 🎯 Recommended Workflow

```bash
# 1. Quick test với subset nhỏ
python run_all.py --quick --rqs rq1 rq2

# 2. Nếu OK, chạy full RQ1 và RQ2
python eval_rq1.py
python eval_rq2.py

# 3. Test RQ4 với 1 project
python eval_rq4.py --projects async-http-client

# 4. Nếu OK, chạy full benchmark
python run_all.py
```

## 📧 Support

Nếu có vấn đề, check:
1. Logs trong `results/`
2. Error messages
3. Example usage in `example_usage.py`

