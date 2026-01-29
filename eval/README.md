# Evaluation Pipeline

## Overview
This pipeline evaluates test oracle generation by:
1. Fixing compilation errors in generated tests
2. Running tests and calculating success rate

## Requirements
- Python 3.7+
- Maven 3.6+
- JDK 1.8+

## Quick Start

### Step 1: Fix Compilation Errors
```bash
python eval/run_compile.py \
    --input_dir data/RQ1 \
    --output_dir data/RQ1_fixed
```

This will:
- Compile each project in `data/RQ1`
- Automatically comment lines with compilation errors
- Iterate until `BUILD SUCCESS` or no more errors
- Save fixed projects to `data/RQ1_fixed`

### Step 2: Run Tests and Calculate Success Rate
```bash
python eval/run_test.py \
    --input_dir data/RQ1_fixed
```

This will:
- Run `mvn test` for each project
- Count compilation errors (Tce), test failures (Tfp), and total tests (T)
- Calculate Success Rate: `SR = (T - Tce - Tfp) / T`
- Save results to `data/RQ1_fixed/test_results.json`

## Results
Results are saved as JSON with:
- `total_tests`: Total number of tests
- `compilation_errors`: Number of compilation errors
- `false_positives`: Number of test failures
- `success_rate`: Success rate (0.0 to 1.0)

## Options
```bash
# Process specific projects only
python eval/run_compile.py --input_dir data/RQ1 --output_dir data/RQ1_fixed \
    --projects project1 project2

# Custom timeout (seconds)
python eval/run_compile.py --input_dir data/RQ1 --output_dir data/RQ1_fixed --timeout 600
```
