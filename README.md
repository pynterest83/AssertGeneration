# AssertGen — Multi-Agent Assertion Generation for Unit Tests

AssertGen is an LLM-based system that generates JUnit / pytest / Jest assertion oracles
for unit tests, given a focal method and a test prefix.

Unlike template classifiers (TOGA) or one-shot LLM prompts (CLAP), AssertGen runs the
generation through a **4-agent LangGraph pipeline** backed by a **tree-sitter +
KùzuDB code graph**. The agents look up real method bodies, fields, and call edges
from the project under test, so the assertion they emit is grounded in the actual
implementation rather than the LLM's prior.

```
test_prefix + focal_method
        │
        ▼
┌────────────────────────┐
│ exception_classifier   │──── is_exception ────► END (write "exception")
└────────────────────────┘
        │ (assertion path)
        ▼
┌────────────────────────┐    search_relevant_code
│ code_analyzer          │◄──── (Kùzu Cypher) ──┐
└────────────────────────┘                       │
        │                                        │
        ▼                                        │
┌────────────────────────┐                       │
│ state_predictor        │◄──────────────────────┘
└────────────────────────┘
        │
        ▼
┌────────────────────────┐
│ assertion_generator    │  ── final assertion
└────────────────────────┘
```

## Repository layout

```
.
├── src/
│   ├── solution/         # the 4-agent pipeline (the thesis contribution)
│   ├── eval/             # evaluation scripts per research question
│   │   ├── RQ1/          #   compile + test execution accuracy
│   │   ├── RQ2/          #   mutation testing (PIT)
│   │   └── RQ3/          #   Defects4J bug-detection comparison
│   └── extension/        # VS Code extension wrapping the pipeline
│       ├── backend/      #   FastAPI server exposing build-graph + infer
│       └── frontend/     #   VS Code extension (TypeScript)
│
├── data/
│   ├── RQ1/{input,output}    # 25 Apache Commons / SF100-like projects
│   ├── RQ2/{input,output}    # PIT-ready test suites + mutation outputs
│   └── RQ3/{input,output}    # 11 Defects4J projects
│
├── CLAP/                 # CLAP baseline (Chat-like Asserts Prediction) — vendored
├── toga-reflect/         # TOGA artifact for replication (ICSE'22)
├── togll/                # TOGLL artifact (LLM-based oracle generation)
├── archieved/            # earlier iterations (solution_2 … solution_5)
└── references/           # related papers (PDFs)
```

## Research Questions

| RQ  | Question | Pipeline + script |
|-----|----------|-------------------|
| RQ1 | How accurate are the generated assertions on real-world Apache projects? | `src/solution/run_pipeline.py` → `src/eval/RQ1/{run_compile.py,run_test.py,aggregate_assertions.py}` |
| RQ2 | Do generated assertions retain mutation-detection strength compared to developer-written assertions? | `src/eval/RQ2/{setup_rq2_input.py,prepare_for_pit.py,run_pit.py,analyze_mutations.py}` |
| RQ3 | How does AssertGen compare against TOGA / TOGLL on Defects4J bug detection? | `src/eval/RQ3/{prepare_d4j_data.py,convert_toga_d4j_infer_format.py,result_analysis.py}` |

Each RQ uses the same generation pipeline (`src/solution/`); only the evaluation harness differs.

## Quick start

### Prerequisites

- Python ≥ 3.11 (recommended via Conda env named `oracle_generation` — used throughout the codebase)
- Docker (for Kùzu Explorer — optional, for graph visualization)
- An LLM endpoint that speaks the OpenAI-compatible Chat Completions API (e.g. Qwen, GPT-4, etc.)

### Install

```bash
conda create -n oracle_generation python=3.11 -y
conda activate oracle_generation
pip install -r src/solution/requirements.txt   # if present, otherwise install:
pip install langchain langchain-openai langgraph pydantic tree-sitter \
            kuzu pandas tqdm rich httpx python-dotenv
```

### Configure

Copy the env template and fill in your LLM endpoint:

```bash
cp src/solution/.env.example src/solution/.env
# edit src/solution/.env — set API_ENDPOINT, API_KEY, MODEL_NAME, INPUT_DIR, OUTPUT_DIR
```

### Run the pipeline

```bash
cd src/solution
python run_pipeline.py --project Csv --language java
```

This will:
1. Parse the project at `$INPUT_DIR/Csv/` with tree-sitter, build a KùzuDB code graph under `$INPUT_DIR/Csv/.code_graph`.
2. Read test items from `$INPUT_DIR/Csv/infer_input/{inputs.csv, meta_llm.csv}`.
3. Run each item through the 4-agent graph (parallelized via `MAX_WORKERS`).
4. Stream results to `$OUTPUT_DIR/Csv/oracle_preds_<model>.csv`, with incremental
   checkpointing — restart resumes from where it left off.

See [src/solution/README.md](src/solution/README.md) for full configuration and
architecture details.

### Reproduce an RQ

```bash
# RQ1 — compile + execute generated assertions on commons-numbers
python src/eval/RQ1/run_compile.py --project commons-numbers-1.0-src
python src/eval/RQ1/run_test.py --project commons-numbers-1.0-src
python src/eval/RQ1/aggregate_assertions.py

# RQ2 — mutation testing with PIT
python src/eval/RQ2/setup_rq2_input.py --project commons-csv
python src/eval/RQ2/prepare_for_pit.py --project commons-csv
python src/eval/RQ2/run_pit.py --project commons-csv
python src/eval/RQ2/analyze_mutations.py

# RQ3 — Defects4J comparison
python src/eval/RQ3/prepare_d4j_data.py
python src/eval/RQ3/result_analysis.py
```

## Try it interactively

Two ways to drive the pipeline without writing CLI flags:

1. **VS Code extension** ([src/extension/](src/extension/)) — packages the FastAPI
   backend + a TypeScript extension. Opens a graph panel (vis-network) showing
   classes, methods, and call edges.
2. **Demo scripts** ([src/solution/demo/](src/solution/demo/)) — step-by-step
   walkthrough used in the thesis defense: parse → CST → entities → graph build
   in one terminal; LangGraph trace viewer for individual samples in another. See
   [src/solution/demo/README.md](src/solution/demo/README.md).

## Baselines vendored in this repo

| Baseline | Where | Used in |
|----------|-------|---------|
| TOGA (Dinella et al., ICSE'22) | [toga-reflect/](toga-reflect/) | RQ1 test-prefix source, RQ3 comparison |
| TOGLL (Hossain et al., 2024) | [togll/](togll/) | RQ3 comparison |
| CLAP (Yang et al., 2024) | [CLAP/](CLAP/) | Related work |

The earlier iterations of our approach (template extractor, single-prompt LLM,
RAG-based retrieval) live in [archieved/](archieved/) for reference. The current
pipeline is `src/solution/`, which started as `archieved/solution_5/`.

## Output format

Each generation run produces a CSV with one row per test, columns:

| column | meaning |
|--------|---------|
| `test_name` | fully-qualified test method name |
| `test_prefix` | the test body up to the assertion point (input to the pipeline) |
| `file_path` | path to the test source file |
| `assert_pred` | predicted assertion text; the literal string `exception` if the test was classified as exception-expecting |

For exception-expecting tests, AssertGen short-circuits after the
`exception_classifier` node — only one LLM call is consumed per such sample.
