# `src/solution/` — Multi-Agent Assertion Generation Pipeline

The core contribution of this thesis: a [LangGraph](https://langchain-ai.github.io/langgraph/)
pipeline of 4 specialized agents that produces a single assertion statement for a
unit test, given the focal method and the test prefix. The agents are grounded by
a per-project **tree-sitter + KùzuDB code graph** that lets them retrieve real
method bodies and call edges instead of hallucinating from priors.

## Architecture at a glance

### Two phases per run

1. **Index** — `CodeGraph(project_path, language)` parses every source file with
   tree-sitter, then materializes classes, methods, fields, and the `HAS_METHOD`
   / `CALLS` / `EXTENDS` / `IMPLEMENTS` / `HAS_FIELD` edges into a KùzuDB at
   `<project_path>/.code_graph`. A sentinel marker file (`.code_graph.complete`)
   is written on success — subsequent runs open the DB **read-only**, so parallel
   inference workers and the VS Code extension can share one indexed graph.
2. **Infer** — for each test item, `build_graph(llm, code_graph, language)`
   compiles a fresh `StateGraph` and invokes it. A new compiled graph **per
   sample** is intentional: each sample needs its own tool closure (the
   per-agent query counter and the external-class cache) for thread safety. The
   overhead is ~10–50 ms, dwarfed by LLM latency.

### Node sequence

```
START → exception_classifier ──(is_exception)──► END
                │
                ▼ (assertion path)
        code_analyzer → state_predictor → assertion_generator → END
```

| Node | Type | Output |
|------|------|--------|
| [`exception_classifier`](agents/exception_classifier.py) | plain LLM, structured output | `is_exception: bool`, short-circuits the rest on `true` |
| [`code_analyzer`](agents/code_analyzer.py) | ReAct agent with `search_relevant_code` + `finish` tools | `CodeAnalysis` — signature, branches, return conditions, dependencies |
| [`state_predictor`](agents/state_predictor.py) | ReAct agent (same tools) | `StatePrediction` — variable states, observable state, assertion target |
| [`assertion_generator`](agents/assertion_generator.py) | plain LLM, structured output | a single assertion string |

The agents communicate through `AssertionState` ([state.py](state.py)), a `TypedDict`
that flows through the graph. `state_predictor` is pre-seeded with the
`known_external` class names that `code_analyzer` already confirmed as
out-of-project, so it doesn't re-query them.

### The code-search tool

[`tools/definitions.py`](tools/definitions.py) exposes one tool to the ReAct agents:

```python
search_relevant_code(class_name: str = '', method_name: str = '',
                     include_callees: bool = False) -> str
```

Behind a single lock, the tool closure carries:

- **`query_counts`** — a 2nd query of the same `(class, method)` pair returns a
  "STOP, you already have this" message instead of re-running the Cypher.
- **`external_cache`** — when a lookup returns no rows, the class is marked as
  external (stdlib / 3rd-party). Any subsequent query referencing that class
  short-circuits with a "STOP ALL TOOL CALLS" message.

Both state buckets are reset per node via `tools[0].reset_counter(known_external=...)`.
A `threading.local()` would not work — LangGraph's `ToolNode` dispatches parallel
tool calls to separate ThreadPoolExecutor threads. A shared dict + lock is
required.

### Language abstraction

Three languages are supported. The config ([lang_config.py](lang_config.py))
controls test framework, assertion verbs, and file extensions:

| Language | Test framework | Assertion API | `class_name` required? |
|----------|----------------|---------------|------------------------|
| Java | JUnit 4 | `assertEquals`, `assertTrue`, ... | Always (for method lookup) |
| Python | pytest | `assert x == y`, `assert x is None`, ... | Optional (`standalone_only`) |
| JavaScript | Jest | `expect(x).toBe(y)`, ... | Required for method lookup |

Tree-sitter queries per language live in [parser/queries.py](parser/queries.py);
Cypher templates in [helpers/db_utils.py](helpers/db_utils.py) (`Queries.*` +
`SCHEMA_SQL`).

## Configuration

Environment variables (load via `.env`, see [.env.example](.env.example)):

| Var | Required | Default | Notes |
|-----|----------|---------|-------|
| `API_ENDPOINT` | yes | — | OpenAI-compatible base URL |
| `API_KEY` | yes | `EMPTY` | Bearer token |
| `MODEL_NAME` | yes | — | e.g. `qwen3-coder-next` |
| `INPUT_DIR` | yes | — | Root that contains `<project>/infer_input/` |
| `OUTPUT_DIR` | yes | — | Root for `<project>/<output_file>.csv` |
| `MAX_WORKERS` | no | `8` | Parallel samples |
| `MAX_TOKENS` | no | `4096` | Per LLM call |
| `TEMPERATURE` | no | `0.0` | |
| `STREAMING` | no | `true` | Set `false` for SSE endpoints that hang on structured output |
| `LANGCHAIN_TRACING_V2` | no | — | Set `true` + `LANGCHAIN_API_KEY` + `LANGCHAIN_PROJECT` for LangSmith |

## CLI

```bash
cd src/solution            # imports are flat (e.g. `from code_graph import …`)
python run_pipeline.py \
    --project Csv \
    --language java \
    [--input_dir DIR] [--output_dir DIR] \
    [--max_workers N] [--temperature F] [--max_tokens N] \
    [--limit N] [--offset N] \
    [--output_file FILE] [--force_reindex] \
    [--no-streaming]
```

Useful flags:

- `--force_reindex` — wipe `.code_graph*` and re-parse the project.
- `--offset N` + `--limit M` — shard the workload across terminals. Always pair
  with `--output_file` so each shard writes its own CSV.
- `--output_file FILE` — override the default `oracle_preds_qwen3-coder-next.csv`.

### Input format

The pipeline reads two CSVs under `<INPUT_DIR>/<project>/infer_input/`:

- `inputs.csv` — at minimum `focal_method`, `test_prefix`; optionally `docstring`.
- `meta_llm.csv` — at minimum `test_name`, `file_path`; optionally `GT_output`.

The two are joined on `test_name` when present, otherwise concatenated by row
order (both files are always produced together so row order is guaranteed to
match).

### Output format

Each run writes one CSV row per processed sample, immediately under a write
lock, so progress survives Ctrl-C / crashes:

| column | meaning |
|--------|---------|
| `test_name` | fully-qualified test method |
| `test_prefix` | the test body up to (but not including) the assertion |
| `file_path` | test source file |
| `assert_pred` | predicted assertion, or the literal `exception` for exception-expecting tests |

## Operational notes

### Resume

`run_inference` (in [run_pipeline.py](run_pipeline.py)) reads the output CSV at
startup and skips any `test_name` already written. So you can interrupt and
re-run the same command — it picks up from where it stopped.

### Quota / rate-limit handling

`is_quota_error()` ([tools/definitions.py](tools/definitions.py)) recognizes
HTTP 429, billing, balance, and credits errors. On the first such error, a
`stop_event` halts new submissions, in-flight work drains, the partial CSV is
preserved, and a `[QUOTA]` message is logged. Non-quota errors are logged but do
**not** stop the batch — the bad sample gets an empty assertion and processing
continues.

### Structured output

`extraction_llm = llm.model_copy(update={'streaming': False}).with_structured_output(...)`.

Streaming **must** be disabled for structured output against many SSE endpoints
— it hangs otherwise. The streaming flag for the main agent LLM is controlled
by `--streaming` / `STREAMING`.

### Java-only post-processing

After a Java run, `merge_test_prefix_from_source()` replaces the `test_prefix`
column in the output CSV with the version from the TOGA artifact at
`toga-reflect/artifact/RQ2/toga-model-inputs-outputs/<project>/toga_output/oracle_preds.csv`.
This aligns the test prefixes with the format expected by the RQ1 / RQ2
evaluation harness.

## Directory layout

```
src/solution/
├── run_pipeline.py        # CLI entry point + parallel orchestration
├── graph.py               # build_graph() — wires the 4 nodes into a StateGraph
├── state.py               # AssertionState TypedDict, MAX_AGENT_STEPS
├── schemas.py             # Pydantic schemas for structured LLM outputs
├── code_graph.py          # CodeGraph — Kùzu DB + caches + search APIs
├── lang_config.py         # per-language config used by prompts
│
├── agents/                # one file per node
├── prompts/               # one file per node, .format(**lang_cfg)
├── tools/                 # search_relevant_code + create_tools() closure
├── parser/                # tree-sitter wrapper + Java/Python/JS queries
├── helpers/               # db_utils (schema/queries), build_utils (GraphBuilder),
│                          # assertion_utils (post-process), parsing_utils
│
└── demo/                  # demo scripts used in thesis defense
    ├── part1_build_graph.py   # step-by-step CST → entities → graph
    ├── part2_query_flow.py    # stream events, dump JSON + standalone HTML
    └── trace_viewer.html      # offline UI for one sample's pipeline trace
```

For a step-by-step walkthrough of the pipeline (useful for demo / debugging),
see [demo/README.md](demo/README.md).
