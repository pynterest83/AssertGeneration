# `src/extension/` — VS Code Extension

A VS Code extension that drives the assertion-generation pipeline from
[`src/solution/`](../solution/) via a local FastAPI backend, with a sidebar UI
for configuration, a tree view for generated test cases, and an interactive
graph panel rendered with `vis-network`.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  VS Code (Electron, Node runtime)                                │
│                                                                  │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│   │  Sidebar     │   │  Test Cases  │   │  Graph Panel     │    │
│   │  (webview)   │   │  (TreeView)  │   │  (vis-network)   │    │
│   └──────┬───────┘   └──────┬───────┘   └──────────┬───────┘    │
│          └───────────┬──────┴──────────────────────┘            │
│                     ▼                                            │
│          frontend/src/extension.ts                               │
│          frontend/src/backend/serverManager.ts                   │
│                     │ spawns + health-checks                     │
└─────────────────────┼────────────────────────────────────────────┘
                      │ HTTP (localhost:18523 by default)
┌─────────────────────┼────────────────────────────────────────────┐
│  Python subprocess  ▼                                            │
│                                                                  │
│   backend/server.py     ── FastAPI endpoints                     │
│   backend/test_extractor.py                                      │
│   backend/pipeline_runner.py ── wraps src/solution               │
│   backend/graph_export.py    ── KùzuDB → vis-network JSON        │
│   backend/injectors/         ── write assertions to test files   │
│                                                                  │
│         imports                                                  │
│            ▼                                                     │
│   src/solution/   (the 4-agent LangGraph pipeline)               │
└──────────────────────────────────────────────────────────────────┘
```

The extension does not embed Python. It spawns a Python process running the
FastAPI server, then talks to it over HTTP. Two Python sources are supported,
resolved at startup:

1. **Bundled runtime** — `backend/assertgen-runtime-linux-x86_64.tar.gz`, a
   `conda-pack`-produced relocatable Python env. Extracted to extension global
   storage on first launch (Linux only at present).
2. **System Python** — falls back to the interpreter at the path in the
   `assertgen.pythonPath` setting, or the active conda env named in
   `assertgen.condaEnv` (default: `oracle_generation`).

## Directory layout

```
src/extension/
├── backend/                  # FastAPI + Python integration layer
│   ├── server.py             # endpoints (health, extract, build-graph, run-pipeline, ...)
│   ├── pipeline_runner.py    # wraps src/solution; one CodeGraph per project_path
│   ├── test_extractor.py     # scans project, emits infer_input/{inputs,meta_llm}.csv
│   ├── graph_export.py       # KùzuDB → {nodes, edges} JSON for vis-network
│   ├── progress.py           # SSE queue for streaming progress to the UI
│   ├── injectors/            # python_injector.py + java_injector.py write
│   │                         #   generated assertions back to test source files
│   ├── requirements.txt
│   └── assertgen-runtime-linux-x86_64.tar.gz   # conda-packed Python runtime
│
└── frontend/                 # VS Code extension (TypeScript)
    ├── src/
    │   ├── extension.ts                       # activation: registers commands + views
    │   ├── backend/
    │   │   ├── serverManager.ts               # spawn / health / shutdown FastAPI process
    │   │   ├── apiClient.ts                   # typed HTTP client (axios)
    │   │   └── progressListener.ts            # SSE consumer
    │   ├── providers/
    │   │   ├── sidebarProvider.ts             # main webview (config + Gen Test button)
    │   │   ├── testCaseTreeProvider.ts        # Test Cases tree view
    │   │   └── graphPanelProvider.ts          # vis-network webview
    │   ├── commands/genTest.ts
    │   ├── utils/{config.ts, pythonEnv.ts}
    │   └── types/api.ts
    ├── media/                                 # icons + webview assets
    ├── scripts/{clean.js, copy-backend.js}    # build helpers
    ├── package.json                           # contributes (commands, views, config)
    ├── README.md                              # marketplace-facing README
    └── assertgen-<version>.vsix               # packaged extension
```

## HTTP endpoints

Served at `http://localhost:18523` (configurable via free-port fallback if 18523
is taken). All endpoints expect JSON; `/progress` is Server-Sent Events.

| Method + path | Purpose |
|---------------|---------|
| `GET /health` | liveness probe |
| `POST /extract` | scan a project, generate `infer_input/inputs.csv` + `meta_llm.csv` |
| `POST /build-graph` | run `CodeGraph(project_path, language, force_reindex)` |
| `GET /graph-status?project_path=…` | check whether `.code_graph.complete` exists |
| `GET /graph-data?project_path=…&language=…` | export ≤1000 nodes + edges as `{nodes, edges}` for `vis-network` |
| `POST /run-pipeline` | start the 4-agent pipeline; runs in a background thread |
| `GET /pipeline-status` | current state of the running batch |
| `GET /progress` | SSE stream of `{stage, message, current, total, ...}` events |

Request bodies for the main endpoints:

```jsonc
// POST /extract
{ "project_path": "/path/to/project", "language": "python" }

// POST /build-graph
{ "project_path": "/path/to/project", "language": "java", "force_reindex": false }

// POST /run-pipeline
{
  "project_path": "/path/to/project",
  "language": "java",
  "api_endpoint": "https://api.openai.com/v1",
  "model_name": "gpt-4o-mini",
  "api_key": "sk-…",
  "max_workers": 8,
  "temperature": 0.0
}
```

## Settings (VS Code `settings.json`)

All under the `assertgen.*` namespace. Set via the sidebar UI or directly in
`settings.json`:

| Key | Default | Purpose |
|-----|---------|---------|
| `assertgen.pythonPath` | — | Override Python interpreter (skips conda lookup) |
| `assertgen.condaEnv` | `oracle_generation` | Conda env name to activate if no `pythonPath` |
| `assertgen.apiEndpoint` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `assertgen.modelName` | `gpt-4o-mini` | Model identifier |
| `assertgen.language` | `python` | Default language for new projects |
| `assertgen.maxWorkers` | `8` | Parallel samples in the pipeline |
| `assertgen.temperature` | `0.0` | LLM temperature |
| `assertgen.forceReindex` | `false` | Wipe `.code_graph` on next build |

API keys are stored in VS Code SecretStorage via `assertgen.setApiKey`, never in
`settings.json`.

## Commands

| Command ID | Title | Bound to |
|------------|-------|----------|
| `assertgen.genTest` | Gen Test | Sidebar button + command palette |
| `assertgen.showGraph` | AssertGen: Show Code Graph | Command palette |
| `assertgen.configure` | AssertGen: Configure | Command palette |
| `assertgen.setApiKey` | AssertGen: Set API Key | Command palette |

Triggering `assertgen.genTest` runs the full flow: `extract` → `build-graph` →
`run-pipeline`, streaming progress through SSE to the sidebar.

## Development

### Run the backend standalone

Useful for iterating on the Python side without rebuilding the extension:

```bash
cd src/extension/backend
conda run -n oracle_generation uvicorn server:app \
    --host 127.0.0.1 --port 18523 --reload
```

The server resolves `src/solution/` via the candidates listed at the top of
`server.py`, so it works in dev mode without copying files.

### Run the extension in dev

```bash
cd src/extension/frontend
npm install
npm run watch        # tsc --watch
# F5 in VS Code launches an Extension Development Host
```

In dev mode the extension does *not* extract the conda-packed tarball — it
spawns whichever Python is configured.

### Package a VSIX

```bash
cd src/extension/frontend
npm run package
```

The `npm run package` script:

1. `tsc` — compile TypeScript to `out/`.
2. `scripts/copy-backend.js` — copies `../backend/` into `frontend/backend/`
   so it ships inside the VSIX (the tarball is the largest payload).
3. `vsce package` — produces `assertgen-<version>.vsix`.

### Rebuild the conda-packed runtime

The bundled runtime tarball is produced from the `oracle_generation` conda env:

```bash
conda activate oracle_generation
pip install conda-pack
conda-pack -n oracle_generation \
    -o src/extension/backend/assertgen-runtime-linux-x86_64.tar.gz \
    --force
```

The platform-specific filename is checked by
`frontend/src/backend/serverManager.ts:packedEnvTarballName()`. macOS and
Windows users currently must rely on `pythonPath` / `condaEnv` — there is no
prebuilt runtime for those platforms in the repo.

## Known limits

- **Linux x86_64 only** for the bundled runtime. Other platforms must point
  `pythonPath` or `condaEnv` at a working install.
- **Graph panel caps at 1000 nodes** (`NODE_LIMIT` in `graph_export.py`). Larger
  projects render the centerpiece classes only; the Cypher console in Kùzu
  Explorer is the right tool for exhaustive exploration.
- **Only Java + Python injectors implemented** in `backend/injectors/`. The
  underlying pipeline supports JavaScript, but the extension does not yet write
  Jest assertions back to source.
