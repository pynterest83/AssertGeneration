# AssertGen

Automatic test assertion generation using a multi-agent LLM pipeline. Supports Java (JUnit), Python (pytest), and JavaScript (Jest).

## Requirements

- VS Code 1.85.0+
- Python 3.8+ available on `PATH` (or set `assertgen.pythonPath` in settings)
- API key for an OpenAI-compatible LLM endpoint

## Install from VSIX

```
code --install-extension assertgen-0.2.2.vsix
```

On first run the extension will prompt to install Python dependencies from `backend/requirements.txt` (~2–3 minutes).

## Configure

Open the AssertGen sidebar and fill in:

- API endpoint (default: `https://api.openai.com/v1`)
- Model name (default: `gpt-4o-mini`)
- API key
- Max workers, temperature

## Usage

Open a project that contains unit tests, then click **Gen Test** in the sidebar. Generated assertions are injected back into the test files and listed in the Test Cases tree view.

## Build from source

```
npm install
npm run package
```

This compiles TypeScript, copies the Python backend into the extension folder, and produces `assertgen-<version>.vsix`.
