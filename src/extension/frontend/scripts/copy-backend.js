const fs = require('fs');
const path = require('path');

const SOURCE = path.resolve(__dirname, '..', '..', 'backend');
const DEST = path.resolve(__dirname, '..', 'backend');
const SOLUTION_SOURCE = path.resolve(__dirname, '..', '..', '..', 'solution');
const SOLUTION_DEST = path.resolve(DEST, '_solution');

const IGNORED = new Set([
    '__pycache__',
    '.venv',
    'venv',
    '.code_graph',
    '.code_graph.complete',
    '.code_graph.wal',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    'node_modules',
    '.git',
    'infer_input',
    'test',
    'tests',
]);

function shouldSkip(name) {
    if (IGNORED.has(name)) return true;
    if (name.endsWith('.pyc')) return true;
    // SECURITY: never bundle .env or any secret/key files
    if (name === '.env' || name.startsWith('.env.')) return true;
    if (/secret|credentials?|api[_-]?key/i.test(name)) return true;
    return false;
}

function copyRecursive(src, dest) {
    const stat = fs.statSync(src);
    if (stat.isDirectory()) {
        fs.mkdirSync(dest, { recursive: true });
        for (const entry of fs.readdirSync(src)) {
            if (shouldSkip(entry)) continue;
            copyRecursive(path.join(src, entry), path.join(dest, entry));
        }
    } else {
        fs.copyFileSync(src, dest);
    }
}

if (!fs.existsSync(SOURCE)) {
    console.error(`Backend source not found: ${SOURCE}`);
    process.exit(1);
}

if (fs.existsSync(DEST)) {
    fs.rmSync(DEST, { recursive: true, force: true });
}

copyRecursive(SOURCE, DEST);
console.log(`Copied backend: ${SOURCE} -> ${DEST}`);

if (fs.existsSync(SOLUTION_SOURCE)) {
    copyRecursive(SOLUTION_SOURCE, SOLUTION_DEST);
    console.log(`Copied solution: ${SOLUTION_SOURCE} -> ${SOLUTION_DEST}`);
} else {
    console.error(`Solution source not found: ${SOLUTION_SOURCE}`);
    process.exit(1);
}

// Bundle vis-network into media/lib/ (node_modules is excluded by .vscodeignore)
const VIS_SRC = path.resolve(__dirname, '..', 'node_modules', 'vis-network',
    'standalone', 'umd', 'vis-network.min.js');
const MEDIA_LIB = path.resolve(__dirname, '..', 'media', 'lib');
const VIS_DEST = path.join(MEDIA_LIB, 'vis-network.min.js');
if (fs.existsSync(VIS_SRC)) {
    fs.mkdirSync(MEDIA_LIB, { recursive: true });
    fs.copyFileSync(VIS_SRC, VIS_DEST);
    console.log(`Copied vis-network: ${VIS_SRC} -> ${VIS_DEST}`);
} else {
    console.error(`vis-network not found: ${VIS_SRC} — run 'npm install' first`);
    process.exit(1);
}
