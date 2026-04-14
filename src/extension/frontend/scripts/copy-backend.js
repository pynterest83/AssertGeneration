const fs = require('fs');
const path = require('path');

const SOURCE = path.resolve(__dirname, '..', '..', 'backend');
const DEST = path.resolve(__dirname, '..', 'backend');

const IGNORED = new Set([
    '__pycache__',
    '.venv',
    'venv',
    '.code_graph',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    'node_modules',
    '.git',
]);

function shouldSkip(name) {
    if (IGNORED.has(name)) return true;
    if (name.endsWith('.pyc')) return true;
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
