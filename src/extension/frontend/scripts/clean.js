const fs = require('fs');
const path = require('path');

const TARGETS = ['out', 'backend', '*.vsix'];

function cleanPath(target) {
    const abs = path.resolve(__dirname, '..', target);
    if (target.includes('*')) {
        const dir = path.dirname(abs);
        const pattern = path.basename(target);
        if (!fs.existsSync(dir)) return;
        const ext = pattern.replace('*', '');
        for (const entry of fs.readdirSync(dir)) {
            if (entry.endsWith(ext)) {
                fs.rmSync(path.join(dir, entry), { force: true });
                console.log(`Removed ${entry}`);
            }
        }
        return;
    }
    if (fs.existsSync(abs)) {
        fs.rmSync(abs, { recursive: true, force: true });
        console.log(`Removed ${target}`);
    }
}

TARGETS.forEach(cleanPath);
