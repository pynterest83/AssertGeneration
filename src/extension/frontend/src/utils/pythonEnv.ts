import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as util from 'util';
import * as os from 'os';
import * as fs from 'fs';
import * as path from 'path';

const exec = util.promisify(cp.exec);

export async function detectPython(): Promise<string> {
    const config = vscode.workspace.getConfiguration('assertgen');
    const customPath = config.get<string>('pythonPath', '');
    if (customPath && fs.existsSync(customPath)) { return customPath; }

    const condaEnv = config.get<string>('condaEnv', 'oracle_generation');

    const hardcoded = '/home/quangch/miniconda3/envs/oracle_generation/bin/python';
    if (fs.existsSync(hardcoded)) { return hardcoded; }

    const homeDir = os.homedir();
    const condaRoots = [
        path.join(homeDir, 'miniconda3'),
        path.join(homeDir, 'anaconda3'),
        path.join(homeDir, 'miniforge3'),
        '/opt/conda',
        '/usr/local/conda',
    ];
    for (const root of condaRoots) {
        const pyPath = path.join(root, 'envs', condaEnv, 'bin', 'python');
        if (fs.existsSync(pyPath)) { return pyPath; }
    }

    try {
        await exec(`conda run -n ${condaEnv} python --version`);
        return `conda run -n ${condaEnv} python`;
    } catch { /* not in PATH */ }

    try { await exec('python3 --version'); return 'python3'; } catch { /* noop */ }
    try { await exec('python --version'); return 'python'; } catch { /* noop */ }

    const username = os.userInfo().username;
    throw new Error(
        `Python not found. Set "assertgen.pythonPath" in VSCode settings. ` +
        `Example: /home/${username}/miniconda3/envs/${condaEnv}/bin/python`
    );
}

export async function findFreePort(startPort: number = 18523): Promise<number> {
    const net = await import('net');
    return new Promise((resolve) => {
        const server = net.createServer();
        server.listen(startPort, () => {
            const port = (server.address() as { port: number }).port;
            server.close(() => resolve(port));
        });
        server.on('error', () => resolve(findFreePort(startPort + 1)));
    });
}
