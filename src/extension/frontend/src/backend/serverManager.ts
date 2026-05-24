import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
import * as http from 'http';
import { detectPython, findFreePort } from '../utils/pythonEnv';
import { getConfig } from '../utils/config';

// Where the unpacked conda environment lives across extension reloads.
function packedEnvDir(): string {
    return path.join(os.homedir(), '.assertgen', 'env');
}

// Map platform/arch → tarball filename produced by conda-pack.
function packedEnvTarballName(): string | null {
    const platform = process.platform;
    const arch = process.arch;
    if (platform === 'linux' && arch === 'x64') {
        return 'assertgen-runtime-linux-x86_64.tar.gz';
    }
    return null;  // Other platforms must use detectPython()/pip flow.
}

const DEFAULT_PORT = 18523;

export class ServerManager {
    private process: cp.ChildProcess | null = null;
    private port: number = DEFAULT_PORT;
    private backendPath: string;
    private outputChannel: vscode.OutputChannel;

    constructor(extensionPath: string, outputChannel: vscode.OutputChannel) {
        this.backendPath = resolveBackendPath(extensionPath);
        this.outputChannel = outputChannel;
    }

    async start(): Promise<void> {
        // If already managing a process, nothing to do
        if (this.process) { return; }

        // Check if a server is already running on the default port
        if (await this.checkHealth(DEFAULT_PORT)) {
            this.port = DEFAULT_PORT;
            this.outputChannel.appendLine(`AssertGen: reusing existing backend on port ${this.port}`);
            return;
        }

        // Find a free port and spawn a new server
        this.port = await findFreePort(DEFAULT_PORT);

        // Prefer the bundled conda-packed runtime if available; otherwise fall
        // back to detectPython() + pip install flow.
        const packedPython = await this.tryUsePackedEnv();
        const python = packedPython ?? await detectPython();
        const config = getConfig();

        if (!packedPython) {
            await this.ensureDependencies(python);
        }

        const env: NodeJS.ProcessEnv = {
            ...process.env,
            API_ENDPOINT: config.apiEndpoint,
            MODEL_NAME: config.modelName,
        };

        const uvicornArgs = ['-m', 'uvicorn', 'server:app', '--host', '127.0.0.1', '--port', String(this.port)];
        const spawnCmd = python.includes('conda run') ? python.split(' ')[0] : python;
        const spawnArgs = python.includes('conda run')
            ? [...python.split(' ').slice(1), ...uvicornArgs]
            : uvicornArgs;

        this.outputChannel.appendLine(`Starting backend: ${spawnCmd} ${spawnArgs.join(' ')}`);
        this.outputChannel.appendLine(`Working dir: ${this.backendPath}`);

        this.process = cp.spawn(spawnCmd, spawnArgs, {
            cwd: this.backendPath,
            env,
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        this.process.stdout?.on('data', (d: Buffer) => this.outputChannel.append(d.toString()));
        this.process.stderr?.on('data', (d: Buffer) => this.outputChannel.append(d.toString()));
        this.process.on('exit', (code) => {
            this.outputChannel.appendLine(`Backend exited with code ${code}`);
            this.process = null;
        });

        await this.waitForHealth(30000);
        this.outputChannel.appendLine(`AssertGen backend started on port ${this.port}`);
    }

    /** Returns true if a healthy server is already responding on the given port. */
    private checkHealth(port: number): Promise<boolean> {
        return new Promise((resolve) => {
            const req = http.get(`http://127.0.0.1:${port}/health`, (res) => {
                resolve(res.statusCode === 200);
                res.resume();
            });
            req.setTimeout(1500, () => { req.destroy(); resolve(false); });
            req.on('error', () => resolve(false));
        });
    }

    private waitForHealth(timeout: number): Promise<void> {
        return new Promise((resolve, reject) => {
            const start = Date.now();
            const check = () => {
                const req = http.get(`http://127.0.0.1:${this.port}/health`, (res) => {
                    if (res.statusCode === 200) { res.resume(); resolve(); return; }
                    res.resume();
                    retry();
                });
                req.on('error', retry);
            };
            const retry = () => {
                if (Date.now() - start > timeout) {
                    reject(new Error(`Backend did not start within ${timeout / 1000}s`));
                    return;
                }
                setTimeout(check, 2000);
            };
            check();
        });
    }

    /**
     * If a bundled conda-packed tarball matching this platform exists, ensure
     * the env is extracted under ~/.assertgen/env and return its python path.
     * Returns null when no tarball is shipped (e.g. running on macOS/Windows
     * or in dev mode without packed env) — caller should fall back.
     */
    private async tryUsePackedEnv(): Promise<string | null> {
        const tarballName = packedEnvTarballName();
        if (!tarballName) { return null; }

        const envDir = packedEnvDir();
        const envPython = path.join(envDir, 'bin', 'python');

        // Already extracted from a previous session
        if (fs.existsSync(envPython)) {
            this.outputChannel.appendLine(`Using packed env at ${envDir}`);
            return envPython;
        }

        const tarballPath = path.join(this.backendPath, tarballName);
        if (!fs.existsSync(tarballPath)) {
            // No bundled tarball — caller will fall back to detectPython().
            return null;
        }

        const choice = await vscode.window.showInformationMessage(
            'AssertGen ships its own Python runtime. Extract now (~150 MB → ~500 MB on disk)?',
            'Extract', 'Skip (use system Python)'
        );
        if (choice !== 'Extract') { return null; }

        try {
            await this.extractPackedEnv(tarballPath, envDir);
        } catch (e) {
            this.outputChannel.appendLine(`Failed to extract packed env: ${e}`);
            vscode.window.showWarningMessage(
                `AssertGen could not extract bundled runtime: ${e}. Falling back to system Python.`
            );
            return null;
        }
        return envPython;
    }

    private async extractPackedEnv(tarballPath: string, envDir: string): Promise<void> {
        const { execFileSync } = require('child_process') as typeof import('child_process');
        fs.mkdirSync(envDir, { recursive: true });

        await vscode.window.withProgress(
            { location: vscode.ProgressLocation.Notification,
              title: 'Extracting AssertGen runtime...', cancellable: false },
            async (progress) => {
                progress.report({ message: 'Unpacking tarball' });
                execFileSync('tar', ['-xzf', tarballPath, '-C', envDir],
                             { stdio: 'inherit' });

                // Run conda-unpack with the env's own python directly (don't rely on
                // the script's `#!/usr/bin/env python` shebang — VSCode-spawned
                // processes often have a PATH without an unqualified `python`).
                progress.report({ message: 'Fixing paths (conda-unpack)' });
                const condaUnpack = path.join(envDir, 'bin', 'conda-unpack');
                const envPython = path.join(envDir, 'bin', 'python');
                if (fs.existsSync(condaUnpack) && fs.existsSync(envPython)) {
                    execFileSync(envPython, [condaUnpack], { stdio: 'inherit' });
                }
            }
        );
        this.outputChannel.appendLine(`Extracted packed env to ${envDir}`);
    }

    private async ensureDependencies(python: string): Promise<void> {
        const reqPath = path.join(this.backendPath, 'requirements.txt');
        try {
            const { execSync } = require('child_process') as typeof import('child_process');
            execSync(`${python} -c "import fastapi"`, { stdio: 'ignore' });
        } catch {
            const install = await vscode.window.showInformationMessage(
                'AssertGen requires Python packages. Install now?', 'Install', 'Cancel'
            );
            if (install === 'Install') {
                await vscode.window.withProgress(
                    { location: vscode.ProgressLocation.Notification, title: 'Installing AssertGen dependencies...' },
                    async () => {
                        const { execSync } = require('child_process') as typeof import('child_process');
                        execSync(`${python} -m pip install -r "${reqPath}"`, { cwd: this.backendPath });
                    }
                );
            }
        }
    }

    getBaseUrl(): string { return `http://127.0.0.1:${this.port}`; }

    async isRunning(): Promise<boolean> {
        return this.process !== null || this.checkHealth(this.port);
    }

    stop(): void {
        if (this.process) {
            this.process.kill('SIGTERM');
            setTimeout(() => { if (this.process) { this.process.kill('SIGKILL'); } }, 5000);
            this.process = null;
        }
    }
}

function resolveBackendPath(extensionPath: string): string {
    const bundled = path.join(extensionPath, 'backend');
    if (fs.existsSync(path.join(bundled, 'server.py'))) {
        return bundled;
    }
    return path.join(extensionPath, '..', 'backend');
}
