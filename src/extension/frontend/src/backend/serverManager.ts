import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as http from 'http';
import { detectPython, findFreePort } from '../utils/pythonEnv';
import { getConfig } from '../utils/config';

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
        const python = await detectPython();
        const config = getConfig();

        await this.ensureDependencies(python);

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
