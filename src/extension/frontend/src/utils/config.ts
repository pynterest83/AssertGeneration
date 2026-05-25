import * as vscode from 'vscode';

export function getConfig() {
    const cfg = vscode.workspace.getConfiguration('assertgen');
    return {
        pythonPath: cfg.get<string>('pythonPath', ''),
        condaEnv: cfg.get<string>('condaEnv', 'oracle_generation'),
        apiEndpoint: cfg.get<string>('apiEndpoint', 'https://api.openai.com/v1'),
        modelName: cfg.get<string>('modelName', 'gpt-4o-mini'),
        language: cfg.get<string>('language', 'auto'),
        maxWorkers: cfg.get<number>('maxWorkers', 8),
        temperature: cfg.get<number>('temperature', 0.0),
        forceReindex: cfg.get<boolean>('forceReindex', false),
        forceReextract: cfg.get<boolean>('forceReextract', false),
    };
}

export function detectLanguage(workspacePath: string): string {
    const fs = require('fs') as typeof import('fs');
    const path = require('path') as typeof import('path');

    // Check for Java files
    const hasJava = fs.existsSync(path.join(workspacePath, 'pom.xml')) ||
                    fs.existsSync(path.join(workspacePath, 'build.gradle'));
    if (hasJava) return 'java';

    // Check for Python files
    const hasPython = fs.existsSync(path.join(workspacePath, 'pyproject.toml')) ||
                      fs.existsSync(path.join(workspacePath, 'setup.py')) ||
                      fs.existsSync(path.join(workspacePath, 'requirements.txt'));
    if (hasPython) return 'python';

    return 'python'; // default
}
