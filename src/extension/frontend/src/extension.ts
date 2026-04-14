import * as vscode from 'vscode';
import { ServerManager } from './backend/serverManager';
import { ApiClient } from './backend/apiClient';
import { ProgressListener } from './backend/progressListener';
import { SidebarProvider } from './providers/sidebarProvider';
import { GraphPanelProvider } from './providers/graphPanelProvider';
import { TestCaseTreeProvider } from './providers/testCaseTreeProvider';
import { genTestCommand } from './commands/genTest';
import { getConfig, detectLanguage } from './utils/config';

export async function activate(context: vscode.ExtensionContext): Promise<void> {
    const outputChannel = vscode.window.createOutputChannel('AssertGen');
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBar.text = '$(beaker) AssertGen: Ready';
    statusBar.command = 'assertgen.genTest';
    statusBar.show();

    const serverManager = new ServerManager(context.extensionPath, outputChannel);
    const apiClient = new ApiClient(() => serverManager.getBaseUrl());
    const progressListener = new ProgressListener();

    const treeProvider = new TestCaseTreeProvider();
    const graphProvider = new GraphPanelProvider(apiClient, context.extensionPath);

    const sidebarProvider = new SidebarProvider(context.extensionPath, async () => {
        if (!await serverManager.isRunning()) { return; }
        const wf = vscode.workspace.workspaceFolders;
        if (!wf?.length) { return; }
        const projectPath = wf[0].uri.fsPath;

        try {
            const graphStatus = await apiClient.getGraphStatus(projectPath);
            if (graphStatus.built) { sidebarProvider.sendGraphReady(); }
        } catch { /* server not ready */ }

        try {
            const pipelineStatus = await apiClient.getPipelineStatus(projectPath);
            if (pipelineStatus.has_results) {
                sidebarProvider.sendRestored(pipelineStatus.test_count);
            }
        } catch { /* ignore */ }
    });

    // Register sidebar webview
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(SidebarProvider.viewId, sidebarProvider, {
            webviewOptions: { retainContextWhenHidden: true }
        })
    );

    // Register tree view
    context.subscriptions.push(
        vscode.window.createTreeView('assertgen-testcases', {
            treeDataProvider: treeProvider,
            showCollapseAll: true,
        })
    );

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('assertgen.genTest', () =>
            genTestCommand(
                serverManager,
                apiClient,
                progressListener,
                sidebarProvider,
                treeProvider,
                statusBar,
                context
            )
        ),

        vscode.commands.registerCommand('assertgen.showGraph', async () => {
            const wf = vscode.workspace.workspaceFolders;
            if (!wf || wf.length === 0) {
                vscode.window.showErrorMessage('No workspace folder open.');
                return;
            }
            const projectPath = wf[0].uri.fsPath;
            const config = getConfig();
            const lang = config.language === 'auto' ? detectLanguage(projectPath) : config.language;
            if (!await serverManager.isRunning()) {
                statusBar.text = '$(loading~spin) AssertGen: Starting server...';
                try {
                    await serverManager.start();
                    statusBar.text = '$(beaker) AssertGen: Ready';
                } catch (e) {
                    vscode.window.showErrorMessage(`Failed to start AssertGen backend: ${e}`);
                    return;
                }
            }
            await graphProvider.show(projectPath, lang);
        }),

        vscode.commands.registerCommand('assertgen.configure', () => {
            vscode.commands.executeCommand('workbench.action.openSettings', 'assertgen');
        }),

        vscode.commands.registerCommand('assertgen.setApiKey', async (keyArg?: string) => {
            const key = keyArg || await vscode.window.showInputBox({
                prompt: 'Enter your LLM API key',
                password: true,
                placeHolder: 'sk-...',
            });
            if (key) {
                await context.secrets.store('assertgen.apiKey', key);
                vscode.window.showInformationMessage('API key saved securely.');
            }
        }),
    );

    // Start backend server in background; after it's ready, push project status to sidebar
    serverManager.start().then(async () => {
        const wf = vscode.workspace.workspaceFolders;
        if (!wf?.length) { return; }
        const projectPath = wf[0].uri.fsPath;
        try {
            const graphStatus = await apiClient.getGraphStatus(projectPath);
            if (graphStatus.built) { sidebarProvider.sendGraphReady(); }
        } catch { /* ignore */ }
        try {
            const pipelineStatus = await apiClient.getPipelineStatus(projectPath);
            if (pipelineStatus.has_results) { sidebarProvider.sendRestored(pipelineStatus.test_count); }
        } catch { /* ignore */ }
    }).catch((e: unknown) => {
        outputChannel.appendLine(`Backend startup failed: ${e}. Will retry on first use.`);
    });

    context.subscriptions.push(statusBar, outputChannel, {
        dispose: () => {
            progressListener.stop();
            serverManager.stop();
        },
    });
}

export function deactivate(): void {}
