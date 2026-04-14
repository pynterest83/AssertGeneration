import * as vscode from 'vscode';
import { ApiClient } from '../backend/apiClient';
import { ServerManager } from '../backend/serverManager';
import { ProgressListener } from '../backend/progressListener';
import { SidebarProvider } from '../providers/sidebarProvider';
import { TestCaseTreeProvider } from '../providers/testCaseTreeProvider';
import { getConfig, detectLanguage } from '../utils/config';
import { ProgressEvent } from '../types/api';

export async function genTestCommand(
    serverManager: ServerManager,
    apiClient: ApiClient,
    progressListener: ProgressListener,
    sidebarProvider: SidebarProvider,
    treeProvider: TestCaseTreeProvider,
    statusBar: vscode.StatusBarItem,
    context: vscode.ExtensionContext,
): Promise<void> {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders || workspaceFolders.length === 0) {
        vscode.window.showErrorMessage('No workspace folder open. Open a project folder first.');
        return;
    }

    const projectPath = workspaceFolders[0].uri.fsPath;
    const config = getConfig();

    // Get API key from secret storage; fall back to empty for local endpoints
    const storedApiKey = await context.secrets.get('assertgen.apiKey');
    const apiKey = storedApiKey || (config.apiEndpoint.includes('localhost') ? 'EMPTY' : '');

    // Detect language
    let language = config.language;
    if (language === 'auto') { language = detectLanguage(projectPath); }

    // Ensure server is running
    if (!await serverManager.isRunning()) {
        statusBar.text = '$(loading~spin) AssertGen: Starting server...';
        try {
            await serverManager.start();
        } catch (e) {
            vscode.window.showErrorMessage(`Failed to start AssertGen backend: ${e}`);
            return;
        }
    }

    // Update UI state
    statusBar.text = '$(loading~spin) AssertGen: Running...';
    sidebarProvider.reset();
    treeProvider.clear();

    // Start listening to SSE progress
    progressListener.stop(); // stop any previous connection
    progressListener.listen(serverManager.getBaseUrl(), (event: ProgressEvent) => {
        sidebarProvider.sendProgress(event);

        if (event.type === 'graph_building_complete') {
            sidebarProvider.sendGraphReady();
        } else if (event.type === 'inference_progress') {
            statusBar.text = `$(loading~spin) AssertGen: ${event.current}/${event.total}`;
        } else if (event.type === 'pipeline_complete') {
            statusBar.text = `$(check) AssertGen: ${event.total_tests} assertions injected`;
            sidebarProvider.sendStatus('done');
            sidebarProvider.sendGraphReady(); // keep showing if missed earlier
            // Open modified test files
            for (const f of event.injected_files || []) {
                vscode.workspace.openTextDocument(f).then(doc =>
                    vscode.window.showTextDocument(doc, vscode.ViewColumn.One)
                );
            }
            progressListener.stop();
        } else if (event.type === 'error') {
            statusBar.text = '$(error) AssertGen: Failed';
            sidebarProvider.sendStatus('idle');
            vscode.window.showErrorMessage(`AssertGen error: ${event.message}`);
            progressListener.stop();
        }
    });

    // Run the pipeline
    try {
        await apiClient.runPipeline({
            project_path: projectPath,
            language,
            api_endpoint: config.apiEndpoint,
            model_name: config.modelName,
            api_key: apiKey || 'EMPTY',
            max_workers: config.maxWorkers,
            temperature: config.temperature,
            force_reindex: config.forceReindex,
        });
    } catch (e) {
        progressListener.stop();
        statusBar.text = '$(error) AssertGen: Failed';
        sidebarProvider.sendStatus('idle');
        vscode.window.showErrorMessage(`AssertGen pipeline failed: ${e}`);
    }
}
