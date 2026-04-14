import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { ProgressEvent } from '../types/api';

export class SidebarProvider implements vscode.WebviewViewProvider {
    public static readonly viewId = 'assertgen-sidebar';
    private view?: vscode.WebviewView;
    private extensionPath: string;
    private onViewReady?: () => void;

    constructor(extensionPath: string, onViewReady?: () => void) {
        this.extensionPath = extensionPath;
        this.onViewReady = onViewReady;
    }

    resolveWebviewView(webviewView: vscode.WebviewView): void {
        this.view = webviewView;
        webviewView.webview.options = { enableScripts: true };
        webviewView.webview.html = this.getHtml(webviewView.webview);
        // Notify after a short delay to let the webview JS initialize
        setTimeout(() => this.onViewReady?.(), 500);
        webviewView.webview.onDidReceiveMessage((msg: {
            command: string;
            apiEndpoint?: string;
            modelName?: string;
            maxWorkers?: number;
            temperature?: number;
            apiKey?: string;
        }) => {
            if (msg.command === 'genTest') {
                vscode.commands.executeCommand('assertgen.genTest');
            } else if (msg.command === 'showGraph') {
                vscode.commands.executeCommand('assertgen.showGraph');
            } else if (msg.command === 'saveConfig') {
                const cfg = vscode.workspace.getConfiguration('assertgen');
                if (msg.apiEndpoint !== undefined) cfg.update('apiEndpoint', msg.apiEndpoint, true);
                if (msg.modelName !== undefined) cfg.update('modelName', msg.modelName, true);
                if (msg.maxWorkers !== undefined) cfg.update('maxWorkers', msg.maxWorkers, true);
                if (msg.temperature !== undefined) cfg.update('temperature', msg.temperature, true);
                if (msg.apiKey) {
                    vscode.commands.executeCommand('assertgen.setApiKey', msg.apiKey);
                }
            }
        });
    }

    sendProgress(event: ProgressEvent): void {
        this.view?.webview.postMessage({ type: 'progress', event });
    }

    sendStatus(status: string, detail?: string): void {
        this.view?.webview.postMessage({ type: 'status', status, detail });
    }

    sendGraphStats(nodeCount: number, edgeCount: number): void {
        this.view?.webview.postMessage({ type: 'graphStats', nodeCount, edgeCount });
    }

    sendGraphReady(): void {
        this.view?.webview.postMessage({ type: 'graph_building_complete' });
    }

    sendRestored(testCount: number): void {
        this.view?.webview.postMessage({ type: 'restored', testCount });
    }

    reset(): void {
        this.view?.webview.postMessage({ type: 'reset' });
    }

    private getHtml(webview: vscode.Webview): string {
        // Load from media/sidebar/sidebar.html
        const htmlPath = path.join(this.extensionPath, 'media', 'sidebar', 'sidebar.html');
        if (fs.existsSync(htmlPath)) {
            let html = fs.readFileSync(htmlPath, 'utf8');
            const jsUri = webview.asWebviewUri(
                vscode.Uri.file(path.join(this.extensionPath, 'media', 'sidebar', 'sidebar.js'))
            );
            const cssUri = webview.asWebviewUri(
                vscode.Uri.file(path.join(this.extensionPath, 'media', 'sidebar', 'sidebar.css'))
            );
            html = html.replace('{{JS_URI}}', jsUri.toString()).replace('{{CSS_URI}}', cssUri.toString());
            return html;
        }
        return this.getInlineHtml();
    }

    private getInlineHtml(): string {
        return `<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); padding: 12px; color: var(--vscode-foreground); background: var(--vscode-sideBar-background); }
  button { width: 100%; padding: 8px; margin: 4px 0; background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; cursor: pointer; border-radius: 3px; font-size: 13px; }
  button:hover { background: var(--vscode-button-hoverBackground); }
  button.primary { background: var(--vscode-button-background); font-weight: bold; }
  .section { margin-bottom: 16px; }
  .section h3 { font-size: 11px; text-transform: uppercase; color: var(--vscode-descriptionForeground); margin-bottom: 8px; letter-spacing: 0.5px; }
  input, select { width: 100%; padding: 4px 6px; background: var(--vscode-input-background); color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border); border-radius: 3px; font-size: 12px; box-sizing: border-box; margin-bottom: 6px; }
  label { font-size: 11px; color: var(--vscode-descriptionForeground); display: block; margin-bottom: 2px; }
  .progress-bar-container { background: var(--vscode-progressBar-background); height: 4px; border-radius: 2px; margin: 8px 0; overflow: hidden; }
  .progress-bar { height: 100%; background: var(--vscode-button-background); transition: width 0.3s; }
  .step-label { font-size: 11px; color: var(--vscode-descriptionForeground); margin-top: 4px; }
  .agents { display: flex; gap: 4px; margin-top: 8px; }
  .agent-box { flex: 1; text-align: center; padding: 4px; font-size: 10px; border-radius: 3px; background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); opacity: 0.5; transition: opacity 0.3s; }
  .agent-box.active { opacity: 1; background: var(--vscode-button-background); color: var(--vscode-button-foreground); }
  .status-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-bottom: 8px; }
  .status-idle { background: var(--vscode-badge-background); }
  .status-running { background: var(--vscode-statusBarItem-warningBackground, #cc6633); }
  .status-done { background: var(--vscode-statusBarItem-debuggingBackground, #336633); }
  .graph-info { font-size: 11px; color: var(--vscode-descriptionForeground); }
</style>
</head><body>
<div class="section">
  <h3>Actions</h3>
  <span class="status-badge status-idle" id="statusBadge">Idle</span>
  <button class="primary" onclick="genTest()">&#9879; Gen Test</button>
  <button id="showGraphBtn" onclick="showGraph()" style="display:none">&#128302; Show Code Graph</button>
</div>

<div class="section" id="progressSection" style="display:none">
  <h3>Progress</h3>
  <div class="step-label" id="stepLabel">Initializing...</div>
  <div class="progress-bar-container"><div class="progress-bar" id="progressBar" style="width:0%"></div></div>
  <div class="agents">
    <div class="agent-box" id="agent-ec">Classifier</div>
    <div class="agent-box" id="agent-ca">Analyzer</div>
    <div class="agent-box" id="agent-sp">Predictor</div>
    <div class="agent-box" id="agent-ag">Generator</div>
  </div>
</div>

<div class="section" id="graphSection" style="display:none">
  <h3>Code Graph</h3>
  <div class="graph-info" id="graphInfo"></div>
</div>

<div class="section">
  <h3>Configuration</h3>
  <label>API Endpoint</label>
  <input type="text" id="apiEndpoint" placeholder="https://api.openai.com/v1">
  <label>Model Name</label>
  <input type="text" id="modelName" placeholder="gpt-4o-mini">
  <label>API Key</label>
  <input type="password" id="apiKey" placeholder="sk-...">
  <label>Max Workers</label>
  <input type="number" id="maxWorkers" value="8" min="1" max="32">
  <label>Temperature</label>
  <input type="number" id="temperature" value="0.0" step="0.1" min="0" max="1">
  <button onclick="saveConfig()">Save Config</button>
</div>

<script>
const vscode = acquireVsCodeApi();

function genTest() { vscode.postMessage({ command: 'genTest' }); }
function showGraph() { vscode.postMessage({ command: 'showGraph' }); }
function saveConfig() {
  vscode.postMessage({
    command: 'saveConfig',
    apiEndpoint: document.getElementById('apiEndpoint').value,
    modelName: document.getElementById('modelName').value,
    apiKey: document.getElementById('apiKey').value,
    maxWorkers: parseInt(document.getElementById('maxWorkers').value),
    temperature: parseFloat(document.getElementById('temperature').value),
  });
}

const agentMap = {
  exception_classifier: 'agent-ec',
  code_analyzer: 'agent-ca',
  state_predictor: 'agent-sp',
  assertion_generator: 'agent-ag'
};

window.addEventListener('message', function(e) {
  var msg = e.data;
  if (msg.type === 'progress') {
    var ev = msg.event;
    var progress = document.getElementById('progressSection');
    progress.style.display = '';

    if (ev.type === 'extraction') {
      setStep('Extracting test cases... ' + ev.current + '/' + ev.total, ev.total > 0 ? (ev.current / ev.total) * 25 : 0);
    } else if (ev.type === 'graph_building') {
      setStep('Building code graph... ' + ev.current + '/' + ev.total, ev.total > 0 ? 25 + (ev.current / ev.total) * 25 : 25);
    } else if (ev.type === 'inference_progress') {
      setStep('Generating assertions... ' + ev.current + '/' + ev.total, ev.total > 0 ? 50 + (ev.current / ev.total) * 40 : 50);
    } else if (ev.type === 'inference') {
      Object.values(agentMap).forEach(function(id) { var el = document.getElementById(id); if (el) el.classList.remove('active'); });
      var agentId = agentMap[ev.agent];
      if (agentId) { var el = document.getElementById(agentId); if (el) el.classList.add('active'); }
    } else if (ev.type === 'injection') {
      setStep('Injecting into ' + ev.file + '...', 95);
    } else if (ev.type === 'pipeline_complete') {
      setStep('Done! ' + ev.total_tests + ' assertions injected.', 100);
      setStatus('done', 'Done');
      Object.values(agentMap).forEach(function(id) { var el = document.getElementById(id); if (el) el.classList.remove('active'); });
      document.getElementById('showGraphBtn').style.display = '';
    } else if (ev.type === 'error') {
      setStep('Error: ' + ev.message, 0);
      setStatus('idle', 'Error');
    }
  } else if (msg.type === 'status') {
    var label = msg.status === 'running' ? 'Running...' : msg.status === 'done' ? 'Done' : 'Idle';
    setStatus(msg.status, label);
    if (msg.status === 'running') document.getElementById('progressSection').style.display = '';
  } else if (msg.type === 'graphStats') {
    document.getElementById('graphSection').style.display = '';
    document.getElementById('graphInfo').textContent = msg.nodeCount + ' nodes \u00b7 ' + msg.edgeCount + ' edges';
    document.getElementById('showGraphBtn').style.display = '';
  } else if (msg.type === 'graph_building_complete') {
    document.getElementById('showGraphBtn').style.display = '';
  } else if (msg.type === 'restored') {
    document.getElementById('progressSection').style.display = '';
    setStep('Previous run: ' + msg.testCount + ' assertions available.', 100);
    setStatus('done', 'Done');
    document.getElementById('showGraphBtn').style.display = '';
  } else if (msg.type === 'reset') {
    setStep('Starting...', 0);
    setStatus('running', 'Running...');
    Object.values(agentMap).forEach(function(id) { var el = document.getElementById(id); if (el) el.classList.remove('active'); });
    document.getElementById('progressSection').style.display = '';
  }
});

function setStep(text, pct) {
  document.getElementById('stepLabel').textContent = text;
  document.getElementById('progressBar').style.width = pct + '%';
}
function setStatus(cls, text) {
  var b = document.getElementById('statusBadge');
  b.className = 'status-badge status-' + cls;
  b.textContent = text;
}
</script>
</body></html>`;
    }
}
