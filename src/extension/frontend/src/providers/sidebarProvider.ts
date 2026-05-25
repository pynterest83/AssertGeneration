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
            language?: string;
            pythonPath?: string;
            condaEnv?: string;
            forceReindex?: boolean;
            forceReextract?: boolean;
        }) => {
            if (msg.command === 'genTest') {
                vscode.commands.executeCommand('assertgen.genTest');
            } else if (msg.command === 'showGraph') {
                vscode.commands.executeCommand('assertgen.showGraph');
            } else if (msg.command === 'loadConfig') {
                this.sendConfig();
            } else if (msg.command === 'saveConfig') {
                const cfg = vscode.workspace.getConfiguration('assertgen');
                if (msg.apiEndpoint !== undefined) cfg.update('apiEndpoint', msg.apiEndpoint, true);
                if (msg.modelName !== undefined) cfg.update('modelName', msg.modelName, true);
                if (msg.maxWorkers !== undefined) cfg.update('maxWorkers', msg.maxWorkers, true);
                if (msg.temperature !== undefined) cfg.update('temperature', msg.temperature, true);
                if (msg.language !== undefined) cfg.update('language', msg.language, true);
                if (msg.pythonPath !== undefined) cfg.update('pythonPath', msg.pythonPath, true);
                if (msg.condaEnv !== undefined) cfg.update('condaEnv', msg.condaEnv, true);
                if (msg.forceReindex !== undefined) cfg.update('forceReindex', msg.forceReindex, true);
                if (msg.forceReextract !== undefined) cfg.update('forceReextract', msg.forceReextract, true);
                if (msg.apiKey) {
                    vscode.commands.executeCommand('assertgen.setApiKey', msg.apiKey);
                }
            }
        });
    }

    private sendConfig(): void {
        const cfg = vscode.workspace.getConfiguration('assertgen');
        this.view?.webview.postMessage({
            type: 'config',
            apiEndpoint: cfg.get<string>('apiEndpoint', 'https://api.openai.com/v1'),
            modelName: cfg.get<string>('modelName', 'gpt-4o-mini'),
            maxWorkers: cfg.get<number>('maxWorkers', 8),
            temperature: cfg.get<number>('temperature', 0.0),
            language: cfg.get<string>('language', 'auto'),
            pythonPath: cfg.get<string>('pythonPath', ''),
            condaEnv: cfg.get<string>('condaEnv', 'oracle_generation'),
            forceReindex: cfg.get<boolean>('forceReindex', false),
            forceReextract: cfg.get<boolean>('forceReextract', false),
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
  .log-panel { max-height: 320px; overflow-y: auto; background: var(--vscode-editor-background); border: 1px solid var(--vscode-panel-border); border-radius: 3px; padding: 6px; font-family: var(--vscode-editor-font-family, monospace); font-size: 11px; line-height: 1.4; }
  .log-entry { margin-bottom: 6px; padding-bottom: 6px; border-bottom: 1px dotted var(--vscode-panel-border); }
  .log-entry:last-child { border-bottom: none; }
  .log-agent { color: var(--vscode-symbolIcon-functionForeground, #dcdcaa); font-weight: bold; }
  .log-tool { color: var(--vscode-symbolIcon-keywordForeground, #569cd6); }
  .log-result { color: var(--vscode-symbolIcon-stringForeground, #ce9178); }
  .log-test { color: var(--vscode-descriptionForeground); font-style: italic; }
  .log-body { white-space: pre-wrap; word-break: break-word; margin-top: 2px; color: var(--vscode-foreground); }
  .log-controls { display: flex; gap: 6px; margin-bottom: 6px; }
  .log-controls button { width: auto; padding: 2px 8px; font-size: 11px; margin: 0; }
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

<div class="section" id="logSection" style="display:none">
  <h3>Agent Log</h3>
  <div class="log-controls">
    <button onclick="clearLog()">Clear</button>
    <button onclick="toggleAutoscroll()" id="autoscrollBtn">Autoscroll: on</button>
  </div>
  <div class="log-panel" id="logPanel"></div>
</div>

<div class="section">
  <h3>Configuration</h3>
  <label>API Endpoint</label>
  <input type="text" id="apiEndpoint" placeholder="https://api.openai.com/v1">
  <label>Model Name</label>
  <input type="text" id="modelName" placeholder="gpt-4o-mini">
  <label>API Key</label>
  <input type="password" id="apiKey" placeholder="sk-...">
  <label>Language</label>
  <select id="language">
    <option value="auto">auto-detect</option>
    <option value="python">python</option>
    <option value="java">java</option>
    <option value="javascript">javascript</option>
  </select>
  <label>Max Workers</label>
  <input type="number" id="maxWorkers" value="8" min="1" max="32">
  <label>Temperature</label>
  <input type="number" id="temperature" value="0.0" step="0.1" min="0" max="1">
  <details style="margin:6px 0 8px">
    <summary style="cursor:pointer;font-size:11px;color:var(--vscode-descriptionForeground)">Advanced</summary>
    <label style="margin-top:6px">Python Path <span style="opacity:0.6">(leave empty to auto-detect)</span></label>
    <input type="text" id="pythonPath" placeholder="/path/to/python">
    <label>Conda Env</label>
    <input type="text" id="condaEnv" placeholder="oracle_generation">
    <label style="display:flex;align-items:center;gap:6px;margin-top:6px">
      <input type="checkbox" id="forceReindex" style="width:auto;margin:0">
      <span>Force re-index code graph</span>
    </label>
    <label style="display:flex;align-items:center;gap:6px;margin-top:6px">
      <input type="checkbox" id="forceReextract" style="width:auto;margin:0">
      <span>Force re-extract test cases</span>
    </label>
  </details>
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
    language: document.getElementById('language').value,
    maxWorkers: parseInt(document.getElementById('maxWorkers').value),
    temperature: parseFloat(document.getElementById('temperature').value),
    pythonPath: document.getElementById('pythonPath').value,
    condaEnv: document.getElementById('condaEnv').value,
    forceReindex: document.getElementById('forceReindex').checked,
    forceReextract: document.getElementById('forceReextract').checked,
  });
}

// Ask extension for current settings as soon as we load
vscode.postMessage({ command: 'loadConfig' });

const agentMap = {
  exception_classifier: 'agent-ec',
  code_analyzer: 'agent-ca',
  state_predictor: 'agent-sp',
  assertion_generator: 'agent-ag'
};
const agentLabel = {
  exception_classifier: 'ExceptionClassifier',
  code_analyzer: 'CodeAnalyzer',
  state_predictor: 'StatePredictor',
  assertion_generator: 'AssertionGenerator'
};

var autoscroll = true;
function clearLog() {
  var p = document.getElementById('logPanel');
  if (p) p.innerHTML = '';
}
function toggleAutoscroll() {
  autoscroll = !autoscroll;
  var b = document.getElementById('autoscrollBtn');
  if (b) b.textContent = 'Autoscroll: ' + (autoscroll ? 'on' : 'off');
}
function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function appendLog(html) {
  var sec = document.getElementById('logSection');
  if (sec) sec.style.display = '';
  var p = document.getElementById('logPanel');
  if (!p) return;
  var div = document.createElement('div');
  div.className = 'log-entry';
  div.innerHTML = html;
  p.appendChild(div);
  if (autoscroll) p.scrollTop = p.scrollHeight;
}

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
      var label = agentLabel[ev.agent] || ev.agent;
      var test = ev.test_name ? ' <span class="log-test">' + escapeHtml(ev.test_name) + '</span>' : '';
      appendLog('<span class="log-agent">&#9655; ' + escapeHtml(label) + '</span>' + test);
    } else if (ev.type === 'tool_call') {
      var args = [];
      if (ev.class_name) args.push('class=' + ev.class_name);
      if (ev.method_name) args.push('method=' + ev.method_name);
      if (ev.include_callees) args.push('+callees');
      appendLog('<span class="log-tool">&#128269; ' + escapeHtml(ev.tool) + '(' + escapeHtml(args.join(', ')) + ')</span>');
    } else if (ev.type === 'tool_result') {
      var tag = ev.external ? ' (external)' : (' &middot; ' + ev.result_count + ' result' + (ev.result_count === 1 ? '' : 's'));
      var body = ev.preview ? '<div class="log-body">' + escapeHtml(ev.preview) + '</div>' : '';
      appendLog('<span class="log-result">&#8629; result' + tag + '</span>' + body);
    } else if (ev.type === 'agent_output') {
      var label2 = agentLabel[ev.agent] || ev.agent;
      var body2 = '';
      if (ev.agent === 'exception_classifier') {
        body2 = '<div class="log-body">is_exception = ' + (ev.is_exception ? 'true' : 'false') + '</div>';
      } else if (ev.analysis) {
        body2 = '<div class="log-body">' + escapeHtml(ev.analysis) + '</div>';
      } else if (ev.prediction) {
        body2 = '<div class="log-body">' + escapeHtml(ev.prediction) + '</div>';
      } else if (ev.assertion) {
        body2 = '<div class="log-body">' + escapeHtml(ev.assertion) + '</div>';
      }
      appendLog('<span class="log-agent">&#10004; ' + escapeHtml(label2) + ' output</span>' + body2);
    } else if (ev.type === 'injection') {
      setStep('Injecting into ' + ev.file + '...', 95);
    } else if (ev.type === 'pipeline_complete') {
      setStep('Done! ' + ev.total_tests + ' assertions injected.', 100);
      setStatus('done', 'Done');
      Object.values(agentMap).forEach(function(id) { var el = document.getElementById(id); if (el) el.classList.remove('active'); });
      document.getElementById('showGraphBtn').style.display = '';
    } else if (ev.type === 'error' || ev.type === 'pipeline_error') {
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
    clearLog();
  } else if (msg.type === 'config') {
    var setVal = function(id, v) { var el = document.getElementById(id); if (el && v !== undefined && v !== null) el.value = v; };
    setVal('apiEndpoint', msg.apiEndpoint);
    setVal('modelName', msg.modelName);
    setVal('language', msg.language);
    setVal('maxWorkers', msg.maxWorkers);
    setVal('temperature', msg.temperature);
    setVal('pythonPath', msg.pythonPath);
    setVal('condaEnv', msg.condaEnv);
    var fr = document.getElementById('forceReindex');
    if (fr) fr.checked = !!msg.forceReindex;
    var fre = document.getElementById('forceReextract');
    if (fre) fre.checked = !!msg.forceReextract;
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
