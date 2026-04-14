import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { ApiClient } from '../backend/apiClient';
import { GraphData } from '../types/api';

export class GraphPanelProvider {
    private panel: vscode.WebviewPanel | null = null;
    private visNetworkJs: string | null = null;

    constructor(private apiClient: ApiClient, private extensionPath: string) {}

    private getVisNetworkJs(): string {
        if (!this.visNetworkJs) {
            const visPath = path.join(
                this.extensionPath,
                'node_modules', 'vis-network', 'standalone', 'umd', 'vis-network.min.js'
            );
            this.visNetworkJs = fs.readFileSync(visPath, 'utf8');
        }
        return this.visNetworkJs;
    }

    async show(projectPath: string, language: string): Promise<void> {
        if (this.panel) { this.panel.reveal(); return; }

        this.panel = vscode.window.createWebviewPanel(
            'assertgen-graph', 'AssertGen: Code Graph',
            vscode.ViewColumn.One,
            { enableScripts: true, retainContextWhenHidden: true }
        );

        this.panel.onDidDispose(() => { this.panel = null; });
        this.panel.webview.html = this.getLoadingHtml();

        try {
            const data = await this.apiClient.getGraphData(projectPath, language);
            this.panel.webview.html = this.getGraphHtml(data);
        } catch (e) {
            this.panel.webview.html = `<html><body style="font-family:sans-serif;padding:20px;color:#ccc;background:#1e1e1e">
                <h3>Graph not available</h3><p>Build the code graph first by running Gen Test.</p>
                <p>Error: ${e}</p></body></html>`;
        }
    }

    private getLoadingHtml(): string {
        return `<html><body style="font-family:sans-serif;padding:20px;color:#ccc;background:#1e1e1e">
            <p>Loading code graph...</p></body></html>`;
    }

    private getGraphHtml(data: GraphData): string {
        const nodesJson = JSON.stringify(data.nodes || []);
        const edgesJson = JSON.stringify(data.edges || []);
        const visJs = this.getVisNetworkJs();
        return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline';">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; margin: 0; padding: 0; }
  body { background: #1e1e1e; color: #ccc; font-family: var(--vscode-font-family, sans-serif); display: flex; flex-direction: column; overflow: hidden; }
  #toolbar { padding: 8px 12px; background: #252526; border-bottom: 1px solid #333; display: flex; gap: 8px; align-items: center; flex-shrink: 0; }
  #toolbar input { background: #3c3c3c; border: 1px solid #555; color: #ccc; padding: 4px 8px; border-radius: 3px; width: 200px; }
  #toolbar span { font-size: 12px; color: #888; }
  #graph { flex: 1; width: 100%; position: relative; }
  #details { position: fixed; right: 0; top: 0; width: 280px; height: 100%; background: #252526; border-left: 1px solid #333; padding: 12px; overflow-y: auto; display: none; font-size: 12px; }
  #details h3 { margin-bottom: 8px; color: #ccc; }
  #details pre { background: #1e1e1e; padding: 8px; border-radius: 3px; overflow-x: auto; white-space: pre-wrap; font-size: 11px; }
  .legend { display: flex; gap: 12px; font-size: 11px; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; }
</style>
</head>
<body>
<div id="toolbar">
  <input id="search" placeholder="Search class/method..." oninput="filterNodes(this.value)">
  <div class="legend">
    <div class="legend-item"><div class="dot" style="background:#4e9bf5"></div>Class</div>
    <div class="legend-item"><div class="dot" style="background:#5cb85c"></div>Method</div>
    <div class="legend-item"><div class="dot" style="background:#f0ad4e"></div>Field</div>
  </div>
  <span id="stats"></span>
</div>
<div id="graph"></div>
<div id="details">
  <button onclick="document.getElementById('details').style.display='none'" style="float:right;background:none;border:none;color:#ccc;cursor:pointer;font-size:16px">&#10005;</button>
  <h3 id="detailTitle"></h3>
  <div id="detailBody"></div>
</div>

<script>${visJs}</script>
<script>
var rawNodes = ${nodesJson};
var rawEdges = ${edgesJson};

document.getElementById('stats').textContent = rawNodes.length + ' nodes \u00b7 ' + rawEdges.length + ' edges';

// Ensure graph container fills available space
var container = document.getElementById('graph');
container.style.height = (window.innerHeight - document.getElementById('toolbar').offsetHeight) + 'px';
window.addEventListener('resize', function() {
  container.style.height = (window.innerHeight - document.getElementById('toolbar').offsetHeight) + 'px';
  if (network) { network.redraw(); network.fit(); }
});

var colorMap = { Class: '#4e9bf5', Method: '#5cb85c', Field: '#f0ad4e' };

var nodes, edges, network;
try {
  nodes = new vis.DataSet(rawNodes.map(function(n) {
    return {
      id: n.id,
      label: n.label,
      title: n.type + (n.filePath ? '\\n' + n.filePath : ''),
      color: {
        background: colorMap[n.type] || '#888',
        border: '#333',
        highlight: { background: '#fff', border: '#666' }
      },
      font: { color: '#fff', size: 12 },
      shape: n.type === 'Class' ? 'box' : 'dot',
      size: n.type === 'Class' ? 20 : 12,
      nodeData: n,
    };
  }));

  var edgeColorMap = { CALLS: '#e74c3c', EXTENDS: '#3498db', IMPLEMENTS: '#9b59b6', HAS_METHOD: '#555', HAS_FIELD: '#444' };

  edges = new vis.DataSet(rawEdges.map(function(e, i) {
    return {
      id: i,
      from: e.from,
      to: e.to,
      color: { color: edgeColorMap[e.type] || '#555', highlight: '#fff' },
      arrows: 'to',
      label: '',
      font: { size: 9, color: '#888' },
      width: e.type === 'CALLS' ? 1.5 : 1,
      dashes: e.type === 'EXTENDS' || e.type === 'IMPLEMENTS',
    };
  }));

  var largeGraph = rawNodes.length > 200;
  network = new vis.Network(container, { nodes: nodes, edges: edges }, {
    physics: {
      enabled: !largeGraph,
      stabilization: { iterations: 50 },
      barnesHut: { gravitationalConstant: -3000, springLength: 100 }
    },
    interaction: { hover: true, navigationButtons: true, keyboard: true },
    layout: { improvedLayout: false },
  });
} catch(err) {
  container.innerHTML = '<p style="color:#f48771;padding:20px;font-family:sans-serif">Render error: ' + err + '</p>';
}

network.on('click', function(params) {
  if (params.nodes.length > 0) {
    var nodeId = params.nodes[0];
    var node = nodes.get(nodeId);
    var details = document.getElementById('details');
    var nd = node.nodeData;
    document.getElementById('detailTitle').textContent = nd.label;
    document.getElementById('detailBody').innerHTML =
      '<p><b>Type:</b> ' + nd.type + '</p>' +
      (nd.filePath ? '<p><b>File:</b> ' + nd.filePath + '</p>' : '') +
      (nd.body ? '<p><b>Body:</b></p><pre>' + escapeHtml(nd.body.substring(0, 500)) + '</pre>' : '');
    details.style.display = 'block';
  }
});

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function filterNodes(q) {
  if (!q) {
    nodes.forEach(function(n) { nodes.update({ id: n.id, hidden: false }); });
    return;
  }
  q = q.toLowerCase();
  nodes.forEach(function(n) {
    nodes.update({ id: n.id, hidden: !n.label.toLowerCase().includes(q) });
  });
}
</script>
</body>
</html>`;
    }
}
