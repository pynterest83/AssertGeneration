import * as http from 'http';
import { ExtractionResult, GraphStats, GraphData, PipelineConfig, PipelineResult } from '../types/api';

function httpRequest<T>(method: string, urlStr: string, body?: unknown): Promise<T> {
    return new Promise((resolve, reject) => {
        const url = new URL(urlStr);
        const payload = body !== undefined ? JSON.stringify(body) : undefined;
        const options: http.RequestOptions = {
            hostname: url.hostname,
            port: Number(url.port) || 80,
            path: url.pathname + url.search,
            method,
            headers: {
                'Content-Type': 'application/json',
                ...(payload ? { 'Content-Length': Buffer.byteLength(payload) } : {}),
            },
        };

        const req = http.request(options, (res) => {
            let data = '';
            res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
            res.on('end', () => {
                if (res.statusCode && res.statusCode >= 400) {
                    reject(new Error(`API error ${res.statusCode}: ${data}`));
                    return;
                }
                try {
                    resolve(JSON.parse(data) as T);
                } catch {
                    reject(new Error(`Invalid JSON response: ${data.substring(0, 200)}`));
                }
            });
        });

        req.on('error', reject);
        req.setTimeout(600000, () => { req.destroy(new Error('Request timeout')); });

        if (payload) { req.write(payload); }
        req.end();
    });
}

export class ApiClient {
    constructor(private getBaseUrl: () => string) {}

    private post<T>(path: string, body: unknown): Promise<T> {
        return httpRequest<T>('POST', `${this.getBaseUrl()}${path}`, body);
    }

    private get<T>(path: string): Promise<T> {
        return httpRequest<T>('GET', `${this.getBaseUrl()}${path}`);
    }

    extractTests(projectPath: string, language: string): Promise<ExtractionResult> {
        return this.post('/extract', { project_path: projectPath, language });
    }

    buildGraph(projectPath: string, language: string, forceReindex: boolean): Promise<GraphStats> {
        return this.post('/build-graph', { project_path: projectPath, language, force_reindex: forceReindex });
    }

    getGraphData(projectPath: string, language: string): Promise<GraphData> {
        return this.get(`/graph-data?project_path=${encodeURIComponent(projectPath)}&language=${language}`);
    }

    runPipeline(config: PipelineConfig): Promise<PipelineResult> {
        return this.post('/run-pipeline', config);
    }

    getGraphStatus(projectPath: string): Promise<{ built: boolean }> {
        return this.get(`/graph-status?project_path=${encodeURIComponent(projectPath)}`);
    }

    getPipelineStatus(projectPath: string): Promise<{ has_results: boolean; test_count: number }> {
        return this.get(`/pipeline-status?project_path=${encodeURIComponent(projectPath)}`);
    }

    health(): Promise<{ status: string }> {
        return this.get('/health');
    }
}
