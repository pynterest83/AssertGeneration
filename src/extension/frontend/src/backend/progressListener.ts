import * as http from 'http';
import { ProgressEvent } from '../types/api';

export class ProgressListener {
    private req: http.ClientRequest | null = null;

    listen(baseUrl: string, onEvent: (event: ProgressEvent) => void): void {
        this.stop();
        const url = new URL(`${baseUrl}/progress`);
        this.req = http.get(
            { hostname: url.hostname, port: url.port, path: url.pathname },
            (res) => {
                let buffer = '';
                res.on('data', (chunk: Buffer) => {
                    buffer += chunk.toString();
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const event = JSON.parse(line.slice(6)) as ProgressEvent;
                                onEvent(event);
                            } catch {
                                // ignore malformed SSE lines
                            }
                        }
                    }
                });
            }
        );
        this.req.on('error', () => {
            // ignore connection errors; caller controls lifecycle via stop()
        });
    }

    stop(): void {
        if (this.req) {
            this.req.destroy();
            this.req = null;
        }
    }
}
