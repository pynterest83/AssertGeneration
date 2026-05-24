export interface ExtractionResult {
    status?: string;
    test_count: number;
    inputs_csv: string;
    meta_csv: string;
}

export interface GraphStats {
    status?: string;
    class_count: number;
    method_count: number;
}

export interface GraphNode {
    id: string;
    label: string;
    type: 'Class' | 'Method' | 'Field';
    filePath?: string;
    body?: string;
}

export interface GraphEdge {
    from: string;
    to: string;
    type: 'CALLS' | 'EXTENDS' | 'IMPLEMENTS' | 'HAS_METHOD' | 'HAS_FIELD';
}

export interface GraphData {
    status?: string;
    nodes: GraphNode[];
    edges: GraphEdge[];
}

export interface PipelineConfig {
    project_path: string;
    language: string;
    api_endpoint: string;
    model_name: string;
    api_key: string;
    max_workers: number;
    temperature: number;
    force_reindex: boolean;
}

export interface PipelineResult {
    status?: string;
    message?: string;
}

export interface TestCaseItem {
    testName: string;
    filePath: string;
    status: 'pending' | 'done' | 'error';
    assertion?: string;
}

export type AgentName =
    | 'exception_classifier'
    | 'code_analyzer'
    | 'state_predictor'
    | 'assertion_generator';

export interface ExtractionProgressEvent {
    type: 'extraction';
    current: number;
    total: number;
    file?: string;
}

export interface ExtractionCompleteEvent {
    type: 'extraction_complete';
    test_count: number;
}

export interface GraphBuildingProgressEvent {
    type: 'graph_building';
    current: number;
    total: number;
    file?: string;
    phase?: string;
}

export interface GraphBuildingCompleteEvent {
    type: 'graph_building_complete';
}

export interface InferenceProgressEvent {
    type: 'inference_progress';
    current: number;
    total: number;
    test_name?: string;
}

export interface InferenceEvent {
    type: 'inference';
    agent: AgentName;
    status?: 'started' | 'finished';
    test_name?: string;
}

export interface InferenceCompleteEvent {
    type: 'inference_complete';
    result_count: number;
}

export interface InjectionEvent {
    type: 'injection';
    file: string;
    count?: number;
}

export interface InjectionWarningEvent {
    type: 'injection_warning';
    message: string;
}

export interface StageEvent {
    type: 'stage';
    stage: string;
    message?: string;
}

export interface PipelineCompleteEvent {
    type: 'pipeline_complete';
    total_tests: number;
    injected_files?: string[];
    message?: string;
}

export interface PipelineErrorEvent {
    type: 'pipeline_error';
    message: string;
}

export interface ErrorEvent {
    type: 'error';
    message: string;
}

export interface PingEvent {
    type: 'ping';
}

export type ProgressEvent =
    | ExtractionProgressEvent
    | ExtractionCompleteEvent
    | GraphBuildingProgressEvent
    | GraphBuildingCompleteEvent
    | InferenceProgressEvent
    | InferenceEvent
    | InferenceCompleteEvent
    | InjectionEvent
    | InjectionWarningEvent
    | StageEvent
    | PipelineCompleteEvent
    | PipelineErrorEvent
    | ErrorEvent
    | PingEvent;
