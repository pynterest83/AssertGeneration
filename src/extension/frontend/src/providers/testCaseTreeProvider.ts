import * as vscode from 'vscode';
import { TestCaseItem } from '../types/api';

export class TestCaseTreeItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly testCase?: TestCaseItem,
    ) {
        super(label, collapsibleState);
        if (testCase) {
            this.tooltip = testCase.assertion || 'Pending';
            this.iconPath = new vscode.ThemeIcon(
                testCase.status === 'done' ? 'pass' :
                testCase.status === 'error' ? 'error' : 'circle-large-outline'
            );
        }
    }
}

export class TestCaseTreeProvider implements vscode.TreeDataProvider<TestCaseTreeItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<TestCaseTreeItem | undefined | void>();
    onDidChangeTreeData = this._onDidChangeTreeData.event;
    private testsByFile: Map<string, TestCaseItem[]> = new Map();

    setTests(tests: TestCaseItem[]): void {
        this.testsByFile.clear();
        for (const t of tests) {
            const key = t.filePath;
            if (!this.testsByFile.has(key)) this.testsByFile.set(key, []);
            this.testsByFile.get(key)!.push(t);
        }
        this._onDidChangeTreeData.fire();
    }

    updateTest(testName: string, status: 'done' | 'error', assertion?: string): void {
        for (const tests of this.testsByFile.values()) {
            const t = tests.find(x => x.testName === testName);
            if (t) {
                t.status = status;
                if (assertion) t.assertion = assertion;
            }
        }
        this._onDidChangeTreeData.fire();
    }

    clear(): void {
        this.testsByFile.clear();
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: TestCaseTreeItem): vscode.TreeItem { return element; }

    getChildren(element?: TestCaseTreeItem): TestCaseTreeItem[] {
        if (!element) {
            return Array.from(this.testsByFile.keys()).map(f =>
                new TestCaseTreeItem(f, vscode.TreeItemCollapsibleState.Expanded)
            );
        }
        const tests = this.testsByFile.get(element.label as string) || [];
        return tests.map(t => {
            const item = new TestCaseTreeItem(t.testName, vscode.TreeItemCollapsibleState.None, t);
            item.description = t.status === 'done' ? t.assertion?.substring(0, 50) : t.status;
            return item;
        });
    }
}
