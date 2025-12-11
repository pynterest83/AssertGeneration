import os
import json
from tqdm import tqdm
import javalang

class ExtractProjectElements:
    def __init__(self, projects_base_dir, output_dir, projects):
        self.projects_base_dir = projects_base_dir
        self.output_dir = output_dir
        self.projects = projects
        os.makedirs(output_dir, exist_ok=True)

    def _find_java_files(self, project_path):
        java_files = []
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
        return java_files

    def _get_method_body_from_source(self, source_code, method_node):
        if not method_node.position:
            return ""
        
        lines = source_code.split('\n')
        start_line = method_node.position.line - 1
        
        char_pos = sum(len(lines[i]) + 1 for i in range(start_line))
        
        brace_pos = source_code.find('{', char_pos)
        if brace_pos == -1:
            return ""
        
        brace_count = 0
        end_pos = brace_pos
        
        for i, char in enumerate(source_code[brace_pos:], start=brace_pos):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        
        return source_code[char_pos:end_pos].strip()

    def _get_return_type(self, method_node):
        if not hasattr(method_node, 'return_type') or method_node.return_type is None:
            return 'void'
        if hasattr(method_node.return_type, 'name'):
            return method_node.return_type.name
        return str(method_node.return_type)

    def _get_parameters(self, node):
        params = []
        if node.parameters:
            for param in node.parameters:
                param_type = 'Object'
                if param.type and hasattr(param.type, 'name'):
                    param_type = param.type.name
                params.append({'type': param_type, 'name': param.name})
        return params

    def _extract_methods_from_file(self, file_path, project_path):
        methods = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read().replace('\r\n', '\n')
        except:
            return methods
        
        try:
            tree = javalang.parse.parse(source_code)
        except:
            return methods
        
        rel_path = os.path.relpath(file_path, project_path)
        
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            class_name = 'Unknown'
            for parent in reversed(path):
                if isinstance(parent, (javalang.tree.ClassDeclaration, 
                                       javalang.tree.InterfaceDeclaration,
                                       javalang.tree.EnumDeclaration)):
                    class_name = parent.name
                    break
            
            return_type = self._get_return_type(node)
            parameters = self._get_parameters(node)
            method_body = self._get_method_body_from_source(source_code, node)
            
            start_pos, end_pos = 0, 0
            if node.position:
                lines = source_code.split('\n')
                start_line = node.position.line - 1
                start_pos = sum(len(lines[i]) + 1 for i in range(start_line))
                end_pos = start_pos + len(method_body)
            
            methods.append({
                'name': node.name,
                'class': class_name,
                'relative_path': rel_path,
                'return_type': return_type,
                'parameters': parameters,
                'body_raw': method_body,
                'start': start_pos,
                'end': end_pos,
            })
        
        for path, node in tree.filter(javalang.tree.ConstructorDeclaration):
            class_name = 'Unknown'
            for parent in reversed(path):
                if isinstance(parent, javalang.tree.ClassDeclaration):
                    class_name = parent.name
                    break
            
            parameters = self._get_parameters(node)
            method_body = self._get_method_body_from_source(source_code, node)
            
            start_pos, end_pos = 0, 0
            if node.position:
                lines = source_code.split('\n')
                start_line = node.position.line - 1
                start_pos = sum(len(lines[i]) + 1 for i in range(start_line))
                end_pos = start_pos + len(method_body)
            
            methods.append({
                'name': node.name,
                'class': class_name,
                'relative_path': rel_path,
                'return_type': class_name,
                'parameters': parameters,
                'body_raw': method_body,
                'start': start_pos,
                'end': end_pos,
            })
        
        return methods

    def _convert_to_context_format(self, method, project_path):
        fpath = os.path.join(project_path, method["relative_path"])
        base_dir_list = os.path.normpath(project_path).split(os.sep)
        
        try:
            with open(fpath, "r", encoding="utf8") as f:
                file_content = f.read().replace("\r\n", "\n")
        except:
            return None
        
        method_content = file_content[method["start"]:method["end"]]
        if not method_content.strip():
            method_content = method["body_raw"]
        
        fpath_list = list(os.path.normpath(fpath).split(os.sep)[len(base_dir_list):])
        
        return {
            "content": method_content,
            "metadata": {
                "fpath_tuple": fpath_list,
                "name": method["name"],
                "class": method["class"],
                "start_line_no": method["start"],
                "end_line_no": method["end"],
                "return_type": method["return_type"],
                "parameters": method["parameters"],
                "body_raw": method["body_raw"],
            },
        }

    def _extract_types(self, parsed_methods):
        types_list = []
        types_dict = {}
        for method in parsed_methods:
            key = (method["class"], method["relative_path"])
            key_str = f"{method['class']}::{method['relative_path']}"
            if key_str not in types_dict:
                types_dict[key_str] = {
                    "class": method["class"],
                    "relative_path": method["relative_path"],
                    "methods": []
                }
            types_dict[key_str]["methods"].append(method["name"])
        
        return list(types_dict.values())

    def extract_project(self, project_name):
        project_path = os.path.join(self.projects_base_dir, project_name)
        
        if not os.path.exists(project_path):
            print(f"Project not found: {project_path}")
            return [], []
        
        java_files = self._find_java_files(project_path)
        print(f"Found {len(java_files)} Java files")
        
        parsed_methods = []
        for java_file in tqdm(java_files, desc=f"Parsing", leave=False):
            methods = self._extract_methods_from_file(java_file, project_path)
            parsed_methods.extend(methods)
        
        if not parsed_methods:
            return [], []
        
        method_contexts = []
        for method in parsed_methods:
            context = self._convert_to_context_format(method, project_path)
            if context:
                method_contexts.append(context)
        
        types_list = self._extract_types(parsed_methods)
        
        project_output_dir = os.path.join(self.output_dir, project_name)
        os.makedirs(project_output_dir, exist_ok=True)
        
        methods_path = os.path.join(project_output_dir, 'methods.jsonl')
        types_path = os.path.join(project_output_dir, 'types.json')
        
        with open(methods_path, 'w', encoding='utf-8') as f:
            for method in method_contexts:
                f.write(json.dumps(method, ensure_ascii=False) + '\n')
        
        with open(types_path, 'w', encoding='utf-8') as f:
            json.dump(types_list, f, indent=2, ensure_ascii=False)
        
        print(f"Extracted {len(method_contexts)} methods, {len(types_list)} types")
        print(f"Saved to {project_output_dir}/")
        
        return method_contexts, types_list

    def extract_all(self):
        for project in tqdm(self.projects, desc="Projects"):
            print(f"\n{'='*50}\n{project}\n{'='*50}")
            self.extract_project(project)


if __name__ == '__main__':
    PROJECTS = [
        'async-http-client',
        'commons-beanutils-1.9.4',
        'commons-lang3-3.12.0-src',
    ]
    
    extractor = ExtractProjectElements(
        projects_base_dir='../RQ2/EvoSuiteTests',
        output_dir='../results',
        projects=PROJECTS
    )
    extractor.extract_all()
