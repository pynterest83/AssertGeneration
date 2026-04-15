import os
from collections import defaultdict

# comment incompatible assertions from error log
def comment_assertions(error_log_path: str):
    file_to_error_lines = defaultdict(set) # prevent duplicate lines
    with open(error_log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            # process error lines based on Java syntax
            if "[ERROR] " in line and ".java:[" in line:
                # Extract file path: từ [ERROR] đến .java
                start = line.find('[ERROR] ') + 8
                end = line.find('.java', start) + 5
                file_path = line[start:end].strip()
                
                # Extract line number: java:[lineNo,col]
                start = line.find('java:[') + 6
                end = line.find(',', start)
                line_no = line[start:end]

                file_to_error_lines[file_path].add(line_no)
    
    total_commented = 0
    for file_path, line_numbers in file_to_error_lines.items():
        if not os.path.exists(file_path):
            continue

        # comment incompatible assertions
        line_count = 0
        with open(file_path, 'r+', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            f.seek(0)
            f.truncate() # remove all content
            for line in lines:
                line_count += 1
                # comment incompatible assertions (compile error)
                if str(line_count) in line_numbers and line.strip().startswith("assert"):
                    line = "//COMPILE_ERROR " + line
                    total_commented += 1
                f.write(line)
    # total Tce
    return total_commented