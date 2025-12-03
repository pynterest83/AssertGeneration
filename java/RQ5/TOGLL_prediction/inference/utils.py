   
import json, shutil

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_values_by_id(data, search_id):
    for item in data:
        if item.get("id") == search_id:
            return item.get("target"), item.get("output"), item.get("score"), item.get("exact_match")
            
    return None


def copy_csv_file(source_file, destination_file):
    try:
        shutil.copyfile(source_file, destination_file)
        print(f"File copied successfully from {source_file} to {destination_file}")
    except Exception as e:
        print(f"Error occurred while copying file: {e}")
