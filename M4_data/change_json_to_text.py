import json

# Function to read JSON file
def read_json_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Function to write text to a file
def write_to_file(file_path, text_list):
    with open(file_path, 'w', encoding='utf-8') as file:
        for text in text_list:
            file.write(text.replace('\n', '\\n') + '\n')

# Main function
def extract_and_write_text(json_file_path, human_file_path, machine_file_path):
    data = read_json_file(json_file_path)
    
    human_texts = [entry['human_text'] for entry in data]
    machine_texts = [entry['machine_text'] for entry in data]
    
    write_to_file(human_file_path, human_texts)
    write_to_file(machine_file_path, machine_texts)

# Example usage
json_file_path = '/data/yl7622/MRes/M4/data/wikihow_chatGPT.jsonl'
human_file_path = '/data/yl7622/MRes/M4/my_data/HPC'
machine_file_path = '/data/yl7622/MRes/M4/my_data/MGC'

extract_and_write_text(json_file_path, human_file_path, machine_file_path)
