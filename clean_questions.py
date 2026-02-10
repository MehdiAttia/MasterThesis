import json
import os

input_file = 'processed_scan_data.json'
output_file = 'processed_scan_data_cleaned.json'

# Check if file exists before processing
if not os.path.exists(input_file):
    print(f"Error: The file '{input_file}' was not found.")
else:
    try:
        # 1. Load the JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. Process each entry to filter questions
        count_removed = 0
        for entry in data:
            if 'questions' in entry:
                original_len = len(entry['questions'])
                # Keep only strings that end with '?' (stripping whitespace first)
                entry['questions'] = [q for q in entry['questions'] if q.strip().endswith('?')]
                count_removed += original_len - len(entry['questions'])

        # 3. Save the cleaned data to a new file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        print(f"Success! Processed {len(data)} records.")
        print(f"Removed {count_removed} non-question items.")
        print(f"Cleaned data saved to: {output_file}")

    except json.JSONDecodeError:
        print(f"Error: The file '{input_file}' is not valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")