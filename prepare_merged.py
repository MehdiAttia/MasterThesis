import json
import os

# Files
input_file = 'processed_scan_data_cleaned.json'
output_file = 'merged_output_valid.json'

# Check if input exists
if not os.path.exists(input_file):
    print(f"Error: '{input_file}' not found. Please run the previous cleaning step first.")
else:
    try:
        # 1. Load the cleaned source data
        with open(input_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)

        transformed_data = []

        # 2. Transform the data
        for entry in source_data:
            image_filename = entry.get("scan_name")
            questions = entry.get("questions", [])

            # Create a separate entry for every single question
            for question in questions:
                new_record = {
                    "image": image_filename,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{question}"
                        },
                        {
                            "from": "gpt",
                            "value": "This is a placeholder for the ground truth answer."
                        }
                    ]
                }
                transformed_data.append(new_record)

        # 3. Save to merged_output_valid.json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, indent=2)

        print(f"Transformation complete.")
        print(f"Generated {len(transformed_data)} question-answer pairs.")
        print(f"Saved to: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")