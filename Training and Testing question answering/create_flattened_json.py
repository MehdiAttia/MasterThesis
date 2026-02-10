import pandas as pd
import json
import argparse
import ast  # Used to safely evaluate string-formatted lists

def main(args):
    """
    Reads a CSV with list-formatted questions/answers and creates a 'flattened'
    JSON file where each Q&A pair for a scan becomes a top-level entry.
    It specifically processes 'a' -> 'b' scan changes and uses a placeholder for the answer.
    """
    print("--- Starting Flattened JSON Conversion for Validation ---")

    try:
        print(f"Reading input CSV: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not read the CSV file. Please check the path.")
        print(f"Details: {e}")
        return

    final_json_data = []
    processed_count = 0
    skipped_rows = 0

    print("Processing rows and flattening conversations...")
    for _, row in df.iterrows():
        try:
            # --- START: NEW FILTERING LOGIC ---
            # Rule 1: Only process rows where the change is from scan 'a' to 'b'
            if row['Prior Scan ID'] != 'a' or row['Current Scan ID'] != 'b':
                skipped_rows += 1
                continue
            # --- END: NEW FILTERING LOGIC ---

            patient_id = row['Patient ID']
            
            # The questions are about the 'Current Scan', which is always the 'b_1' version
            image_filename = f"train_{patient_id}_b_1.nii.gz"
            
            # The 'Questions' and 'Answers' columns are strings that look like lists.
            # We use ast.literal_eval to safely parse them into actual Python lists.
            questions = ast.literal_eval(row['Questions'])
            # We still parse the answers to check for length consistency, but won't use them.
            answers = ast.literal_eval(row['Answers'])
            
            # Ensure we have the same number of questions and answers
            if len(questions) != len(answers):
                print(f"Warning: Mismatch in Q&A count for Patient ID {patient_id}. Skipping row.")
                skipped_rows += 1
                continue

            # Loop through each question-answer pair for this scan
            for i in range(len(questions)):
                question_text = questions[i]
                
                # --- START: NEW ANSWER LOGIC ---
                # Rule 2: Use a placeholder for the answer, so the model must generate it.
                placeholder_answer = "This is a placeholder for the ground truth answer."
                # --- END: NEW ANSWER LOGIC ---

                # Create the conversational structure for this single Q&A pair
                conversation = [
                    {
                        "from": "human",
                        "value": f"<image>\n{question_text}"
                    },
                    {
                        "from": "gpt",
                        "value": placeholder_answer
                    }
                ]
                
                # Create the final JSON object for this single Q&A pair
                image_object = {
                    "image": image_filename,
                    "conversations": conversation
                }
                
                final_json_data.append(image_object)
                processed_count += 1

        except (SyntaxError, ValueError) as e:
            print(f"Warning: Could not parse Q&A lists for Patient ID {patient_id}. Skipping row. Error: {e}")
            skipped_rows += 1
            continue
        except KeyError as e:
            print(f"Warning: Missing column {e} in a row. Skipping.")
            skipped_rows += 1
            continue

    try:
        print(f"\nSuccessfully processed {processed_count} individual Q&A pairs.")
        print(f"Skipped {skipped_rows} rows that were not 'a' to 'b' transitions or had errors.")
        print(f"Saving final JSON to: {args.output_json}")
        with open(args.output_json, 'w') as f:
            json.dump(final_json_data, f, indent=2)
        print("--- Conversion Successful! ---")
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not write the JSON file.")
        print(f"Details: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and flatten a validation CSV to LLaVA JSON format.")
    
    parser.add_argument(
        "--input-csv", 
        type=str, 
        required=True,
        help="Path to the input CSV file (e.g., filtered_dataset_max700_grounded_questions_v2.csv)."
    )
    parser.add_argument(
        "--output-json", 
        type=str, 
        required=True,
        help="Path for the output JSON file (e.g., merged_output_valid.json)."
    )
    
    args = parser.parse_args()
    main(args)