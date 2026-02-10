import json
import os
import argparse

def main(args):
    """
    Finds all '_1.nii.gz' files in a directory and creates a JSON file
    that prompts the model to generate a report for each one.
    """
    print("--- Starting Report Generation JSON Creation ---")

    # The specific prompt you want to use for every scan
    report_prompt = "<image>\nPlease generate the report for the chest CT volume provided."

    # The placeholder answer (the script requires this structure)
    placeholder_answer = "This is a placeholder for the ground truth answer."

    if not os.path.isdir(args.scan_directory):
        print(f"\n[FATAL ERROR] The specified scan directory does not exist: {args.scan_directory}")
        return

    print(f"Searching for scans in: {args.scan_directory}")
    
    final_json_data = []

    # Walk through the directory and find all relevant files
    for root, dirs, files in os.walk(args.scan_directory):
        for filename in files:
            if filename.endswith("_1.nii.gz"):
                
                # Create the conversational structure
                conversation = [
                    {
                        "from": "human",
                        "value": report_prompt
                    },
                    {
                        "from": "gpt",
                        "value": placeholder_answer
                    }
                ]
                
                # Create the final object for this image
                image_object = {
                    "image": filename,
                    "conversations": conversation
                }
                
                final_json_data.append(image_object)

    if not final_json_data:
        print(f"\n[WARNING] No files ending in '_1.nii.gz' were found in the directory.")
        return

    try:
        print(f"Found {len(final_json_data)} scans to process.")
        print(f"Saving final JSON to: {args.output_json}")
        with open(args.output_json, 'w') as f:
            json.dump(final_json_data, f, indent=2)
        print("--- JSON File Creation Successful! ---")
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not write the JSON file.")
        print(f"Details: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a LLaVA JSON file for report generation from a directory of scans.")
    
    parser.add_argument(
        "--scan-directory", 
        type=str, 
        required=True,
        help="Path to the directory containing your downloaded NIfTI scans (e.g., ~/validation_scans)."
    )
    parser.add_argument(
        "--output-json", 
        type=str, 
        required=True,
        help="Path for the output JSON file (e.g., merged_output_valid.json)."
    )
    
    args = parser.parse_args()
    main(args)