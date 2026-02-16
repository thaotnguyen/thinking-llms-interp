import os
import subprocess
import glob

BASE_DIR = "white_box_medqa-edited"
SCRIPT_PATH = "analyze_traces_state_transition.py"

def main():
    # Find all results.labeled.json files
    pattern = os.path.join(BASE_DIR, "**", "results.labeled.json")
    # GLOB recursive
    files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(files)} labeled JSON files.")
    
    for json_file in files:
        # Determine model name and output directory
        # Path is like: white_box_medqa-edited/MODEL_NAME/medqa_with_second_responses/results.labeled.json
        parts = json_file.split(os.sep)
        try:
            # Assumes standard structure, ensuring we pick the folder inside white_box_medqa-edited
            if parts[0] == BASE_DIR:
                model_name = parts[1]
            else:
                # If path is absolute or different relative path
                try:
                    idx = parts.index("white_box_medqa-edited")
                    model_name = parts[idx + 1]
                except ValueError:
                     # Fallback
                    model_name = os.path.basename(os.path.dirname(os.path.dirname(json_file)))
        except IndexError:
            model_name = "unknown_model"
            
        print(f"Processing model: {model_name} from {json_file}")
        
        # Create output dir specific to model - creating it as a sibling to the results file or inside analysis folder?
        # User asked to "run this on all subfolders". Usually artifacts are placed near the data.
        # Let's put it in `analysis_outputs_pdf` in the same directory as the json file to keep it clean.
        out_dir = os.path.join(os.path.dirname(json_file), "analysis_outputs_pdf")
        
        # Run the analysis script
        cmd = [
            "python", SCRIPT_PATH,
            "--labeled_csv", json_file,
            "--out_dir", out_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully processed {model_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {model_name}: {e}")

if __name__ == "__main__":
    main()
