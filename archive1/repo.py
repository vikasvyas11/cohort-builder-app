import os


def consolidate_repository(target_directory, output_filename):
    # Folders to completely skip to keep the text file clean and small
    EXCLUDE_FOLDERS = {'.git', '__pycache__', '.venv', 'env', 'venv', '.ipynb_checkpoints'}

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write("=== CONSOLIDATED SOURCE CODE MANIFEST ===\n\n")

        # 1. Explicitly look for and capture the requirements file first
        requirements_path = os.path.join(target_directory, 'requirements.txt')
        if os.path.exists(requirements_path):
            outfile.write("==================================================\n")
            outfile.write("FILE: requirements.txt\n")
            outfile.write("==================================================\n\n")
            with open(requirements_path, 'r', encoding='utf-8') as req_file:
                outfile.write(req_file.read())
            outfile.write("\n\n")

        # 2. Recursively crawl through the directory for Python files
        for root, dirs, files in os.walk(target_directory):
            # Modify dirs in-place to ignore internal env or git folders
            dirs[:] = [d for d in dirs if d not in EXCLUDE_FOLDERS]

            for file in files:
                # Target only Python source code files
                if file.endswith('.py') and file != 'bundle_code.py':
                    full_file_path = os.path.join(root, file)
                    # Compute relative path from the root directory for easy structural reference
                    relative_path = os.path.relpath(full_file_path, target_directory)

                    outfile.write("==================================================\n")
                    outfile.write(f"FILE: {relative_path}\n")
                    outfile.write("==================================================\n\n")

                    try:
                        with open(full_file_path, 'r', encoding='utf-8') as source_file:
                            outfile.write(source_file.read())
                    except Exception as error:
                        outfile.write(f"[ERROR READING FILE CONTENTS: {error}]\n")

                    outfile.write("\n\n")

    print(f"Compilation successful! Consolidated file generated at: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
    # Target the current directory where the script is executed
    consolidate_repository('.', 'consolidated_codebase.txt')