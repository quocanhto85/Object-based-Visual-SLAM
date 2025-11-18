import os

# --- Configuration ---
input_file = '../third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory.txt'
# Create a temporary output file path
output_file = input_file + '.cleaned_temp' 

print(f"Attempting to clean file: {input_file}")

try:
    # Read the file and clean each line
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines_cleaned = 0
        for line in infile:
            # .rstrip() removes all trailing whitespace (spaces, tabs, newlines)
            # We then add a single newline character ('\n') back for proper line ending
            cleaned_line = line.rstrip()
            outfile.write(cleaned_line + '\n')
            
            # Check if any cleaning was done (i.e., if the original line had trailing space)
            if len(line) != len(cleaned_line) + 1: # +1 accounts for the newline that was stripped
                lines_cleaned += 1

    # Overwrite the original file with the cleaned content
    os.replace(output_file, input_file)
    
    print(f"✅ Successfully cleaned and updated: {input_file}")
    print(f"   Removed trailing whitespace from {lines_cleaned} lines.")

except FileNotFoundError:
    print(f"❌ ERROR: File not found at path: {input_file}")
    print("Please ensure your Current Working Directory is correct or use an absolute path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # Ensure the temporary file is deleted if the script failed before the replace step
    if os.path.exists(output_file):
        os.remove(output_file)