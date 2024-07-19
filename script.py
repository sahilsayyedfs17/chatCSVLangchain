import pandas as pd

# Function to remove the specified column from the CSV file
def remove_column(input_csv, output_csv, column_name):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Check if the column exists and drop it
    if column_name in df.columns:
        df = df.drop(columns=[column_name])
        print(f"Column '{column_name}' removed successfully.")
    else:
        print(f"Column '{column_name}' not found in the CSV file.")
    
    # Write the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV file saved as '{output_csv}'.")

if __name__ == "__main__":
    input_csv = 'modified_VTVQA_all.csv'  # Replace with your input CSV file path
    output_csv = 'outputVTVQA_all.csv'  # Replace with your desired output CSV file path
    column_name = 'Unnamed: 3'  # Column to be removed
    
    remove_column(input_csv, output_csv, column_name)
