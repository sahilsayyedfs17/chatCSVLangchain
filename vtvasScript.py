import csv

# Input and output file names
input_file = 'vtvas_issues.csv'
output_file = 'vtvas_issues_clean.csv'

# Function to trim description at the first full stop
def trim_description(description):
    if '.' in description:
        return description.split('.')[0] + '.'
    return description

# Read the input CSV and process the descriptions
with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    rows = list(reader)

# Modify the Description field
for row in rows:
    row['Description'] = trim_description(row['Description'])

# Write the output CSV
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Descriptions trimmed and saved to {output_file}")
