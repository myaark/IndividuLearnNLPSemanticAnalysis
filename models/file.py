import pandas as pd

# Load the CSV file
file_path = "goemotionsmodified.csv"  # Update with your actual file path
df = pd.read_csv(file_path)  # Use read_csv instead of read_excel

# Define the columns to check
columns_to_check = ["anger", "annoyance", "neutral", "joy", "optimism"]

# Drop rows where all specified columns are zero
df_cleaned = df.loc[~(df[columns_to_check] == 0).all(axis=1)]

# Save the cleaned data back to CSV
output_path = "goemotionsfinal.csv"  # Update with desired output file name
df_cleaned.to_csv(output_path, index=False)  # Use to_csv instead of to_excel

print(f"Rows removed. Cleaned file saved as {output_path}")
