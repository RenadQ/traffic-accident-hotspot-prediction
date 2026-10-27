import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------
# Step 1: Load Excel
# ------------------------------
input_file = "US_Accidents_March23_sampled_10k_cleaned.xlsx"
df = pd.read_excel(input_file)
print(f"[Step 1] Loaded Excel file: {input_file}")
print(f"Shape: {df.shape}\n")

# ------------------------------
# Step 2: Clean trailing separators
# ------------------------------
df = df.applymap(lambda x: str(x).replace(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;", "") if isinstance(x, str) else x)
print("[Step 2] Removed trailing separators\n")

# ------------------------------
# Step 3: Drop useless column
# ------------------------------
if 'Precipitation(in)' in df.columns:
    df = df.drop(columns=['Precipitation(in)'])
    print("[Step 3] Dropped 'Precipitation(in)'\n")

# ------------------------------
# Step 4: Handle boolean columns
# ------------------------------
bool_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
             'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
             'Traffic_Signal', 'Turning_Loop']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].map({'TRUE':1, 'FALSE':0}).fillna(0)
print("[Step 4] Converted boolean columns to numeric\n")

# ------------------------------
# Step 5: Handle datetime columns
# ------------------------------
datetime_cols = ['Start_Time', 'End_Time', 'Weather_Timestamp']
for col in datetime_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
print("[Step 5] Converted datetime columns to datetime objects\n")

# ------------------------------
# Step 6: Ensure numeric columns
# ------------------------------
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print(f"[Step 6] Numeric columns ensured: {numeric_cols}\n")

# ------------------------------
# Step 7: Row-wise Train/Test split
# ------------------------------
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
print(f"[Step 7] Split rows into train ({train_df.shape}) and test ({test_df.shape})\n")

# ------------------------------
# Step 8: Save Excel files
# ------------------------------
max_excel_cols = 16384
train_df.iloc[:, :max_excel_cols].to_excel("transformed_data_train_V88.xlsx", index=False)
test_df.iloc[:, :max_excel_cols].to_excel("transformed_data_test_V88.xlsx", index=False)
print("[Step 8] Saved train and test Excel files\n")

# ------------------------------
# Step 9: Create summary table
# ------------------------------
summary = {
    'Total rows': [df.shape[0]],
    'Train rows': [train_df.shape[0]],
    'Test rows': [test_df.shape[0]],
    'Total columns': [df.shape[1]],
    'Boolean columns converted': [len([c for c in bool_cols if c in df.columns])],
    'Datetime columns converted': [len([c for c in datetime_cols if c in df.columns])]
}
summary_df = pd.DataFrame(summary)
summary_df.to_excel("transformed_data_summary_V88.xlsx", index=False)
print("[Step 9] Saved summary table\n")
print(summary_df)
