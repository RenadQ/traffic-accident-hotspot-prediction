#Auther Renad Al kahtani (Phase4_V08)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
FILE_NAME = "transformed_data_train_V88.xlsx"
OUTPUT_FOLDER = "full_visualizations_V09"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# === LOAD DATA ===
df = pd.read_excel(FILE_NAME)
n_rows, n_cols = df.shape
print(f"✅ Loaded file: {FILE_NAME} with {n_rows} rows and {n_cols} columns.")

summary_lines = []
summary_lines.append(f"Dataset Overview:\n- Rows: {n_rows}\n- Columns: {n_cols}\n")
summary_lines.append("- Missing values per column:")
summary_lines.append(df.isnull().sum().to_string())
summary_lines.append("\n")

# === Convert numeric-like columns stored as text/custom ===
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# === Convert datetime columns ===
datetime_cols = [col for col in df.columns if 'Time' in col]
for col in datetime_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Extract time features if Start_Time exists
if 'Start_Time' in df.columns:
    df['Year'] = df['Start_Time'].dt.year
    df['Month'] = df['Start_Time'].dt.month
    df['Day'] = df['Start_Time'].dt.day
    df['Hour'] = df['Start_Time'].dt.hour
    df['Weekday'] = df['Start_Time'].dt.dayofweek

# === Identify numeric, categorical, boolean columns ===
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
boolean_cols = [col for col in df.columns if df[col].dropna().isin([0,1]).all()]

# === NUMERIC SUMMARY AND OUTLIERS ===
for col in numeric_cols:
    desc = df[col].describe()
    summary_lines.append(f"Numeric Summary: {col}")
    summary_lines.append(desc.to_string())
    
    # Outlier detection using IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    summary_lines.append(f"Outliers in {col}: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)\n")
    
    # Histogram
    plt.figure(figsize=(8,5))
    sns.histplot(df[col].dropna(), bins=30, kde=True, color='skyblue')
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/hist_{col}.png")
    plt.close()
    
    # Boxplot
    plt.figure(figsize=(6,5))
    sns.boxplot(y=df[col].dropna(), color='coral')
    plt.title(f"Box Plot: {col}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/box_{col}.png")
    plt.close()

# === CATEGORICAL SUMMARY ===
for col in categorical_cols:
    counts = df[col].value_counts()
    summary_lines.append(f"Categorical Summary: {col}")
    summary_lines.append(counts.to_string())
    
    if df[col].nunique() < 20:
        plt.figure(figsize=(8,5))
        sns.countplot(y=col, data=df, order=counts.index, palette="viridis")
        plt.title(f"Count Plot: {col}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/count_{col}.png")
        plt.close()

# === BOOLEAN FEATURES ===
for col in boolean_cols:
    counts = df[col].value_counts()
    summary_lines.append(f"Boolean Feature Summary: {col}")
    summary_lines.append(counts.to_string())
    
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, data=df, palette="Set2")
    plt.title(f"Boolean Count: {col}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/boolean_{col}.png")
    plt.close()

# === CORRELATION HEATMAP ===
if len(numeric_cols) >= 2:
    plt.figure(figsize=(12,10))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/correlation_heatmap.png")
    plt.close()
    summary_lines.append("- Heatmap shows correlations among numeric variables.")

# === SCATTER PLOTS vs Severity ===
if 'Severity' in df.columns:
    for col in numeric_cols:
        if col != 'Severity':
            plt.figure(figsize=(8,5))
            sns.scatterplot(x=col, y='Severity', data=df, alpha=0.6, color='teal')
            plt.title(f"Scatter Plot: {col} vs Severity")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_FOLDER}/scatter_{col}_Severity.png")
            plt.close()

# === TIME-BASED VISUALIZATIONS ===
time_cols = ['Hour','Day','Month','Weekday','Year']
for col in time_cols:
    if col in df.columns:
        plt.figure(figsize=(8,5))
        sns.countplot(x=col, data=df, palette="coolwarm")
        plt.title(f"Accidents by {col}")
        plt.xlabel(col)
        plt.ylabel("Number of Accidents")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/time_{col}.png")
        plt.close()
        
        # Summary per time
        summary_lines.append(f"Accidents by {col}:")
        summary_lines.append(df[col].value_counts().sort_index().to_string())
        summary_lines.append("\n")

# === YEARLY & MONTHLY COMPARISON ===
if 'Year' in df.columns and 'Month' in df.columns:
    plt.figure(figsize=(12,6))
    sns.countplot(x='Month', hue='Year', data=df, palette='tab20')
    plt.title("Monthly Accidents Comparison by Year")
    plt.xlabel("Month")
    plt.ylabel("Number of Accidents")
    plt.legend(title='Year', bbox_to_anchor=(1.05,1))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/accidents_per_month_by_year.png")
    plt.close()
    
    summary_lines.append("- Monthly accidents by year shows seasonal trends and yearly comparison.")

# === WEATHER CONDITIONS BY YEAR ===
if 'Weather_Condition' in df.columns and 'Year' in df.columns:
    plt.figure(figsize=(12,6))
    sns.countplot(y='Weather_Condition', hue='Year', data=df[df['Year'].notna()],
                  order=df['Weather_Condition'].value_counts().index, palette='Paired')
    plt.title("Weather Conditions per Year")
    plt.xlabel("Number of Accidents")
    plt.legend(title='Year', bbox_to_anchor=(1.05,1))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/weather_per_year.png")
    plt.close()
    
    # Summary of weather vs Severity
    summary_lines.append("Weather conditions by year (counts):")
    summary_lines.append(df.groupby(['Year','Weather_Condition']).size().to_string())

# === OUTLIER DETECTION PER YEAR ===
for col in numeric_cols:
    if 'Year' in df.columns:
        plt.figure(figsize=(12,6))
        sns.boxplot(x='Year', y=col, data=df, palette='Set3')
        plt.title(f"{col} Distribution by Year (Outliers highlighted)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/box_{col}_by_year.png")
        plt.close()
        
        summary_lines.append(f"{col} boxplot by year shows outliers per year.")

# === GEOSPATIAL SCATTER ===
for lat, lng, label in [('Start_Lat','Start_Lng','Start'), ('End_Lat','End_Lng','End')]:
    if lat in df.columns and lng in df.columns:
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=lng, y=lat, data=df, alpha=0.4, s=20, color='red')
        plt.title(f"{label} Location Scatter")
        plt.xlabel(f"{label} Longitude")
        plt.ylabel(f"{label} Latitude")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/scatter_{label}_Location.png")
        plt.close()
        
        summary_lines.append(f"- {label} coordinates scatter shows spatial distribution of accidents.")

# === SCATTER Distance vs Severity PER YEAR ===
if 'Severity' in df.columns and 'Distance(mi)' in df.columns and 'Year' in df.columns:
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='Distance(mi)', y='Severity', hue='Year', data=df, alpha=0.5, palette='tab10')
    plt.title("Distance vs Severity per Year")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/scatter_distance_severity_by_year.png")
    plt.close()
    
    summary_lines.append("- Scatter of Distance vs Severity per year shows trends over time.")

# === SAVE FULL AUDIT/REPORT ===
summary_file = os.path.join(OUTPUT_FOLDER, "EDA_audit_report.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("=== Full EDA Audit & Report for Accident Dataset ===\n\n")
    f.write("\n".join(summary_lines))

print(f"✅ All visualizations saved in '{OUTPUT_FOLDER}'.")
print(f"✅ Full audit report saved as '{summary_file}'.")
