import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb  # --- NEW: Import XGBoost ---
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
import time

# --- Global Settings ---
warnings.filterwarnings('ignore')
audit_log = []
start_time = time.time()

# 1. SET YOUR FILENAME
file_name = 'transformed_data_train_V88.xlsx' 

audit_log.append("="*60)
audit_log.append("       MODELING AUDIT REPORT - (RF, GB, and XGBoost)")
audit_log.append("="*60)
audit_log.append(f"PROJECT GOAL: Predict High-Risk Traffic Accident Locations")


# --- 1. Load Data ---

log_entry = f"\n--- 1. Data Loading ---\n"
log_entry += f"Loading data from: '{file_name}'"
print(log_entry)
audit_log.append(log_entry)

try:
    df = pd.read_excel(file_name)
except FileNotFoundError:
    log_entry = f"\n!!! ERROR: The file '{file_name}' was not found. !!!"
    print(log_entry)
    exit()

log_entry = f"Original accident records loaded: {df.shape[0]}"
print(log_entry)
audit_log.append(log_entry)


# --- 2. Feature Engineering (Create Geographic Grid) ---

log_entry = "\n--- 2. Feature Engineering: Creating Grid Cells ---"
print(log_entry)
audit_log.append(log_entry)

# Define grid precision
grid_precision = 2
df['Lat_Grid'] = df['Start_Lat'].round(grid_precision)
df['Lng_Grid'] = df['Start_Lng'].round(grid_precision)

log_entry = f"Grouping {df.shape[0]} accidents into geographic grid cells..."
print(log_entry)
audit_log.append(log_entry)

# Group by the new grid cells
agg_rules = {
    'ID': 'count',
    'Temperature(F)': 'mean',
    'Humidity(%)': 'mean',
    'Visibility(mi)': 'mean',
    'Wind_Speed(mph)': 'mean',
    'Crossing': 'max',
    'Junction': 'max',
    'Traffic_Signal': 'max'
}
grid_df = df.groupby(['Lat_Grid', 'Lng_Grid']).agg(agg_rules).reset_index()

# Rename columns
grid_df.rename(columns={
    'ID': 'Accident_Count',
    'Temperature(F)': 'Mean_Temp',
    'Humidity(%)': 'Mean_Humidity',
    'Visibility(mi)': 'Mean_Visibility',
    'Wind_Speed(mph)': 'Mean_Wind_Speed',
    'Crossing': 'Has_Crossing',
    'Junction': 'Has_Junction',
    'Traffic_Signal': 'Has_Traffic_Signal'
}, inplace=True)

log_entry = f"Created a new dataset of {grid_df.shape[0]} unique grid cells."
print(log_entry)
audit_log.append(log_entry)


# --- 3. Create Target Variable (Is_Hotspot) ---

log_entry = "\n--- 3. Creating Target Variable: 'Is_Hotspot' ---"
print(log_entry)
audit_log.append(log_entry)

# Define a "hotspot" as the top 10% most dangerous cells
threshold = grid_df['Accident_Count'].quantile(0.90)
grid_df['Is_Hotspot'] = (grid_df['Accident_Count'] > threshold).astype(int)

hotspot_count = grid_df['Is_Hotspot'].sum()
non_hotspot_count = grid_df.shape[0] - hotspot_count

log_entry = f"Hotspot Threshold: Cells with > {threshold:.1f} accidents are 'Hotspots'."
log_entry += f"\nTotal Hotspot cells (1):   {hotspot_count}"
log_entry += f"\nTotal Non-Hotspot cells (0): {non_hotspot_count}"
print(log_entry)
audit_log.append(log_entry)


# --- 4. Prepare for Modeling ---

log_entry = "\n--- 4. Final Model Preparation ---"
print(log_entry)
audit_log.append(log_entry)

# Define our new Features (X) and Target (y)
features = [
    'Lat_Grid', 'Lng_Grid',
    'Mean_Temp', 'Mean_Humidity', 'Mean_Visibility', 'Mean_Wind_Speed',
    'Has_Crossing', 'Has_Junction', 'Has_Traffic_Signal'
]
target = 'Is_Hotspot'

X = grid_df[features]
y = grid_df[target]

# Split the GRID data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

log_entry = f"Training samples (grid cells): {X_train.shape[0]}"
log_entry += f"\nTesting samples (grid cells): {X_test.shape[0]}"
print(log_entry)
audit_log.append(log_entry)

# --- NEW: Calculate scale_pos_weight for imbalanced data ---
# This is (count of negative class) / (count of positive class)
# We calculate it from the *training data*
train_non_hotspot_count = (y_train == 0).sum()
train_hotspot_count = (y_train == 1).sum()
scale_pos_weight = train_non_hotspot_count / train_hotspot_count

log_entry = f"\nCalculated 'scale_pos_weight' for XGBoost: {scale_pos_weight:.2f}"
log_entry += "\n(This tells XGBoost to pay more attention to the rare 'Hotspot' class)"
print(log_entry)
audit_log.append(log_entry)


# --- 5. Task 3: Implement Baseline Models ---

log_entry = "\n\n" + "="*50
log_entry += "\n Task 3: Baseline Model Implementation"
log_entry += "\n" + "="*50
print(log_entry)
audit_log.append(log_entry)

# --- Random Forest (Baseline) ---
print("\nTraining Baseline Random Forest...")
rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_train, y_train)
y_pred_rf_base = rf_baseline.predict(X_test)
base_rf_accuracy = accuracy_score(y_test, y_pred_rf_base)

log_entry = "\n\n--- Random Forest (Baseline) ---"
log_entry += f"\nAccuracy: {base_rf_accuracy:.4f}"
log_entry += "\nClassification Report (0=Not_Hotspot, 1=Is_Hotspot):\n"
log_entry += classification_report(y_test, y_pred_rf_base)
print(log_entry)
audit_log.append(log_entry)

# --- Gradient Boosting (Baseline) ---
print("\nTraining Baseline Gradient Boosting...")
gb_baseline = GradientBoostingClassifier(random_state=42)
gb_baseline.fit(X_train, y_train)
y_pred_gb_base = gb_baseline.predict(X_test)
base_gb_accuracy = accuracy_score(y_test, y_pred_gb_base)

log_entry = "\n\n--- Gradient Boosting (Baseline) ---"
log_entry += f"\nAccuracy: {base_gb_accuracy:.4f}"
log_entry += "\nClassification Report (0=Not_Hotspot, 1=Is_Hotspot):\n"
log_entry += classification_report(y_test, y_pred_gb_base)
print(log_entry)
audit_log.append(log_entry)

# --- NEW: XGBoost (Baseline) ---
print("\nTraining Baseline XGBoost...")
# We use the scale_pos_weight even in the baseline
xgb_baseline = xgb.XGBClassifier(
    random_state=42, 
    scale_pos_weight=scale_pos_weight, 
    use_label_encoder=False, 
    eval_metric='logloss'
)
xgb_baseline.fit(X_train, y_train)
y_pred_xgb_base = xgb_baseline.predict(X_test)
base_xgb_accuracy = accuracy_score(y_test, y_pred_xgb_base)

log_entry = "\n\n--- XGBoost (Baseline) ---"
log_entry += f"\nAccuracy: {base_xgb_accuracy:.4f}"
log_entry += "\nClassification Report (0=Not_Hotspot, 1=Is_Hotspot):\n"
log_entry += classification_report(y_test, y_pred_xgb_base)
print(log_entry)
audit_log.append(log_entry)


# --- 6. Task 4: Hyperparameter Tuning & Validation ---

log_entry = "\n\n" + "="*50
log_entry += "\n Task 4: Hyperparameter Tuning & Validation"
log_entry += "\n" + "="*50
print(log_entry)
audit_log.append(log_entry)

# --- Parameter Grids ---
param_grid_rf = {'n_estimators': [100, 150], 'max_depth': [10, 20]}
param_grid_gb = {'n_estimators': [100, 150], 'max_depth': [5, 10], 'learning_rate': [0.05, 0.1]}
# --- NEW: XGBoost Grid ---
param_grid_xgb = {
    'n_estimators': [100, 150],
    'max_depth': [5, 10],
    'learning_rate': [0.05, 0.1]
}

# --- Tuning Random Forest ---
log_entry = f"\n\nTuning Random Forest with GridSearchCV (cv=5)..."
print(log_entry)
audit_log.append(log_entry)
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
log_entry = "Random Forest tuning complete."
log_entry += f"\nBest RF Parameters: {grid_rf.best_params_}"
log_entry += f"\nBest cross-validation accuracy: {grid_rf.best_score_:.4f}"
print(log_entry)
audit_log.append(log_entry)

# --- Tuning Gradient Boosting ---
log_entry = f"\n\nTuning Gradient Boosting with GridSearchCV (cv=5)..."
print(log_entry)
audit_log.append(log_entry)
grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1)
grid_gb.fit(X_train, y_train)
best_gb = grid_gb.best_estimator_
log_entry = "Gradient Boosting tuning complete."
log_entry += f"\nBest GB Parameters: {grid_gb.best_params_}"
log_entry += f"\nBest cross-validation accuracy: {grid_gb.best_score_:.4f}"
print(log_entry)
audit_log.append(log_entry)

# --- NEW: Tuning XGBoost ---
log_entry = f"\n\nTuning XGBoost with GridSearchCV (cv=5)..."
print(log_entry)
audit_log.append(log_entry)
# We must pass the scale_pos_weight to the estimator here as well
grid_xgb = GridSearchCV(
    estimator=xgb.XGBClassifier(
        random_state=42, 
        scale_pos_weight=scale_pos_weight, 
        use_label_encoder=False, 
        eval_metric='logloss'
    ),
    param_grid=param_grid_xgb,
    cv=5,
    scoring='accuracy', # You could also try 'f1_weighted' or 'recall_weighted'
    n_jobs=-1
)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
log_entry = "XGBoost tuning complete."
log_entry += f"\nBest XGB Parameters: {grid_xgb.best_params_}"
log_entry += f"\nBest cross-validation accuracy: {grid_xgb.best_score_:.4f}"
print(log_entry)
audit_log.append(log_entry)


# --- 7. Final Tuned Model Performance ---

log_entry = "\n\n" + "="*50
log_entry += "\n 7. Final Tuned Model Performance & Predictions"
log_entry += "\n" + "="*50
print(log_entry)
audit_log.append(log_entry)

# --- Tuned Random Forest ---
y_pred_rf_tuned = best_rf.predict(X_test)
tuned_rf_accuracy = accuracy_score(y_test, y_pred_rf_tuned)
log_entry = "\n\n--- Random Forest (Tuned) ---"
log_entry += f"\nBaseline Accuracy: {base_rf_accuracy:.4f}"
log_entry += f"\nTuned Accuracy:   {tuned_rf_accuracy:.4f}"
log_entry += f"\nImprovement:      {tuned_rf_accuracy - base_rf_accuracy:+.4f}"
log_entry += "\nTuned Classification Report (0=Not_Hotspot, 1=Is_Hotspot):\n"
log_entry += classification_report(y_test, y_pred_rf_tuned)
print(log_entry)
audit_log.append(log_entry)

# --- Tuned Gradient Boosting ---
y_pred_gb_tuned = best_gb.predict(X_test)
tuned_gb_accuracy = accuracy_score(y_test, y_pred_gb_tuned)
log_entry = "\n\n--- Gradient Boosting (Tuned) ---"
log_entry += f"\nBaseline Accuracy: {base_gb_accuracy:.4f}"
log_entry += f"\nTuned Accuracy:   {tuned_gb_accuracy:.4f}"
log_entry += f"\nImprovement:      {tuned_gb_accuracy - base_gb_accuracy:+.4f}"
log_entry += "\nTuned Classification Report (0=Not_Hotspot, 1=Is_Hotspot):\n"
log_entry += classification_report(y_test, y_pred_gb_tuned)
print(log_entry)
audit_log.append(log_entry)

# --- NEW: Tuned XGBoost ---
y_pred_xgb_tuned = best_xgb.predict(X_test)
tuned_xgb_accuracy = accuracy_score(y_test, y_pred_xgb_tuned)
log_entry = "\n\n--- XGBoost (Tuned) ---"
log_entry += f"\nBaseline Accuracy: {base_xgb_accuracy:.4f}"
log_entry += f"\nTuned Accuracy:   {tuned_xgb_accuracy:.4f}"
log_entry += f"\nImprovement:      {tuned_xgb_accuracy - base_xgb_accuracy:+.4f}"
log_entry += "\nTuned Classification Report (0=Not_Hotspot, 1=Is_Hotspot):\n"
log_entry += classification_report(y_test, y_pred_xgb_tuned)
print(log_entry)
audit_log.append(log_entry)


# --- 8. Sample Predictions & SAVING FULL PREDICTION FILE ---
# --- NOW BASED ON XGBOOST ---

log_entry = "\n\n" + "="*50
log_entry += "\n 8. Sample Predictions & Saving Full Report (from Tuned XGBoost)"
log_entry += "\n" + "="*50
print(log_entry)
audit_log.append(log_entry)

# Create a DataFrame with all features, actuals, and predictions
prediction_df = X_test.copy()
prediction_df['Actual_Is_Hotspot'] = y_test
prediction_df['Predicted_Is_Hotspot'] = y_pred_xgb_tuned # Using XGBoost model

# --- 1. Show a 10-row sample in the console/audit file ---
log_entry = "\nShowing first 10 predictions from the test set of grid cells:\n"
log_entry += prediction_df[['Lat_Grid', 'Lng_Grid', 'Actual_Is_Hotspot', 'Predicted_Is_Hotspot']].head(10).to_string()
print(log_entry)
audit_log.append(log_entry)

# --- 2. Save ALL predictions to a new Excel file ---
prediction_file_name = "full_predictions_report.xlsx"
try:
    prediction_df.to_excel(prediction_file_name, index=False)
    log_entry = f"\nSuccessfully saved ALL {len(prediction_df)} predictions to a separate file: '{prediction_file_name}'"
    print(log_entry)
    audit_log.append(log_entry)
except Exception as e:
    log_entry = f"\nError saving full prediction file: {e}"
    print(log_entry)
    audit_log.append(log_entry)


# --- 9. Generate Visualizations ---
# --- NOW BASED ON XGBOOST ---

log_entry = "\n\n" + "="*50
log_entry += "\n 9. Generating Visualizations (from Tuned XGBoost)"
log_entry += "\n" + "="*50
print(log_entry)
audit_log.append(log_entry)

final_model = best_xgb  # --- CHANGED to best_xgb
model_name = "Tuned XGBoost"
y_pred_final = y_pred_xgb_tuned

# --- 1. Hotspot Map ---
log_entry = "Generating Predicted Hotspot Map..."
print(log_entry)
audit_log.append(log_entry)

plot_df = X_test.copy()
plot_df['Predicted'] = y_pred_final
plot_df['Actual'] = y_test

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=plot_df[plot_df['Predicted'] == 0],
    x='Lng_Grid', y='Lat_Grid',
    color='gray', alpha=0.3, label='Predicted: Not-Hotspot (0)'
)
sns.scatterplot(
    data=plot_df[plot_df['Predicted'] == 1],
    x='Lng_Grid', y='Lat_Grid',
    color='red', marker='X', s=100, label='Predicted: Hotspot (1)'
)
plt.title(f'Map of Predicted Accident Hotspots (from {model_name})')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.tight_layout()
plt.savefig('predicted_hotspot_map.png')
log_entry = "Saved 'predicted_hotspot_map.png'"
print(log_entry)
audit_log.append(log_entry)


# --- 2. Feature Importance Plot ---
log_entry = f"Generating Feature Importance Plot for '{model_name}'..."
print(log_entry)
audit_log.append(log_entry)

# XGBoost has a built-in feature importance
importances = final_model.feature_importances_
feature_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title(f'Feature Importance - {model_name}')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
log_entry = "Saved 'feature_importance.png'"
print(log_entry)
audit_log.append(log_entry)


# --- 3. Confusion Matrix ---
log_entry = f"Generating Confusion Matrix for '{model_name}'..."
print(log_entry)
audit_log.append(log_entry)

cm = confusion_matrix(y_test, y_pred_final)
class_labels = ['Not-Hotspot (0)', 'Is-Hotspot (1)']

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.title(f'Confusion Matrix - {model_name}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
log_entry = "Saved 'confusion_matrix.png'"
print(log_entry)
audit_log.append(log_entry)


# --- 10. Write Smart Audit File ---

log_entry = "\n\n" + "="*50
log_entry += "\n 10. Writing Smart Audit File"
log_entry += "\n" + "="*50
print(log_entry)
audit_log.append(log_entry)

end_time = time.time()
total_time = end_time - start_time
log_entry = f"\nTotal execution time: {total_time:.2f} seconds."
print(log_entry)
audit_log.append(log_entry)

audit_file_name = "model_audit_report.txt"
audit_content = "\n".join(audit_log)

try:
    with open(audit_file_name, "w", encoding="utf-8") as f:
        f.write(audit_content)
    log_entry = f"\nSuccessfully saved smart audit to '{audit_file_name}'"
    print(log_entry)
except Exception as e:
    log_entry = f"\nError saving audit file: {e}"
    print(log_entry)

print("\nAll tasks complete.")