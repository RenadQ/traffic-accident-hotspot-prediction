import sys, os, glob, io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

plt.rcParams["figure.dpi"] = 130


USER_PROVIDED_PATH = "" 
def pick_file_local():
    try:
       
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        fpath = filedialog.askopenfilename(title="اختر ملف البيانات (Excel/CSV)",
                                           filetypes=[("Data files","*.xlsx *.csv *.xls")])
        return fpath
    except Exception:
        return ""

def pick_file_colab():
    try:
        from google.colab import files  
        up = files.upload()  
        assert len(up) > 0, "لم يتم اختيار ملف"
        return list(up.keys())[0]
    except Exception:
        return ""

def auto_find_common():
   
    candidates = []
    here = os.getcwd()
    candidates += glob.glob(os.path.join(here, "transformed.*"))
    candidates += glob.glob(os.path.join(here, "US_Accidents.*"))
   
    home = str(Path.home())
    for pat in [r"transformed_data_train_V88*.xls*", r"US_Accidents*.csv", r"transformed.xls*"]:
        candidates += glob.glob(os.path.join(home, "Downloads", pat))
    return candidates[0] if candidates else ""

def load_df_safely():
    
    if USER_PROVIDED_PATH and os.path.exists(USER_PROVIDED_PATH):
        return USER_PROVIDED_PATH

   
    f = pick_file_colab()
    if f and os.path.exists(f):
        return f

    
    f = pick_file_local()
    if f and os.path.exists(f):
        return f

    
    f = auto_find_common()
    if f and os.path.exists(f):
        return f

    raise FileNotFoundError("ما قدرت ألقى الملف تلقائيًا. لو سمحتي: املي USER_PROVIDED_PATH بالمسار الصحيح.")


fpath = load_df_safely()
ext = Path(fpath).suffix.lower()
if ext in [".xlsx",".xls"]:
    df = pd.read_excel(fpath)
elif ext == ".csv":
    df = pd.read_csv(fpath)
else:
    raise ValueError(f"امتداد غير مدعوم: {ext}")

print("✅ Loaded file:", fpath)
print("Shape:", df.shape)
display(df.head(3))


target = "Severity"
assert target in df.columns, f"Target '{target}' غير موجود في الأعمدة: {list(df.columns[:10])} ..."

candidate_features = [
    "Distance(mi)","Visibility(mi)","Temperature(F)","Humidity(%)","Wind_Speed(mph)","Pressure(in)",
    "Hour","Month","Weekday","Weather_Condition","State","Sunrise_Sunset"
]
features = [c for c in candidate_features if c in df.columns]
if not features:
    features = [c for c in df.columns if c != target]
    print("ℹ Using all columns except target (fallback).")

X = df[features].copy()
y = df[target].copy()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except Exception:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
print("Train/Test:", X_train.shape, X_test.shape)

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]
preprocess = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced"), {}),
    "Random Forest (Tuned)": (RandomForestClassifier(random_state=42), {"clf_n_estimators":[150], "clf_max_depth":[20]}),
    "Gradient Boosting (Tuned)": (GradientBoostingClassifier(random_state=42), {"clf_n_estimators":[150], "clflearning_rate":[0.1], "clf_max_depth":[5]})
}

fitted, rows = {}, []
for name, (clf, grid) in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    if grid:
        search = GridSearchCV(pipe, grid, cv=5, scoring="f1_weighted", n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        tuned = "Yes"
    else:
        model = pipe.fit(X_train, y_train)
        tuned = "No"
    fitted[name] = model

    y_pred = model.predict(X_test)
    rows.append({
        "Model": name, "Tuned?": tuned,
        "Accuracy": round(accuracy_score(y_test, y_pred),4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0),4),
        "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0),4),
        "F1 Score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0),4)
    })

results = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
display(results)

out = Path("phase5_outputs"); out.mkdir(exist_ok=True)
results.to_csv(out/"model_results_phase5.csv", index=False)

ax = results.set_index("Model")[["Accuracy","Precision","Recall","F1 Score"]].plot(kind="bar", figsize=(10,6))
plt.title("Figure 5.1 – Model Performance Comparison")
plt.ylabel("Score"); plt.ylim(0,1); plt.grid(axis="y", linestyle="--", alpha=0.5); plt.tight_layout()
plt.savefig(out/"Figure_5_1_Model_Performance_Comparison.png", dpi=200); plt.show()

best_name = results.iloc[0]["Model"]
best = fitted[best_name]
y_pred_best = best.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(cm).plot(cmap="Blues", values_format="d")
plt.title(f"Figure 5.2 – Confusion Matrix ({best_name})"); plt.tight_layout()
plt.savefig(out/"Figure_5_2_Confusion_Matrix.png", dpi=200); plt.show()

def feat_names(prep, num_cols, cat_cols):
    ohe = prep.named_transformers_["cat"] if len(cat_cols) else None
    cats = ohe.get_feature_names_out(cat_cols) if ohe is not None else []
    return list(num_cols) + list(cats)

feat_cols = feat_names(best.named_steps["prep"], num_cols, cat_cols)
if hasattr(best.named_steps["clf"], "feature_importances_"):
    import seaborn as sns
    imp = pd.Series(best.named_steps["clf"].feature_importances_, index=feat_cols).sort_values(ascending=False).head(10)
    plt.figure(figsize=(8,6)); sns.barplot(x=imp.values, y=imp.index)
    plt.title("Figure 5.3 – Top 10 Most Important Features"); plt.xlabel("Importance Score"); plt.ylabel("Feature"); plt.tight_layout()
    plt.savefig(out/"Figure_5_3_Feature_Importance.png", dpi=200); plt.show()
else:
    print(f"⚠ {best_name} لا يدعم feature_importances_. إذا طلع هو الأفضل، اعرضي coefficients أو استخدمي SHAP.")

print("\n✅ DONE. Saved files in:", out.resolve())
for p in out.iterdir(): print("•", p.name)