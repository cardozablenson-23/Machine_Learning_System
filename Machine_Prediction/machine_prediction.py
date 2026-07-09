"""
================================================================================
MACHINE FAILURE PREDICTION 
================================================================================
Predicts Machine Failure directly from sensor readings
Uses simple models: Logistic Regression and Random Forest Classifier
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MACHINE FAILURE PREDICTION - USING ORIGINAL FEATURES")
print("="*80)

# Load dataset
print("\nLoading data...")
df = pd.read_csv('data/ai4i2020.csv')
print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

sensor_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

# ============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

print("\nDataset Info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution (Machine Failure):")
print(df['Machine failure'].value_counts())
failure_rate = df['Machine failure'].mean() * 100
print(f"Failure Rate: {failure_rate:.2f}%")

# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df['Machine failure'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Machine Failure Distribution')
axes[0].set_xlabel('Status (0=Normal, 1=Failure)')
axes[0].set_ylabel('Count')

df['Machine failure'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.2f%%', 
                                          colors=['green', 'red'], explode=[0, 0.1])
axes[1].set_title('Failure Rate')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=150)
print("Saved: target_distribution.png")
plt.close('all')

print("\nSensor Data Distributions:")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Sensor Data Distributions', fontsize=16)

for idx, col in enumerate(sensor_cols):
    row, col_idx = idx // 3, idx % 3
    ax = axes[row, col_idx]
    ax.hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title(col)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

if len(sensor_cols) < 6:
    for idx in range(len(sensor_cols), 6):
        row, col_idx = idx // 3, idx % 3
        fig.delaxes(axes[row, col_idx])

plt.tight_layout()
plt.savefig('sensor_distributions.png', dpi=150)
print("Saved: sensor_distributions.png")
plt.close('all')

print("\nCorrelation with Machine Failure:")
for col in sensor_cols:
    corr = df[col].corr(df['Machine failure'])
    print(f"  {col}: {corr:.4f}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# Use only sensor columns as features
X = df[sensor_cols].copy()
y = df['Machine failure'].copy()

print(f"Features: {list(X.columns)}")
print(f"Target: Machine Failure (0=Normal, 1=Failure)")

# ============================================================================
# SPLIT DATA
# ============================================================================

print("\nSplitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

print("\nTraining set distribution:")
print(y_train.value_counts())
print(f"Failure rate in training: {y_train.mean()*100:.2f}%")

print("\nTest set distribution:")
print(y_test.value_counts())
print(f"Failure rate in test: {y_test.mean()*100:.2f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled")

# ============================================================================
# TRAIN MODELS
# ============================================================================

print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

# Logistic Regression (Simple baseline)
print("\nTraining Logistic Regression...")
start_time = time.time()
lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_time = time.time() - start_time
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
print(f"Logistic Regression trained in {lr_time:.4f} seconds")

# Random Forest (Advanced)
print("\nTraining Random Forest...")
start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    min_samples_split=5,
    class_weight='balanced',
    random_state=42, 
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_time = time.time() - start_time
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
print(f"Random Forest trained in {rf_time:.4f} seconds")

# ============================================================================
# EVALUATE MODELS
# ============================================================================

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

def calculate_metrics(y_true, y_pred, y_proba, name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {auc:.4f}")
    
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 
            'F1': f1, 'ROC-AUC': auc}

print("\nModel Performance:")
print("-"*80)

metrics_lr = calculate_metrics(y_test, y_pred_lr, y_pred_proba_lr, "Logistic Regression")
metrics_rf = calculate_metrics(y_test, y_pred_rf, y_pred_proba_rf, "Random Forest")

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression - Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest - Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
print("\nSaved: confusion_matrices.png")
plt.close('all')

# ============================================================================
# CROSS-VALIDATION
# ============================================================================

print("\n" + "="*80)
print("CROSS-VALIDATION COMPARISON")
print("="*80)

try:
    cv_scores_lr = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='f1')
    cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='f1')
    
    print(f"\nLogistic Regression CV F1: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")
    print(f"Random Forest CV F1:    {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")
except Exception as e:
    print(f"Cross-validation warning: {e}")
    cv_scores_lr = np.array([0])
    cv_scores_rf = np.array([0])

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'CV F1 Mean', 'Train Time (s)'],
    'Logistic Regression': [
        metrics_lr['Accuracy'], 
        metrics_lr['Precision'], 
        metrics_lr['Recall'], 
        metrics_lr['F1'],
        metrics_lr['ROC-AUC'],
        cv_scores_lr.mean() if len(cv_scores_lr) > 0 else 0,
        lr_time
    ],
    'Random Forest': [
        metrics_rf['Accuracy'], 
        metrics_rf['Precision'], 
        metrics_rf['Recall'], 
        metrics_rf['F1'],
        metrics_rf['ROC-AUC'],
        cv_scores_rf.mean() if len(cv_scores_rf) > 0 else 0,
        rf_time
    ]
})
print(comparison_df.to_string(index=False))

# ============================================================================
# DETERMINE BEST MODEL
# ============================================================================

print("\n" + "="*80)
print("BEST MODEL SELECTION")
print("="*80)

if metrics_rf['F1'] > metrics_lr['F1']:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_metrics = metrics_rf
    print("\nRandom Forest has better F1-Score")
elif metrics_lr['F1'] > metrics_rf['F1']:
    best_model = lr_model
    best_model_name = "Logistic Regression"
    best_metrics = metrics_lr
    print("\nLogistic Regression has better F1-Score")
else:
    if lr_time < rf_time:
        best_model = lr_model
        best_model_name = "Logistic Regression"
        best_metrics = metrics_lr
        print("\nBoth models have same F1. Logistic Regression is faster.")
    else:
        best_model = rf_model
        best_model_name = "Random Forest"
        best_metrics = metrics_rf
        print("\nBoth models have same F1. Random Forest is more robust.")

print(f"\nBEST MODEL: {best_model_name}")
print(f"F1-Score: {best_metrics['F1']:.4f}")
print(f"Accuracy: {best_metrics['Accuracy']:.4f}")
print(f"Precision: {best_metrics['Precision']:.4f}")
print(f"Recall: {best_metrics['Recall']:.4f}")

# Feature Importance
print("\n" + "-"*80)
print("FEATURE IMPORTANCE (Random Forest)")
print("-"*80)

feature_importance = pd.DataFrame({
    'Feature': sensor_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['Feature']:35s}: {row['Importance']:.4f}")

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Machine Failure Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("\nSaved: feature_importance.png")
plt.close('all')

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

os.makedirs('models', exist_ok=True)

joblib.dump(lr_model, 'models/logistic_regression.pkl')
joblib.dump(rf_model, 'models/random_forest.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Models saved to 'models/' directory")
print("  - models/logistic_regression.pkl")
print("  - models/random_forest.pkl")
print("  - models/scaler.pkl")

# ============================================================================
# INTERACTIVE PREDICTION
# ============================================================================

print("\n" + "="*80)
print("INTERACTIVE PREDICTION")
print("="*80)

print("\nUnderstanding the Prediction:")
print("  Output: 0 = Normal Operation (Machine is healthy)")
print("  Output: 1 = Failure Risk (Machine may fail soon)")
print("  Probability > 0.5 = High risk of failure")

def get_user_input():
    print("\nEnter sensor readings:")
    print("-" * 50)
    print("Reference ranges:")
    print("  Air temperature [K]: 295 - 310")
    print("  Process temperature [K]: 305 - 320")
    print("  Rotational speed [rpm]: 1000 - 3000")
    print("  Torque [Nm]: 30 - 50")
    print("  Tool wear [min]: 0 - 250")
    print("-" * 50)
    
    try:
        air_temp = float(input("Air temperature [K]: "))
        process_temp = float(input("Process temperature [K]: "))
        speed = float(input("Rotational speed [rpm]: "))
        torque = float(input("Torque [Nm]: "))
        tool_wear = float(input("Tool wear [min]: "))
        
        return np.array([[air_temp, process_temp, speed, torque, tool_wear]])
    except ValueError:
        print("Invalid input! Please enter numeric values.")
        return None

def predict_failure():
    print(f"\nUsing model: {best_model_name}")
    print(f"F1-Score: {best_metrics['F1']:.4f}")
    print(f"Accuracy: {best_metrics['Accuracy']:.4f}")
    
    while True:
        print("\n" + "-"*50)
        print("OPTIONS:")
        print("1. Check for Machine Failure Risk")
        print("2. Exit")
        print("-"*50)
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '2':
            print("\nExiting system...")
            break
        
        elif choice == '1':
            input_data = get_user_input()
            if input_data is None:
                continue
            
            try:
                # Scale input
                input_scaled = scaler.transform(input_data)
                
                # Get prediction and probability
                prediction = best_model.predict(input_scaled)[0]
                probability = best_model.predict_proba(input_scaled)[0][1]
                
                print("\n" + "="*60)
                print("PREDICTION RESULT")
                print("="*60)
                
                if prediction == 0:
                    print("\nSTATUS: NORMAL OPERATION")
                    print(f"Failure Probability: {probability*100:.2f}%")
                    print("  Machine appears to be healthy.")
                    print("  Continue normal operation.")
                else:
                    print("\nSTATUS: FAILURE RISK DETECTED")
                    print(f"Failure Probability: {probability*100:.2f}%")
                    print("  Machine may fail soon!")
                    print("  Recommended: Schedule maintenance immediately.")
                
                # Additional insights
                temp_diff = input_data[0][1] - input_data[0][0]
                power = input_data[0][3] * input_data[0][2] / 60
                wear = input_data[0][4]
                
                print("\nADDITIONAL INSIGHTS:")
                
                if temp_diff > 12:
                    print("  - High temperature differential detected")
                    print("  - May indicate overheating issues")
                elif temp_diff > 8:
                    print("  - Moderate temperature differential")
                    print("  - Monitor temperature trends")
                else:
                    print("  - Normal temperature differential")
                
                if power > 2000:
                    print("  - High power consumption")
                    print("  - Machine under heavy load")
                elif power > 1500:
                    print("  - Moderate power consumption")
                else:
                    print("  - Normal power consumption")
                
                if wear > 200:
                    print("  - High tool wear detected")
                    print("  - Tool replacement needed soon")
                elif wear > 100:
                    print("  - Moderate tool wear")
                    print("  - Monitor tool wear rate")
                else:
                    print("  - Low tool wear")
                
                if probability > 0.7:
                    print("\nHIGH RISK: Immediate action recommended!")
                elif probability > 0.5:
                    print("\nMEDIUM RISK: Schedule maintenance soon")
                else:
                    print("\nLOW RISK: Continue normal operation")
                
                print("="*60)
                
            except Exception as e:
                print(f"Error making prediction: {e}")
        
        else:
            print("Invalid choice. Please enter 1 or 2.")

# ============================================================================
# RUN
# ============================================================================

predict_failure()

print("\n" + "="*80)
print("PREDICTION COMPLETE!")
print("="*80)