import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def train_and_evaluate(input_path='data/processed/youtube_features.csv', output_dir='results/models'):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading feature dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    # -------------------------------------------------------------
    # 1. Target Label Creation (Engagement Class: Low, Medium, High)
    # -------------------------------------------------------------
    p33 = df['engagement_score'].quantile(0.3333)
    p66 = df['engagement_score'].quantile(0.6666)
    
    def assign_label(val):
        if val <= p33:
            return 'Low'
        elif val <= p66:
            return 'Medium'
        else:
            return 'High'
            
    df['engagement_label'] = df['engagement_score'].apply(assign_label)
    label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df['target'] = df['engagement_label'].map(label_mapping)
    
    print("Class distribution:")
    print(df['engagement_label'].value_counts())
    
    # -------------------------------------------------------------
    # 2. Feature Selection & Preprocessing
    # -------------------------------------------------------------
    numeric_features = [
        'title_length',
        'title_word_count',
        'description_length',
        'upload_hour',
        'upload_day_num',
        'upload_month',
        'is_weekend',
        'video_age_days'
    ]
    
    # One-hot encode category_id
    cat_dummies = pd.get_dummies(df['category_id'], prefix='cat', drop_first=True)
    
    X = pd.concat([df[numeric_features], cat_dummies], axis=1)
    y = df['target']
    
    # Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling for Linear Models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), X_train_scaled, X_test_scaled),
        'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42), X_train, X_test),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42), X_train, X_test)
    }
    
    metrics_summary = {}
    class_names = ['Low', 'Medium', 'High']
    
    plt.figure(figsize=(16, 5))
    
    for idx, (m_name, (model, tr_data, te_data)) in enumerate(models.items(), 1):
        print(f"\nTraining {m_name}...")
        model.fit(tr_data, y_train)
        y_pred = model.predict(te_data)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        metrics_summary[m_name] = {
            'accuracy': float(acc),
            'precision_weighted': float(prec),
            'recall_weighted': float(rec),
            'f1_weighted': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        print(f"[{m_name}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
        # Plot confusion matrix
        plt.subplot(1, 3, idx)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{m_name}\nAcc: {acc:.3f} | F1: {f1:.3f}", fontsize=11, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Feature Importance for Random Forest
    # -------------------------------------------------------------
    rf_model = models['Random Forest'][0]
    importances = rf_model.feature_importances_
    feat_names = X.columns
    fi_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False).head(12)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df, x='importance', y='feature', hue='feature', palette='viridis', legend=False)
    plt.title('Top 12 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Gini Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_rf.png'), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Generate Sample Predictions CSV for Demonstration
    # -------------------------------------------------------------
    test_indices = y_test.index
    sample_df = df.loc[test_indices, ['video_id', 'title', 'channel_title', 'category_name', 'views', 'engagement_score', 'engagement_label']].copy()
    sample_df['rf_predicted_label'] = [class_names[p] for p in models['Random Forest'][0].predict(X_test)]
    sample_df['lr_predicted_label'] = [class_names[p] for p in models['Logistic Regression'][0].predict(X_test_scaled)]
    
    sample_df.head(100).to_csv(os.path.join(output_dir, 'sample_predictions.csv'), index=False)
    
    # Save Metrics JSON
    with open(os.path.join(output_dir, 'model_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"\nML Pipeline Completed! Metrics & Visualizations saved to {output_dir}.")
    return metrics_summary

if __name__ == '__main__':
    train_and_evaluate()