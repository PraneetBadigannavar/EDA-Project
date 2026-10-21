import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src.models.feature_pipeline import YouTubeFeaturePipeline
except ImportError:
    from feature_pipeline import YouTubeFeaturePipeline

class SoftVotingEnsemble:
    """
    Weighted Soft-Voting Ensemble combining probability distributions
    from XGBoost, CatBoost, and Random Forest.
    """
    def __init__(self, models, weights=None):
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            w_sum = sum(weights)
            self.weights = [w / w_sum for w in weights]

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 3))
        for model, w in zip(self.models, self.weights):
            probs += w * model.predict_proba(X)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

def run_pipeline(
    input_path='data/processed/youtube_features.csv',
    output_dir='results/models',
    random_state=42
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("[PIPELINE] TRENDPULSE ML PIPELINE: MODEL TRAINING, TUNING & EVALUATION")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[Step 1] Loading processed dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Total dataset records: {len(df):,}")

    # 2. Target Creation (Tertile split: Low, Medium, High)
    p33 = df['engagement_score'].quantile(0.3333)
    p66 = df['engagement_score'].quantile(0.6666)

    def assign_label(val):
        if val <= p33:
            return 0
        elif val <= p66:
            return 1
        else:
            return 2

    df['target'] = df['engagement_score'].apply(assign_label)
    class_names = ['Low', 'Medium', 'High']
    counts = df['target'].value_counts().sort_index()
    print(f"\nClass Distribution:")
    for idx, name in enumerate(class_names):
        print(f"  - Class {idx} ({name}): {counts[idx]:,} samples ({counts[idx]/len(df)*100:.2f}%)")

    # 3. Stratified Train / Test Holdout Split (80% Train, 20% Test)
    # The held-out test set is NEVER used during pipeline fitting or tuning!
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=df['target']
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    y_train = train_df['target'].values
    y_test = test_df['target'].values
    print(f"\nData Partitioning:")
    print(f"  - Training Set: {len(train_df):,} samples (80.0%)")
    print(f"  - Held-out Test Set: {len(test_df):,} samples (20.0%)")

    # 4. Stratified 5-Fold Cross Validation on Training Data
    print("\n" + "=" * 70)
    print("[STEP 2] 5-FOLD STRATIFIED CROSS-VALIDATION ON TRAINING DATA")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    model_cv_metrics = {
        'Baseline Gradient Boosting': {'acc': [], 'macro_f1': [], 'weighted_f1': []},
        'Baseline Random Forest': {'acc': [], 'macro_f1': [], 'weighted_f1': []},
        'Tuned XGBoost': {'acc': [], 'macro_f1': [], 'weighted_f1': []},
        'CatBoost Classifier': {'acc': [], 'macro_f1': [], 'weighted_f1': []},
        'Soft-Voting Ensemble': {'acc': [], 'macro_f1': [], 'weighted_f1': []}
    }

    fold_idx = 1
    for tr_idx, val_idx in skf.split(train_df, y_train):
        print(f"\n--- Running Fold {fold_idx}/5 ---")
        f_tr_df, f_val_df = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        f_y_tr, f_y_val = y_train[tr_idx], y_train[val_idx]

        # Fit feature pipeline strictly on fold training data
        f_pipe = YouTubeFeaturePipeline(tfidf_max_features=100, smoothing=10.0, random_state=random_state)
        f_X_tr = f_pipe.fit_transform(f_tr_df, f_y_tr)
        f_X_val = f_pipe.transform(f_val_df)

        # Baseline Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=random_state)
        gb.fit(f_X_tr, f_y_tr)
        p_gb = gb.predict(f_X_val)
        model_cv_metrics['Baseline Gradient Boosting']['acc'].append(accuracy_score(f_y_val, p_gb))
        model_cv_metrics['Baseline Gradient Boosting']['macro_f1'].append(precision_recall_fscore_support(f_y_val, p_gb, average='macro')[2])
        model_cv_metrics['Baseline Gradient Boosting']['weighted_f1'].append(precision_recall_fscore_support(f_y_val, p_gb, average='weighted')[2])

        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_split=4, random_state=random_state, n_jobs=-1)
        rf.fit(f_X_tr, f_y_tr)
        p_rf = rf.predict(f_X_val)
        model_cv_metrics['Baseline Random Forest']['acc'].append(accuracy_score(f_y_val, p_rf))
        model_cv_metrics['Baseline Random Forest']['macro_f1'].append(precision_recall_fscore_support(f_y_val, p_rf, average='macro')[2])
        model_cv_metrics['Baseline Random Forest']['weighted_f1'].append(precision_recall_fscore_support(f_y_val, p_rf, average='weighted')[2])

        # Tuned XGBoost with sub-sampling & regularization
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.06,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        xgb.fit(f_X_tr, f_y_tr)
        p_xgb = xgb.predict(f_X_val)
        model_cv_metrics['Tuned XGBoost']['acc'].append(accuracy_score(f_y_val, p_xgb))
        model_cv_metrics['Tuned XGBoost']['macro_f1'].append(precision_recall_fscore_support(f_y_val, p_xgb, average='macro')[2])
        model_cv_metrics['Tuned XGBoost']['weighted_f1'].append(precision_recall_fscore_support(f_y_val, p_xgb, average='weighted')[2])

        # CatBoost Classifier
        cat = CatBoostClassifier(iterations=350, learning_rate=0.07, depth=6, random_seed=random_state, verbose=0)
        cat.fit(f_X_tr, f_y_tr)
        p_cat = cat.predict(f_X_val)
        model_cv_metrics['CatBoost Classifier']['acc'].append(accuracy_score(f_y_val, p_cat))
        model_cv_metrics['CatBoost Classifier']['macro_f1'].append(precision_recall_fscore_support(f_y_val, p_cat, average='macro')[2])
        model_cv_metrics['CatBoost Classifier']['weighted_f1'].append(precision_recall_fscore_support(f_y_val, p_cat, average='weighted')[2])

        # Soft-Voting Ensemble
        prob_xgb = xgb.predict_proba(f_X_val)
        prob_cat = cat.predict_proba(f_X_val)
        prob_rf = rf.predict_proba(f_X_val)
        prob_ens = (prob_xgb * 0.45 + prob_cat * 0.35 + prob_rf * 0.20)
        p_ens = np.argmax(prob_ens, axis=1)
        model_cv_metrics['Soft-Voting Ensemble']['acc'].append(accuracy_score(f_y_val, p_ens))
        model_cv_metrics['Soft-Voting Ensemble']['macro_f1'].append(precision_recall_fscore_support(f_y_val, p_ens, average='macro')[2])
        model_cv_metrics['Soft-Voting Ensemble']['weighted_f1'].append(precision_recall_fscore_support(f_y_val, p_ens, average='weighted')[2])

        print(f"  Fold {fold_idx} Accuracies -> GB: {model_cv_metrics['Baseline Gradient Boosting']['acc'][-1]:.4f} | RF: {model_cv_metrics['Baseline Random Forest']['acc'][-1]:.4f} | XGB: {model_cv_metrics['Tuned XGBoost']['acc'][-1]:.4f} | CatBoost: {model_cv_metrics['CatBoost Classifier']['acc'][-1]:.4f} | Ensemble: {model_cv_metrics['Soft-Voting Ensemble']['acc'][-1]:.4f}")
        fold_idx += 1

    print("\n" + "=" * 70)
    print("[SUMMARY] 5-FOLD CROSS VALIDATION SUMMARY (MEAN +/- STD)")
    print("=" * 70)
    cv_summary = {}
    for m_name, mets in model_cv_metrics.items():
        acc_mean, acc_std = np.mean(mets['acc']), np.std(mets['acc'])
        f1_mean, f1_std = np.mean(mets['macro_f1']), np.std(mets['macro_f1'])
        wf1_mean = np.mean(mets['weighted_f1'])
        cv_summary[m_name] = {
            'cv_accuracy_mean': float(acc_mean),
            'cv_accuracy_std': float(acc_std),
            'cv_macro_f1_mean': float(f1_mean),
            'cv_macro_f1_std': float(f1_std),
            'cv_weighted_f1_mean': float(wf1_mean)
        }
        print(f"{m_name:28s} | Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}% | Macro F1: {f1_mean:.4f} ± {f1_std:.4f}")

    # 5. Fit Pipeline on Entire Training Set & Evaluate on Untouched Test Set
    print("\n" + "=" * 70)
    print("[STEP 3] FINAL EVALUATION ON UNTOUCHED HELD-OUT TEST SET (N = 3,472)")
    print("=" * 70)

    final_pipeline = YouTubeFeaturePipeline(tfidf_max_features=100, smoothing=10.0, random_state=random_state)
    X_train_full = final_pipeline.fit_transform(train_df, y_train)
    X_test_full = final_pipeline.transform(test_df)

    # Train Final Models
    print("Training final models on 13,885 training records...")
    final_rf = RandomForestClassifier(n_estimators=250, max_depth=16, min_samples_split=4, random_state=random_state, n_jobs=-1)
    final_rf.fit(X_train_full, y_train)

    final_xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    final_xgb.fit(X_train_full, y_train)

    final_cat = CatBoostClassifier(iterations=350, learning_rate=0.07, depth=6, random_seed=random_state, verbose=0)
    final_cat.fit(X_train_full, y_train)

    final_ensemble = SoftVotingEnsemble(
        models=[final_xgb, final_cat, final_rf],
        weights=[0.45, 0.35, 0.20]
    )

    models_to_eval = {
        'Random Forest (Tuned)': final_rf,
        'CatBoost Classifier': final_cat,
        'XGBoost Classifier': final_xgb,
        'Soft-Voting Ensemble': final_ensemble
    }

    test_results = {}
    predictions_dict = {}

    for m_name, model in models_to_eval.items():
        y_pred = model.predict(X_test_full)
        predictions_dict[m_name] = y_pred
        acc = accuracy_score(y_test, y_pred)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        test_results[m_name] = {
            'test_accuracy': float(acc),
            'test_macro_precision': float(prec_m),
            'test_macro_recall': float(rec_m),
            'test_macro_f1': float(f1_m),
            'test_weighted_f1': float(f1_w),
            'confusion_matrix': cm.tolist()
        }
        print(f"\n[{m_name}] Test Set Performance:")
        print(f"  Accuracy:         {acc*100:.2f}%")
        print(f"  Macro F1-Score:   {f1_m:.4f}")
        print(f"  Weighted F1-Score:{f1_w:.4f}")

    # Best Model Selection
    best_model_name = max(test_results, key=lambda k: test_results[k]['test_accuracy'])
    print("\n" + "=" * 70)
    print(f"[BEST MODEL] TOP PERFORMING MODEL: {best_model_name}")
    print(f"   Final Test Accuracy: {test_results[best_model_name]['test_accuracy']*100:.2f}%")
    print(f"   Final Macro F1:      {test_results[best_model_name]['test_macro_f1']:.4f}")
    print("=" * 70)

    # Classification Report of Best Model
    print(f"\nDetailed Classification Report for {best_model_name}:")
    print(classification_report(y_test, predictions_dict[best_model_name], target_names=class_names, digits=4))

    # 6. Generate Controlled Feature Ablation Study
    print("\n" + "=" * 70)
    print("[STEP 4] CONTROLLED FEATURE ABLATION STUDY")
    print("=" * 70)

    # Let's run ablation configurations
    ablation_results = {}

    # Config 1: Original 8 numeric features + Category One-Hot (Baseline GB)
    orig_cols = ['title_length', 'title_word_count', 'description_length', 'upload_hour', 'upload_day_num', 'upload_month', 'is_weekend', 'video_age_days']
    cat_enc_orig = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_tr_c1 = np.hstack([train_df[orig_cols].values, cat_enc_orig.fit_transform(train_df[['category_id']])])
    X_te_c1 = np.hstack([test_df[orig_cols].values, cat_enc_orig.transform(test_df[['category_id']])])
    m_c1 = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=random_state)
    m_c1.fit(X_tr_c1, y_train)
    p_c1 = m_c1.predict(X_te_c1)
    ablation_results['1. Original Features (Baseline GB)'] = {
        'features': '8 Numerical + Category ID',
        'model': 'GradientBoosting (n=100, d=5)',
        'test_accuracy': float(accuracy_score(y_test, p_c1)),
        'test_macro_f1': float(precision_recall_fscore_support(y_test, p_c1, average='macro')[2])
    }

    # Config 2: Original + Style & Cyclical Features
    style_cols = orig_cols + ['title_caps_ratio', 'has_exclamation', 'has_question', 'has_digits', 'title_character_density', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    X_tr_c2 = np.hstack([train_df[style_cols].values, cat_enc_orig.transform(train_df[['category_id']])])
    X_te_c2 = np.hstack([test_df[style_cols].values, cat_enc_orig.transform(test_df[['category_id']])])
    m_c2 = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=random_state)
    m_c2.fit(X_tr_c2, y_train)
    p_c2 = m_c2.predict(X_te_c2)
    ablation_results['2. + Style & Cyclical Features'] = {
        'features': '17 Numerical/Temporal + Category ID',
        'model': 'GradientBoosting',
        'test_accuracy': float(accuracy_score(y_test, p_c2)),
        'test_macro_f1': float(precision_recall_fscore_support(y_test, p_c2, average='macro')[2])
    }

    # Config 3: + Channel Authority (Freq + OOF Target Enc)
    ch_freq_tr = train_df['channel_title'].map(final_pipeline.channel_counts).fillna(1).values.reshape(-1, 1)
    ch_te_tr = np.array([final_pipeline.channel_target_map.get(c, final_pipeline.global_target_mean) for c in train_df['channel_title']]).reshape(-1, 1)
    ch_freq_te = test_df['channel_title'].map(final_pipeline.channel_counts).fillna(1).values.reshape(-1, 1)
    ch_te_te = np.array([final_pipeline.channel_target_map.get(c, final_pipeline.global_target_mean) for c in test_df['channel_title']]).reshape(-1, 1)
    X_tr_c3 = np.hstack([X_tr_c2, ch_freq_tr, ch_te_tr])
    X_te_c3 = np.hstack([X_te_c2, ch_freq_te, ch_te_te])
    m_c3 = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=random_state)
    m_c3.fit(X_tr_c3, y_train)
    p_c3 = m_c3.predict(X_te_c3)
    ablation_results['3. + Channel Authority (Freq + OOF Target)'] = {
        'features': '17 Num/Temporal + Channel Authority + Category ID',
        'model': 'GradientBoosting',
        'test_accuracy': float(accuracy_score(y_test, p_c3)),
        'test_macro_f1': float(precision_recall_fscore_support(y_test, p_c3, average='macro')[2])
    }

    # Config 4: + Title TF-IDF Unigrams & Bigrams (100)
    m_c4 = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=random_state)
    m_c4.fit(X_train_full, y_train)
    p_c4 = m_c4.predict(X_test_full)
    ablation_results['4. + Title TF-IDF (100 n-grams)'] = {
        'features': 'All Features (Num + Channel + Cat + TF-IDF)',
        'model': 'GradientBoosting',
        'test_accuracy': float(accuracy_score(y_test, p_c4)),
        'test_macro_f1': float(precision_recall_fscore_support(y_test, p_c4, average='macro')[2])
    }

    # Config 5: All Features with Default XGBoost
    xgb_def = XGBClassifier(random_state=random_state, eval_metric='mlogloss')
    xgb_def.fit(X_train_full, y_train)
    p_c5 = xgb_def.predict(X_test_full)
    ablation_results['5. All Features + Default XGBoost'] = {
        'features': 'All Features',
        'model': 'XGBoost (Default)',
        'test_accuracy': float(accuracy_score(y_test, p_c5)),
        'test_macro_f1': float(precision_recall_fscore_support(y_test, p_c5, average='macro')[2])
    }

    # Config 6: Tuned XGBoost
    ablation_results['6. Tuned XGBoost (depth=6, lr=0.06, sub=0.8)'] = {
        'features': 'All Features',
        'model': 'XGBoost (Tuned)',
        'test_accuracy': float(test_results['XGBoost Classifier']['test_accuracy']),
        'test_macro_f1': float(test_results['XGBoost Classifier']['test_macro_f1'])
    }

    # Config 7: CatBoost Classifier
    ablation_results['7. CatBoost Classifier (iter=350, d=6)'] = {
        'features': 'All Features',
        'model': 'CatBoost',
        'test_accuracy': float(test_results['CatBoost Classifier']['test_accuracy']),
        'test_macro_f1': float(test_results['CatBoost Classifier']['test_macro_f1'])
    }

    # Config 8: Soft-Voting Ensemble
    ablation_results['8. Soft-Voting Ensemble (XGB+Cat+RF)'] = {
        'features': 'All Features',
        'model': 'Soft-Voting Ensemble',
        'test_accuracy': float(test_results['Soft-Voting Ensemble']['test_accuracy']),
        'test_macro_f1': float(test_results['Soft-Voting Ensemble']['test_macro_f1'])
    }

    print("\nAblation Experiment Summary:")
    for k, v in ablation_results.items():
        print(f"  {k:50s} -> Acc: {v['test_accuracy']*100:.2f}% | F1: {v['test_macro_f1']:.4f}")

    # 7. Visualizations
    print("\n[Step 5] Generating Visual Evaluation Figures...")
    
    # 7A: Confusion Matrices Heatmap
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    model_keys = ['Random Forest (Tuned)', 'CatBoost Classifier', 'XGBoost Classifier', 'Soft-Voting Ensemble']
    for idx, (m_key, ax) in enumerate(zip(model_keys, axes)):
        cm = np.array(test_results[m_key]['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
        ax.set_title(f"{m_key}\nAcc: {test_results[m_key]['test_accuracy']*100:.2f}% | F1: {test_results[m_key]['test_macro_f1']:.3f}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrices_enhanced.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  - Saved confusion matrices to {cm_path}")

    # 7B: Feature Importance for XGBoost
    importances = final_xgb.feature_importances_
    feat_names = final_pipeline.feature_names
    fi_df = pd.DataFrame({'feature': feat_names[:len(importances)], 'importance': importances}).sort_values('importance', ascending=False).head(20)

    plt.figure(figsize=(11, 7))
    sns.barplot(data=fi_df, x='importance', y='feature', hue='feature', palette='viridis', legend=False)
    plt.title('Top 20 Most Predictive Features (Tuned XGBoost)', fontsize=14, fontweight='bold')
    plt.xlabel('XGBoost Relative Gain / Feature Importance')
    plt.tight_layout()
    fi_path = os.path.join(output_dir, 'feature_importance_enhanced.png')
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"  - Saved feature importance plot to {fi_path}")

    # 8. Save Metrics & Sample Predictions
    metrics_export = {
        'cross_validation_5fold': cv_summary,
        'held_out_test_metrics': test_results,
        'best_model': best_model_name,
        'accuracy_gain_vs_baseline_pct': float((test_results[best_model_name]['test_accuracy'] - ablation_results['1. Original Features (Baseline GB)']['test_accuracy']) * 100)
    }
    with open(os.path.join(output_dir, 'model_comparison_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_export, f, indent=4)

    with open(os.path.join(output_dir, 'ablation_study.json'), 'w', encoding='utf-8') as f:
        json.dump(ablation_results, f, indent=4)

    # Sample Predictions CSV
    sample_df = test_df[['video_id', 'title', 'channel_title', 'category_name', 'views', 'likes', 'comment_count', 'engagement_score', 'target']].copy()
    sample_df['true_class'] = [class_names[t] for t in sample_df['target']]
    sample_df['xgb_pred'] = [class_names[p] for p in predictions_dict['XGBoost Classifier']]
    sample_df['ensemble_pred'] = [class_names[p] for p in predictions_dict['Soft-Voting Ensemble']]
    sample_df.head(150).to_csv(os.path.join(output_dir, 'sample_predictions_enhanced.csv'), index=False)
    print(f"  - Saved 150 test sample predictions to {os.path.join(output_dir, 'sample_predictions_enhanced.csv')}")

    # 9. Artifact Serialization
    print("\n[Step 6] Serializing Trained Models & Preprocessing Objects...")
    joblib.dump(final_pipeline, os.path.join(output_dir, 'feature_pipeline.joblib'))
    joblib.dump(final_xgb, os.path.join(output_dir, 'best_model.joblib'))
    joblib.dump(final_ensemble, os.path.join(output_dir, 'ensemble_model.joblib'))
    print(f"  - Serialized pipeline to {os.path.join(output_dir, 'feature_pipeline.joblib')}")
    print(f"  - Serialized best model to {os.path.join(output_dir, 'best_model.joblib')}")
    print(f"  - Serialized ensemble to {os.path.join(output_dir, 'ensemble_model.joblib')}")

    print("\n" + "=" * 70)
    print("[COMPLETE] ML PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return metrics_export

if __name__ == '__main__':
    run_pipeline()