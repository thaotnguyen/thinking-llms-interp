#!/usr/bin/env python3
"""
Text Mining Analysis with CORRECT Model-Specific Difficulty Definition

Uses the 10-run data from /home/ttn/Development/bmj/analysis_runs/**/round1_10/*.json

Difficulty per model is defined as:
- very_easy: Model got it right 10/10 times
- easy: Model got it right 8-9/10 times  
- medium: Model got it right 3-7/10 times
- hard: Model got it right 1-2/10 times
- very_hard: Model got it right 0/10 times
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def extract_thinking_content(response_text):
    """Extract content between <think> tags"""
    if not response_text or len(response_text) < 10:
        return ""
    
    match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        if len(content) < 50:
            return response_text
        return content
    return response_text


def extract_base_case_id(question_id):
    """Extract base case ID from question_id like PMC5728002_3 -> PMC5728002"""
    if '_' in question_id:
        return question_id.rsplit('_', 1)[0]
    return question_id


def load_round1_10_data(base_dir="/home/ttn/Development/bmj/analysis_runs"):
    """Load all graded JSON files from round1_10 directories"""
    base_path = Path(base_dir)
    graded_files = list(base_path.glob("**/round1_10/*.graded.json"))
    
    all_data = []
    model_names = []
    
    for file_path in graded_files:
        # Extract model name from path
        parts = file_path.parts
        model_idx = parts.index('analysis_runs') + 1
        model_name = parts[model_idx]
        
        if model_name not in model_names:
            model_names.append(model_name)
        
        print(f"Loading {model_name}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Add model name and extract thinking
        for item in data:
            item['model'] = model_name
            item['thinking_text'] = extract_thinking_content(item.get('full_response', ''))
            item['base_case_id'] = extract_base_case_id(item['question_id'])
            all_data.append(item)
    
    print(f"\nLoaded {len(all_data)} responses from {len(model_names)} models:")
    print(f"Models: {', '.join(model_names)}")
    
    return pd.DataFrame(all_data), model_names


def calculate_model_specific_difficulty_correct(df):
    """
    Calculate difficulty per BASE CASE per MODEL based on success rate out of 10 attempts.
    
    Difficulty categories:
    - very_easy: 10/10 correct
    - easy: 8-9/10 correct
    - medium: 3-7/10 correct
    - hard: 1-2/10 correct
    - very_hard: 0/10 correct
    """
    
    # Group by model and base_case_id, calculate success rate
    case_stats = df.groupby(['model', 'base_case_id']).agg({
        'is_correct': ['sum', 'count'],
        'question': 'first',
        'gold_answer': 'first',
    }).reset_index()
    
    case_stats.columns = ['model', 'base_case_id', 'num_correct', 'num_attempts', 
                          'question', 'gold_answer']
    
    def categorize_difficulty(num_correct):
        """Categorize based on number correct out of 10"""
        if num_correct == 10:
            return 'very_easy'
        elif num_correct >= 8:
            return 'easy'
        elif num_correct >= 3:
            return 'medium'
        elif num_correct >= 1:
            return 'hard'
        else:
            return 'very_hard'
    
    case_stats['difficulty_category'] = case_stats['num_correct'].apply(categorize_difficulty)
    case_stats['accuracy_rate'] = case_stats['num_correct'] / case_stats['num_attempts']
    
    # Merge back to original dataframe
    df = df.merge(
        case_stats[['model', 'base_case_id', 'difficulty_category', 'num_correct', 'accuracy_rate']],
        on=['model', 'base_case_id'],
        how='left'
    )
    
    return df, case_stats


def get_top_discriminating_features(vectorizer, clf, feature_names, n=50):
    """Get top features that discriminate between classes"""
    if hasattr(clf, 'coef_'):
        coef = clf.coef_[0]
    else:
        coef = clf.feature_importances_
    
    top_positive_idx = np.argsort(coef)[-n:][::-1]
    top_negative_idx = np.argsort(coef)[:n]
    
    positive_features = [(feature_names[i], coef[i]) for i in top_positive_idx]
    negative_features = [(feature_names[i], coef[i]) for i in top_negative_idx]
    
    return positive_features, negative_features


def analyze_correctness_predictors(df, output_dir):
    """Analyze words/phrases associated with correctness across all models"""
    print("\n" + "="*80)
    print("ANALYZING CORRECTNESS PREDICTORS (ALL MODELS)")
    print("="*80)
    
    df_valid = df[df['thinking_text'].str.len() > 50].copy()
    print(f"Valid samples: {len(df_valid)}")
    print(f"Correct: {df_valid['is_correct'].sum()} ({df_valid['is_correct'].mean():.1%})")
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(df_valid['thinking_text'])
    y = df_valid['is_correct'].astype(int)
    
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X, y)
    
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"\nCV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    feature_names = vectorizer.get_feature_names_out()
    correct_words, incorrect_words = get_top_discriminating_features(
        vectorizer, clf, feature_names, n=30
    )
    
    results = {
        'cv_accuracy': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'words_for_correct': correct_words,
        'words_for_incorrect': incorrect_words
    }
    
    with open(f"{output_dir}/correctness_predictors_all_models.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n📈 Top words for CORRECT:")
    for word, coef in correct_words[:15]:
        print(f"  • {word:40s} ({coef:6.3f})")
    
    print("\n📉 Top words for INCORRECT:")
    for word, coef in incorrect_words[:15]:
        print(f"  • {word:40s} ({coef:6.3f})")
    
    return results


def analyze_model_specific(df, case_stats, model_name, output_dir):
    """Model-specific analysis with correct difficulty stratification"""
    print(f"\n{'='*80}")
    print(f"ANALYZING MODEL: {model_name}")
    print(f"{'='*80}")
    
    model_df = df[df['model'] == model_name].copy()
    model_cases = case_stats[case_stats['model'] == model_name].copy()
    
    print(f"\n📊 Total unique cases: {len(model_cases)}")
    print(f"📊 Total responses (10 per case): {len(model_df)}")
    print(f"📊 Overall accuracy: {model_df['is_correct'].mean():.1%}")
    
    print("\n📊 Difficulty distribution:")
    diff_dist = model_cases['difficulty_category'].value_counts()
    for cat in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        count = diff_dist.get(cat, 0)
        pct = count / len(model_cases) * 100 if len(model_cases) > 0 else 0
        print(f"  {cat:12s}: {count:4d} cases ({pct:5.1f}%)")
    
    results = {
        'model': model_name,
        'total_cases': len(model_cases),
        'total_responses': len(model_df),
        'overall_accuracy': float(model_df['is_correct'].mean()),
        'difficulty_distribution': diff_dist.to_dict(),
        'analyses': {}
    }
    
    # Overall correctness predictors
    print("\n1️⃣  OVERALL CORRECTNESS PREDICTORS")
    
    model_valid = model_df[model_df['thinking_text'].str.len() > 50]
    if len(model_valid) < 50:
        print("  ⚠️  Insufficient data")
        return results
    
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 3),
        min_df=3,
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(model_valid['thinking_text'])
    y = model_valid['is_correct'].astype(int)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    
    feature_names = vectorizer.get_feature_names_out()
    correct_words, incorrect_words = get_top_discriminating_features(
        vectorizer, clf, feature_names, n=20
    )
    
    results['analyses']['overall'] = {
        'correct_words': correct_words,
        'incorrect_words': incorrect_words
    }
    
    print(f"\n  Top words for CORRECT:")
    for word, coef in correct_words[:10]:
        print(f"    • {word}")
    
    print(f"\n  Top words for INCORRECT:")
    for word, coef in incorrect_words[:10]:
        print(f"    • {word}")
    
    # Difficulty-stratified analysis
    print("\n2️⃣  CORRECTNESS PREDICTORS BY DIFFICULTY LEVEL")
    
    for diff_cat in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        subset = model_df[model_df['difficulty_category'] == diff_cat].copy()
        subset = subset[subset['thinking_text'].str.len() > 50]
        
        if len(subset) < 20:
            print(f"\n  ⚠️  {diff_cat.upper()}: Insufficient data ({len(subset)} samples)")
            continue
        
        accuracy = subset['is_correct'].mean()
        print(f"\n  📌 {diff_cat.upper()} (n={len(subset)}, accuracy={accuracy:.1%})")
        
        if subset['is_correct'].nunique() < 2:
            print(f"     ⚠️  All answers are {'correct' if subset['is_correct'].iloc[0] else 'incorrect'}")
            continue
        
        try:
            vectorizer_strat = TfidfVectorizer(
                max_features=300,
                ngram_range=(1, 2),
                min_df=2,
                stop_words='english'
            )
            
            X_strat = vectorizer_strat.fit_transform(subset['thinking_text'])
            y_strat = subset['is_correct'].astype(int)
            
            clf_strat = LogisticRegression(max_iter=1000, random_state=42)
            clf_strat.fit(X_strat, y_strat)
            
            feature_names_strat = vectorizer_strat.get_feature_names_out()
            correct_strat, incorrect_strat = get_top_discriminating_features(
                vectorizer_strat, clf_strat, feature_names_strat, n=10
            )
            
            results['analyses'][f'{diff_cat}'] = {
                'n_samples': len(subset),
                'accuracy': float(accuracy),
                'correct_words': correct_strat,
                'incorrect_words': incorrect_strat
            }
            
            print(f"     Top words for CORRECT on {diff_cat}:")
            for word, coef in correct_strat[:5]:
                print(f"       • {word}")
        
        except Exception as e:
            print(f"     ⚠️  Error: {str(e)}")
    
    # Save results
    safe_model_name = model_name.replace('/', '_').replace('.', '_')
    with open(f"{output_dir}/model_analysis_{safe_model_name}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def create_summary_report(df, case_stats, output_dir):
    """Create summary report"""
    print("\n" + "="*80)
    print("CREATING SUMMARY REPORT")
    print("="*80)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("TEXT MINING ANALYSIS REPORT")
    report_lines.append("With CORRECT Model-Specific Difficulty (based on 10 runs per case)")
    report_lines.append("="*80)
    report_lines.append("")
    
    report_lines.append("📊 OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"Total responses: {len(df)}")
    report_lines.append(f"Total unique cases: {df['base_case_id'].nunique()}")
    report_lines.append(f"Models analyzed: {df['model'].nunique()}")
    report_lines.append(f"Responses per case per model: 10")
    report_lines.append("")
    
    report_lines.append("📈 MODEL ACCURACIES")
    report_lines.append("-" * 80)
    model_acc = df.groupby('model')['is_correct'].agg(['mean', 'count'])
    model_acc = model_acc.sort_values('mean', ascending=False)
    for model, row in model_acc.iterrows():
        report_lines.append(f"{model:40s} {row['mean']:6.1%} (n={int(row['count'])})")
    report_lines.append("")
    
    report_lines.append("🎯 DIFFICULTY DISTRIBUTION BY MODEL (cases, not responses)")
    report_lines.append("-" * 80)
    for model in df['model'].unique():
        model_cases = case_stats[case_stats['model'] == model]
        report_lines.append(f"\n{model}:")
        diff_dist = model_cases['difficulty_category'].value_counts()
        for cat in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
            count = diff_dist.get(cat, 0)
            pct = count / len(model_cases) * 100 if len(model_cases) > 0 else 0
            report_lines.append(f"  {cat:12s}: {count:4d} cases ({pct:5.1f}%)")
    
    report_text = "\n".join(report_lines)
    with open(f"{output_dir}/summary_report.txt", 'w') as f:
        f.write(report_text)
    
    print(report_text)


def main():
    output_dir = "results/text_mining_analysis_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("TEXT MINING ANALYSIS WITH CORRECT DIFFICULTY DEFINITION")
    print("Using 10-run data from analysis_runs/")
    print("="*80)
    
    # Load data
    df, model_names = load_round1_10_data()
    
    # Calculate model-specific difficulty (CORRECT way)
    df, case_stats = calculate_model_specific_difficulty_correct(df)
    
    print("\n📊 Overall difficulty distribution across all models:")
    print(case_stats['difficulty_category'].value_counts())
    
    # Save case statistics
    case_stats.to_csv(f"{output_dir}/case_difficulty_stats.csv", index=False)
    
    # Save per-response data
    df[['question_id', 'base_case_id', 'model', 'is_correct', 'difficulty_category', 
        'num_correct', 'accuracy_rate']].to_csv(
        f"{output_dir}/response_level_data.csv", index=False
    )
    
    # Global analyses
    correctness_results = analyze_correctness_predictors(df, output_dir)
    
    # Model-specific analyses
    all_model_results = {}
    for model in model_names:
        try:
            result = analyze_model_specific(df, case_stats, model, output_dir)
            all_model_results[model] = result
        except Exception as e:
            print(f"\n⚠️  Error analyzing {model}: {str(e)}")
    
    # Summary report
    create_summary_report(df, case_stats, output_dir)
    
    # Save complete results
    complete_results = {
        'correctness_all_models': correctness_results,
        'model_specific': all_model_results
    }
    
    with open(f"{output_dir}/complete_analysis.json", 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nKey files:")
    print(f"  • summary_report.txt")
    print(f"  • case_difficulty_stats.csv - Difficulty per case per model")
    print(f"  • response_level_data.csv - All 10 responses per case")
    print(f"  • correctness_predictors_all_models.json")
    print(f"  • model_analysis_*.json")


if __name__ == "__main__":
    main()

