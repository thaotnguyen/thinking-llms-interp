#!/usr/bin/env python3
"""
Text Mining Analysis for Multi-Model Pipeline Outputs

Analyzes graded responses from multiple models to find words/phrases associated with:
1. Correctness
2. Difficulty (number of times models got it right out of 10)
3. Model-specific patterns
4. Correctness adjusting for difficulty and model
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def extract_thinking_content(response_text):
    """Extract content between <think> tags"""
    if not response_text or len(response_text) < 10:
        return ""
    
    match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        # If extracted content is too short, fallback to full response
        if len(content) < 50:
            return response_text
        return content
    # If no think tags found, use full response
    return response_text


def load_graded_responses(results_dir):
    """Load all graded response files from the results directory"""
    results_path = Path(results_dir)
    graded_files = list(results_path.glob("responses_*.graded.json"))
    
    all_data = []
    model_names = []
    
    for file_path in graded_files:
        # Extract model name from filename
        model_name = file_path.stem.replace("responses_", "").replace(".graded", "")
        model_names.append(model_name)
        
        print(f"Loading {model_name}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Add model name and extract thinking
        for item in data:
            item['model'] = model_name
            item['thinking_text'] = extract_thinking_content(item.get('full_response', ''))
            all_data.append(item)
    
    print(f"\nLoaded {len(all_data)} responses from {len(model_names)} models:")
    print(f"Models: {', '.join(model_names)}")
    
    return pd.DataFrame(all_data), model_names


def calculate_model_specific_difficulty(df):
    """
    Calculate difficulty per question PER MODEL based on model-specific performance.
    
    Since each model answers each question once, we categorize based on:
    - Correctness (is_correct: True/False)
    - Model's overall performance percentile
    - Response characteristics
    
    Categories per model:
    - very_easy: Model got it right AND it's in the model's top performance tier
    - easy: Model got it right, standard case
    - medium: Borderline cases or medium confidence
    - hard: Model got it wrong, standard case  
    - very_hard: Model got it wrong AND shows clear failure signals
    """
    
    # Calculate response length as a proxy for reasoning complexity
    df['response_length'] = df['thinking_text'].str.len()
    
    # For each model, assign difficulty based on correctness and percentiles
    difficulty_categories = []
    
    for idx, row in df.iterrows():
        model = row['model']
        is_correct = row['is_correct']
        
        # Get model-specific stats
        model_df = df[df['model'] == model]
        model_accuracy = model_df['is_correct'].mean()
        
        # Calculate percentile of response length for this model
        response_len = row['response_length']
        percentile = (model_df['response_length'] < response_len).sum() / len(model_df)
        
        # Categorize difficulty per model
        if is_correct:
            # For correct answers: very_easy or easy
            if percentile < 0.3:  # Shorter responses (less struggle) = very easy
                category = 'very_easy'
            else:
                category = 'easy'
        else:
            # For incorrect answers: hard or very_hard
            if percentile > 0.7:  # Longer responses (more struggle) = very hard
                category = 'very_hard'
            elif model_accuracy > 0.3:  # Model usually does OK, so this is just hard
                category = 'hard'
            else:  # Model struggles overall, medium difficulty
                category = 'medium'
        
        # Check for medium cases: borderline responses
        if is_correct and percentile > 0.6:  # Correct but struggled (long response)
            category = 'medium'
        if not is_correct and percentile < 0.3 and model_accuracy > 0.25:  # Wrong but quick (maybe guessing)
            category = 'medium'
            
        difficulty_categories.append(category)
    
    df['difficulty_category'] = difficulty_categories
    
    # Calculate numeric difficulty score (0=very_easy, 1=very_hard)
    difficulty_map = {'very_easy': 0.0, 'easy': 0.25, 'medium': 0.5, 'hard': 0.75, 'very_hard': 1.0}
    df['difficulty'] = df['difficulty_category'].map(difficulty_map)
    
    return df


def get_top_discriminating_features(vectorizer, clf, feature_names, n=50):
    """Get top features that discriminate between classes"""
    if hasattr(clf, 'coef_'):
        coef = clf.coef_[0]
    else:
        coef = clf.feature_importances_
    
    # Get indices of top positive and negative coefficients
    top_positive_idx = np.argsort(coef)[-n:][::-1]
    top_negative_idx = np.argsort(coef)[:n]
    
    positive_features = [(feature_names[i], coef[i]) for i in top_positive_idx]
    negative_features = [(feature_names[i], coef[i]) for i in top_negative_idx]
    
    return positive_features, negative_features


def analyze_correctness_predictors(df, output_dir):
    """
    Analyze words/phrases associated with correctness across all models
    """
    print("\n" + "="*80)
    print("ANALYZING CORRECTNESS PREDICTORS (ALL MODELS)")
    print("="*80)
    
    # Filter out empty thinking texts
    df_valid = df[df['thinking_text'].str.len() > 50].copy()
    print(f"Valid samples for analysis: {len(df_valid)}")
    
    if len(df_valid) < 50:
        print("⚠️  Insufficient valid samples for analysis")
        return None
    
    # Check class balance
    print(f"Correct: {df_valid['is_correct'].sum()} ({df_valid['is_correct'].mean():.1%})")
    print(f"Incorrect: {(~df_valid['is_correct']).sum()} ({(~df_valid['is_correct']).mean():.1%})")
    
    # Vectorize thinking text with more relaxed parameters
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 3),
        min_df=3,  # Reduced from 5
        max_df=0.9,  # Increased from 0.8
        stop_words='english',
        token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
    )
    
    try:
        X = vectorizer.fit_transform(df_valid['thinking_text'])
        print(f"Feature matrix shape: {X.shape}")
    except ValueError as e:
        print(f"⚠️  Error during vectorization: {e}")
        print("Attempting with more relaxed parameters...")
        vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words=None,  # Don't filter stop words
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )
        X = vectorizer.fit_transform(df_valid['thinking_text'])
        print(f"Feature matrix shape: {X.shape}")
    
    y = df_valid['is_correct'].astype(int)
    
    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X, y)
    
    # Cross-validation score
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Get top features
    feature_names = vectorizer.get_feature_names_out()
    positive_features, negative_features = get_top_discriminating_features(
        vectorizer, clf, feature_names, n=30
    )
    
    # Save results
    results = {
        'cv_accuracy': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'words_phrases_for_correct': positive_features,
        'words_phrases_for_incorrect': negative_features
    }
    
    with open(f"{output_dir}/correctness_predictors_all_models.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print top features
    print("\n📈 Top words/phrases associated with CORRECT answers:")
    for word, coef in positive_features[:15]:
        print(f"  • {word:40s} (coef: {coef:6.3f})")
    
    print("\n📉 Top words/phrases associated with INCORRECT answers:")
    for word, coef in negative_features[:15]:
        print(f"  • {word:40s} (coef: {coef:6.3f})")
    
    return results


def analyze_difficulty_predictors(df, output_dir):
    """
    Analyze words/phrases in THINKING TEXT associated with difficulty per model.
    Now that difficulty is model-specific, we analyze what in the model's reasoning
    predicts whether it found the question difficult.
    """
    print("\n" + "="*80)
    print("ANALYZING DIFFICULTY PREDICTORS (MODEL-SPECIFIC)")
    print("="*80)
    
    # Use thinking text since difficulty is now about the model's experience
    df_valid = df[df['thinking_text'].str.len() > 50].copy()
    
    print(f"Valid samples: {len(df_valid)}")
    print(f"Difficulty distribution:")
    print(df_valid['difficulty_category'].value_counts())
    
    # Vectorize thinking text
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(df_valid['thinking_text'])
    
    # Binary classification: very_hard/hard vs very_easy/easy
    df_valid['is_difficult'] = df_valid['difficulty_category'].isin(['hard', 'very_hard'])
    y = df_valid['is_difficult'].astype(int)
    
    if y.nunique() < 2:
        print("⚠️  Insufficient class diversity for difficulty prediction")
        return None
    
    # Train classifier
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X, y)
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"\nDifficulty prediction CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Get top features
    feature_names = vectorizer.get_feature_names_out()
    difficult_features, easy_features = get_top_discriminating_features(
        vectorizer, clf, feature_names, n=30
    )
    
    # Save results
    results = {
        'cv_accuracy': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'words_phrases_for_difficult': difficult_features,
        'words_phrases_for_easy': easy_features,
        'note': 'Difficulty is now model-specific; these are words in thinking text that predict struggle'
    }
    
    with open(f"{output_dir}/difficulty_predictors.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n🔴 When models use these words in their thinking, they're STRUGGLING (hard/very hard):")
    for word, coef in difficult_features[:15]:
        print(f"  • {word:40s} (coef: {coef:6.3f})")
    
    print("\n🟢 When models use these words in their thinking, they FOUND IT EASY (easy/very easy):")
    for word, coef in easy_features[:15]:
        print(f"  • {word:40s} (coef: {coef:6.3f})")
    
    return results


def analyze_model_specific(df, model_name, output_dir):
    """
    Model-specific analysis with difficulty stratification
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING MODEL: {model_name}")
    print(f"{'='*80}")
    
    model_df = df[df['model'] == model_name].copy()
    model_df = model_df[model_df['thinking_text'].str.len() > 50]
    
    if len(model_df) < 50:
        print(f"⚠️  Insufficient data for {model_name} (only {len(model_df)} samples)")
        return None
    
    results = {
        'model': model_name,
        'total_responses': len(model_df),
        'overall_accuracy': float(model_df['is_correct'].mean()),
        'analyses': {}
    }
    
    print(f"\n📊 Overall accuracy: {results['overall_accuracy']:.1%}")
    print(f"📊 Total responses: {results['total_responses']}")
    
    # 1. Overall correctness predictors
    print("\n1️⃣  OVERALL CORRECTNESS PREDICTORS")
    vectorizer_overall = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.8,
        stop_words='english'
    )
    
    X_overall = vectorizer_overall.fit_transform(model_df['thinking_text'])
    y_overall = model_df['is_correct'].astype(int)
    
    clf_overall = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_overall.fit(X_overall, y_overall)
    
    feature_names = vectorizer_overall.get_feature_names_out()
    correct_words, incorrect_words = get_top_discriminating_features(
        vectorizer_overall, clf_overall, feature_names, n=20
    )
    
    results['analyses']['overall'] = {
        'correct_words': correct_words,
        'incorrect_words': incorrect_words
    }
    
    print(f"\n  When {model_name} uses these words, it's MORE likely to be CORRECT:")
    for word, coef in correct_words[:10]:
        print(f"    • {word}")
    
    print(f"\n  When {model_name} uses these words, it's MORE likely to be INCORRECT:")
    for word, coef in incorrect_words[:10]:
        print(f"    • {word}")
    
    # 2. Difficulty predictors (what words predict hard questions)
    print("\n2️⃣  DIFFICULTY PREDICTORS")
    questions_for_model = model_df.drop_duplicates(subset=['question_id']).copy()
    
    if len(questions_for_model) >= 30:
        vectorizer_diff = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            min_df=2,
            stop_words='english'
        )
        
        X_diff = vectorizer_diff.fit_transform(questions_for_model['question'])
        y_diff = questions_for_model['difficulty']
        
        # Use median split for binary classification
        median_diff = y_diff.median()
        y_diff_binary = (y_diff > median_diff).astype(int)
        
        clf_diff = LogisticRegression(max_iter=1000, random_state=42)
        clf_diff.fit(X_diff, y_diff_binary)
        
        feature_names_diff = vectorizer_diff.get_feature_names_out()
        hard_words, easy_words = get_top_discriminating_features(
            vectorizer_diff, clf_diff, feature_names_diff, n=15
        )
        
        results['analyses']['difficulty'] = {
            'hard_words': hard_words,
            'easy_words': easy_words
        }
        
        print(f"\n  When questions contain these words, they're MORE DIFFICULT for {model_name}:")
        for word, coef in hard_words[:10]:
            print(f"    • {word}")
    
    # 3. Difficulty-stratified correctness analysis
    print("\n3️⃣  CORRECTNESS PREDICTORS BY DIFFICULTY LEVEL")
    
    for diff_cat in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        subset = model_df[model_df['difficulty_category'] == diff_cat].copy()
        
        if len(subset) < 20:
            print(f"\n  ⚠️  {diff_cat.upper()}: Insufficient data ({len(subset)} samples)")
            continue
        
        accuracy = subset['is_correct'].mean()
        print(f"\n  📌 {diff_cat.upper()} questions (n={len(subset)}, accuracy={accuracy:.1%})")
        
        # Skip if all same label
        if subset['is_correct'].nunique() < 2:
            print(f"     ⚠️  All answers are {'correct' if subset['is_correct'].iloc[0] else 'incorrect'}")
            continue
        
        vectorizer_strat = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            stop_words='english'
        )
        
        try:
            X_strat = vectorizer_strat.fit_transform(subset['thinking_text'])
            y_strat = subset['is_correct'].astype(int)
            
            clf_strat = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf_strat.fit(X_strat, y_strat)
            
            feature_names_strat = vectorizer_strat.get_feature_names_out()
            correct_strat, incorrect_strat = get_top_discriminating_features(
                vectorizer_strat, clf_strat, feature_names_strat, n=10
            )
            
            results['analyses'][f'{diff_cat}_questions'] = {
                'n_samples': len(subset),
                'accuracy': float(accuracy),
                'correct_words': correct_strat,
                'incorrect_words': incorrect_strat
            }
            
            print(f"     When {model_name} uses these words on {diff_cat} questions, it's more likely CORRECT:")
            for word, coef in correct_strat[:5]:
                print(f"       • {word}")
        
        except Exception as e:
            print(f"     ⚠️  Error in analysis: {str(e)}")
    
    # Save model-specific results
    safe_model_name = model_name.replace('/', '_').replace('.', '_')
    with open(f"{output_dir}/model_analysis_{safe_model_name}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def create_summary_report(df, all_results, output_dir):
    """Create a comprehensive summary report"""
    print("\n" + "="*80)
    print("CREATING SUMMARY REPORT")
    print("="*80)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("TEXT MINING ANALYSIS REPORT")
    report_lines.append("Multi-Model Response Analysis")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Overview statistics
    report_lines.append("📊 OVERVIEW STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total responses: {len(df)}")
    report_lines.append(f"Unique questions: {df['question_id'].nunique()}")
    report_lines.append(f"Models analyzed: {df['model'].nunique()}")
    report_lines.append("")
    
    # Model accuracies
    report_lines.append("📈 MODEL ACCURACIES")
    report_lines.append("-" * 80)
    model_acc = df.groupby('model')['is_correct'].agg(['mean', 'count'])
    model_acc = model_acc.sort_values('mean', ascending=False)
    for model, row in model_acc.iterrows():
        report_lines.append(f"{model:40s} {row['mean']:6.1%} (n={int(row['count'])})")
    report_lines.append("")
    
    # Difficulty distribution (now model-specific)
    report_lines.append("🎯 DIFFICULTY DISTRIBUTION (MODEL-SPECIFIC)")
    report_lines.append("-" * 80)
    report_lines.append("Note: Difficulty is now calculated per model based on their own performance")
    report_lines.append("")
    diff_dist = df['difficulty_category'].value_counts()
    for cat, count in diff_dist.items():
        pct = count / len(df) * 100
        report_lines.append(f"{cat.upper():15s} {count:5d} responses ({pct:5.1f}%)")
    report_lines.append("")
    
    # Key findings for each model
    report_lines.append("🔍 MODEL-SPECIFIC KEY FINDINGS")
    report_lines.append("=" * 80)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        report_lines.append(f"\n📱 {model.upper()}")
        report_lines.append("-" * 80)
        report_lines.append(f"Overall accuracy: {model_data['is_correct'].mean():.1%}")
        
        # Difficulty breakdown (model-specific)
        report_lines.append("\nDifficulty breakdown for this model:")
        for diff in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
            subset = model_data[model_data['difficulty_category'] == diff]
            if len(subset) > 0:
                pct = len(subset) / len(model_data) * 100
                # Note: for very_easy/easy, accuracy should be high by definition
                # for hard/very_hard, accuracy should be low
                correct_pct = subset['is_correct'].sum() / len(subset) * 100 if len(subset) > 0 else 0
                report_lines.append(f"  {diff:12s}: {len(subset):4d} questions ({pct:5.1f}%) - {correct_pct:5.1f}% correct")
        
        report_lines.append("")
    
    # Save report
    report_text = "\n".join(report_lines)
    with open(f"{output_dir}/summary_report.txt", 'w') as f:
        f.write(report_text)
    
    print(report_text)
    
    # Create visualizations
    create_visualizations(df, output_dir)


def create_visualizations(df, output_dir):
    """Create summary visualizations"""
    
    # 1. Model accuracy comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall accuracy by model
    model_acc = df.groupby('model')['is_correct'].mean().sort_values(ascending=False)
    axes[0, 0].barh(range(len(model_acc)), model_acc.values)
    axes[0, 0].set_yticks(range(len(model_acc)))
    axes[0, 0].set_yticklabels([m.split('-')[-1] for m in model_acc.index])
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_title('Overall Model Accuracy')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Accuracy by difficulty (model-specific now)
    difficulty_order = ['very_easy', 'easy', 'medium', 'hard', 'very_hard']
    pivot = df.pivot_table(
        values='is_correct', 
        index='model', 
        columns='difficulty_category',
        aggfunc='mean'
    )
    # Reorder columns to match difficulty order
    available_cols = [col for col in difficulty_order if col in pivot.columns]
    pivot = pivot[available_cols]
    
    pivot.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy by Difficulty Level')
    axes[0, 1].legend(title='Difficulty')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Difficulty distribution (model-specific now)
    diff_dist = df['difficulty_category'].value_counts()
    colors = {'very_easy': '#00ff00', 'easy': '#90EE90', 'medium': '#FFD700', 
              'hard': '#FFA500', 'very_hard': '#FF0000'}
    pie_colors = [colors.get(cat, '#808080') for cat in diff_dist.index]
    axes[1, 0].pie(diff_dist.values, labels=diff_dist.index, autopct='%1.1f%%', colors=pie_colors)
    axes[1, 0].set_title('Difficulty Distribution (Model-Specific)')
    
    # Response count by model
    response_counts = df.groupby('model').size().sort_values(ascending=False)
    axes[1, 1].barh(range(len(response_counts)), response_counts.values)
    axes[1, 1].set_yticks(range(len(response_counts)))
    axes[1, 1].set_yticklabels([m.split('-')[-1] for m in response_counts.index])
    axes[1, 1].set_xlabel('Number of Responses')
    axes[1, 1].set_title('Responses per Model')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_visualizations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualizations saved to {output_dir}/summary_visualizations.png")


def main():
    # Configuration
    results_dir = "results/vars"
    output_dir = "results/text_mining_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("TEXT MINING ANALYSIS FOR MULTI-MODEL OUTPUTS")
    print("="*80)
    
    # Load data
    df, model_names = load_graded_responses(results_dir)
    
    # Calculate MODEL-SPECIFIC difficulty
    df = calculate_model_specific_difficulty(df)
    
    print("\n📊 Model-specific difficulty distribution:")
    print(df.groupby('model')['difficulty_category'].value_counts().unstack(fill_value=0))
    
    print("\n📊 Overall difficulty distribution:")
    print(df['difficulty_category'].value_counts())
    
    # Save per-response difficulty stats
    df[['question_id', 'model', 'is_correct', 'difficulty_category', 'difficulty', 'response_length']].to_csv(
        f"{output_dir}/model_specific_difficulty_stats.csv", index=False
    )
    
    # Global analyses
    all_results = {}
    
    # 1. Analyze correctness predictors (all models)
    correctness_results = analyze_correctness_predictors(df, output_dir)
    all_results['correctness_all_models'] = correctness_results
    
    # 2. Analyze difficulty predictors
    difficulty_results = analyze_difficulty_predictors(df, output_dir)
    all_results['difficulty_predictors'] = difficulty_results
    
    # 3. Model-specific analyses
    model_results = {}
    for model in model_names:
        try:
            result = analyze_model_specific(df, model, output_dir)
            if result:
                model_results[model] = result
        except Exception as e:
            print(f"\n⚠️  Error analyzing {model}: {str(e)}")
    
    all_results['model_specific'] = model_results
    
    # 4. Create summary report
    create_summary_report(df, all_results, output_dir)
    
    # Save all results
    with open(f"{output_dir}/complete_analysis.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nKey output files:")
    print(f"  • summary_report.txt - Human-readable summary")
    print(f"  • complete_analysis.json - Full analysis results")
    print(f"  • question_difficulty_stats.csv - Question-level statistics")
    print(f"  • correctness_predictors_all_models.json - Global correctness predictors")
    print(f"  • difficulty_predictors.json - Question difficulty predictors")
    print(f"  • model_analysis_*.json - Individual model analyses")
    print(f"  • summary_visualizations.png - Overview plots")


if __name__ == "__main__":
    main()

