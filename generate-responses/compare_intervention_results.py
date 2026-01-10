#!/usr/bin/env python3
"""
Compare baseline vs intervention results.

Analyzes:
1. Overall accuracy difference
2. Accuracy improvement on cases with interventions
3. Statistical significance
4. Examples of successful interventions
"""

import argparse
import json
import numpy as np
from scipy import stats
from collections import defaultdict

parser = argparse.ArgumentParser(description="Compare baseline vs intervention results")
parser.add_argument("--baseline", required=True, help="Baseline graded JSON")
parser.add_argument("--intervention", required=True, help="Intervention graded JSON")
parser.add_argument("--output", required=True, help="Output comparison JSON")
args = parser.parse_args()


def load_graded(path):
    """Load graded JSON and index by question_id."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    by_id = {}
    for item in data:
        qid = item.get('question_id')
        by_id[qid] = item
    
    return data, by_id


def main():
    print("="*80)
    print("INTERVENTION EXPERIMENT RESULTS")
    print("="*80)
    
    # Load data
    baseline_list, baseline_dict = load_graded(args.baseline)
    intervention_list, intervention_dict = load_graded(args.intervention)
    
    print(f"\nLoaded:")
    print(f"  Baseline: {len(baseline_list)} cases")
    print(f"  Intervention: {len(intervention_list)} cases")
    
    # Overall accuracy
    baseline_acc = np.mean([item['is_correct'] for item in baseline_list])
    intervention_acc = np.mean([item['is_correct'] for item in intervention_list])
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"  Baseline accuracy:     {baseline_acc:.1%} ({sum(item['is_correct'] for item in baseline_list)}/{len(baseline_list)})")
    print(f"  Intervention accuracy: {intervention_acc:.1%} ({sum(item['is_correct'] for item in intervention_list)}/{len(intervention_list)})")
    print(f"  Difference:            {(intervention_acc - baseline_acc)*100:+.1f} percentage points")
    
    # Paired comparison (same questions)
    common_ids = set(baseline_dict.keys()) & set(intervention_dict.keys())
    print(f"\nCommon cases: {len(common_ids)}")
    
    baseline_correct = []
    intervention_correct = []
    
    for qid in common_ids:
        baseline_correct.append(int(baseline_dict[qid]['is_correct']))
        intervention_correct.append(int(intervention_dict[qid]['is_correct']))
    
    # McNemar's test for paired binary data
    # Create contingency table
    both_correct = sum(b == 1 and i == 1 for b, i in zip(baseline_correct, intervention_correct))
    both_incorrect = sum(b == 0 and i == 0 for b, i in zip(baseline_correct, intervention_correct))
    baseline_only = sum(b == 1 and i == 0 for b, i in zip(baseline_correct, intervention_correct))
    intervention_only = sum(b == 0 and i == 1 for b, i in zip(baseline_correct, intervention_correct))
    
    print(f"\n📊 PAIRED COMPARISON:")
    print(f"  Both correct:           {both_correct}")
    print(f"  Both incorrect:         {both_incorrect}")
    print(f"  Baseline only correct:  {baseline_only}")
    print(f"  Intervention only correct: {intervention_only}")
    
    # McNemar's test
    if baseline_only + intervention_only > 0:
        mcnemar_stat = (abs(baseline_only - intervention_only) - 1)**2 / (baseline_only + intervention_only)
        mcnemar_p = stats.chi2.sf(mcnemar_stat, 1)
        print(f"\nMcNemar's test:")
        print(f"  Statistic: {mcnemar_stat:.4f}")
        print(f"  P-value: {mcnemar_p:.4f}")
        print(f"  Significant (p<0.05): {'Yes' if mcnemar_p < 0.05 else 'No'}")
    
    # Analyze cases with interventions
    cases_with_intervention = [item for item in intervention_list if item.get('num_interventions', 0) > 0]
    cases_without_intervention = [item for item in intervention_list if item.get('num_interventions', 0) == 0]
    
    print(f"\n📊 INTERVENTION ANALYSIS:")
    print(f"  Cases with intervention: {len(cases_with_intervention)} ({len(cases_with_intervention)/len(intervention_list)*100:.1f}%)")
    print(f"  Cases without intervention: {len(cases_without_intervention)}")
    
    if cases_with_intervention:
        acc_with = np.mean([item['is_correct'] for item in cases_with_intervention])
        acc_without = np.mean([item['is_correct'] for item in cases_without_intervention])
        
        print(f"\n  Accuracy on cases WITH intervention:    {acc_with:.1%}")
        print(f"  Accuracy on cases WITHOUT intervention: {acc_without:.1%}")
        
        # Compare to baseline for cases that had interventions
        baseline_acc_on_intervention_cases = np.mean([
            baseline_dict[item['question_id']]['is_correct'] 
            for item in cases_with_intervention 
            if item['question_id'] in baseline_dict
        ])
        
        print(f"\n  Baseline accuracy on same cases: {baseline_acc_on_intervention_cases:.1%}")
        print(f"  Improvement from intervention:   {(acc_with - baseline_acc_on_intervention_cases)*100:+.1f} percentage points")
    
    # Find successful interventions (baseline wrong → intervention right)
    successful_interventions = []
    failed_interventions = []
    
    for qid in common_ids:
        b_item = baseline_dict[qid]
        i_item = intervention_dict[qid]
        
        if i_item.get('num_interventions', 0) > 0:
            if not b_item['is_correct'] and i_item['is_correct']:
                successful_interventions.append({
                    'question_id': qid,
                    'baseline_answer': b_item.get('extracted_answer', ''),
                    'intervention_answer': i_item.get('extracted_answer', ''),
                    'gold_answer': i_item.get('gold_answer', ''),
                    'interventions': i_item.get('interventions', [])
                })
            elif b_item['is_correct'] and not i_item['is_correct']:
                failed_interventions.append({
                    'question_id': qid,
                    'baseline_answer': b_item.get('extracted_answer', ''),
                    'intervention_answer': i_item.get('extracted_answer', ''),
                    'gold_answer': i_item.get('gold_answer', ''),
                    'interventions': i_item.get('interventions', [])
                })
    
    print(f"\n✅ SUCCESSFUL INTERVENTIONS (baseline wrong → intervention right): {len(successful_interventions)}")
    if successful_interventions:
        print("\nExamples:")
        for ex in successful_interventions[:3]:
            print(f"  • Question: {ex['question_id']}")
            print(f"    Gold: {ex['gold_answer']}")
            print(f"    Baseline: {ex['baseline_answer']} ❌")
            print(f"    Intervention: {ex['intervention_answer']} ✅")
            print(f"    Interventions: {len(ex['interventions'])}")
            print()
    
    print(f"❌ FAILED INTERVENTIONS (baseline right → intervention wrong): {len(failed_interventions)}")
    if failed_interventions:
        print("\nExamples:")
        for ex in failed_interventions[:3]:
            print(f"  • Question: {ex['question_id']}")
            print(f"    Gold: {ex['gold_answer']}")
            print(f"    Baseline: {ex['baseline_answer']} ✅")
            print(f"    Intervention: {ex['intervention_answer']} ❌")
            print(f"    Interventions: {len(ex['interventions'])}")
            print()
    
    # Save comparison results
    results = {
        'baseline_accuracy': float(baseline_acc),
        'intervention_accuracy': float(intervention_acc),
        'improvement': float(intervention_acc - baseline_acc),
        'num_cases': len(common_ids),
        'contingency_table': {
            'both_correct': both_correct,
            'both_incorrect': both_incorrect,
            'baseline_only_correct': baseline_only,
            'intervention_only_correct': intervention_only
        },
        'cases_with_intervention': len(cases_with_intervention),
        'successful_interventions': len(successful_interventions),
        'failed_interventions': len(failed_interventions),
        'successful_examples': successful_interventions,
        'failed_examples': failed_interventions
    }
    
    if baseline_only + intervention_only > 0:
        results['mcnemar_test'] = {
            'statistic': float(mcnemar_stat),
            'p_value': float(mcnemar_p),
            'significant': bool(mcnemar_p < 0.05)
        }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {args.output}")
    
    # Overall conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if intervention_acc > baseline_acc:
        print(f"✅ Intervention IMPROVED accuracy by {(intervention_acc - baseline_acc)*100:.1f} percentage points")
        if mcnemar_p < 0.05:
            print(f"✅ Improvement is STATISTICALLY SIGNIFICANT (p={mcnemar_p:.4f})")
        else:
            print(f"⚠️  Improvement is NOT statistically significant (p={mcnemar_p:.4f})")
    elif intervention_acc < baseline_acc:
        print(f"❌ Intervention DECREASED accuracy by {(baseline_acc - intervention_acc)*100:.1f} percentage points")
    else:
        print(f"➖ No difference in accuracy")
    
    print(f"\nSuccessful interventions: {len(successful_interventions)}")
    print(f"Failed interventions: {len(failed_interventions)}")
    print(f"Net gain: {len(successful_interventions) - len(failed_interventions)}")


if __name__ == "__main__":
    main()

