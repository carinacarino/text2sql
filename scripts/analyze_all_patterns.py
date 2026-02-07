"""
Analyze success/failure patterns using detailed results from all methods.
"""

import sys
sys.path.append('src')

import json
from pathlib import Path

print("=" * 70)
print("ANALYZING SUCCESS/FAILURE PATTERNS ACROSS ALL METHODS")
print("=" * 70)

# Load all detailed results
results_dir = Path(r"F:\text2sql\results")

print("\nLoading detailed results...")

# Load baseline
with open(results_dir / 'baseline.json', 'r') as f:
    baseline_data = json.load(f)
    
# Load schema only
with open(results_dir / 'schema_only.json', 'r') as f:
    schema_data = json.load(f)
    
# Load examples only  
with open(results_dir / 'examples_only.json', 'r') as f:
    examples_data = json.load(f)
    
# Load dense RAG
with open(results_dir / 'full_rag.json', 'r') as f:
    dense_data = json.load(f)
    
# Load BM25 (if detailed version exists)
try:
    with open(results_dir / 'bm25_results.json', 'r') as f:
        bm25_data = json.load(f)
        if 'results' in bm25_data:
            print("  Loaded BM25 detailed results")
        else:
            print("  BM25 results found but no detailed data - re-run test_bm25.py")
            bm25_data = None
except FileNotFoundError:
    print("  BM25 results not found - run test_bm25.py first")
    bm25_data = None
    
# Load CoT (if detailed version exists)
try:
    with open(results_dir / 'cot_results.json', 'r') as f:
        cot_data = json.load(f)
        if 'results' in cot_data:
            print("  Loaded CoT detailed results")
        else:
            print("  CoT results found but no detailed data - re-run test_cot.py")
            cot_data = None
except FileNotFoundError:
    print("  CoT results not found - run test_cot.py first")
    cot_data = None

# Get number of examples
num_examples = len(baseline_data['results'])
print(f"\nAnalyzing {num_examples} examples")

# Build comprehensive comparison
comparisons = []
for i in range(num_examples):
    comp = {
        'index': i,
        'question': baseline_data['results'][i]['question'],
        'db_id': baseline_data['results'][i]['db_id'],
        'gold_sql': baseline_data['results'][i]['gold_sql'],
        'baseline': baseline_data['results'][i]['execution_match'],
        'schema_only': schema_data['results'][i]['execution_match'],
        'examples_only': examples_data['results'][i]['execution_match'],
        'dense_rag': dense_data['results'][i]['execution_match']
    }
    
    # Add BM25 if available
    if bm25_data:
        comp['bm25_rag'] = bm25_data['results'][i]['execution_match']
    
    # Add CoT if available
    if cot_data:
        comp['cot_rag'] = cot_data['results'][i]['execution_match']
    
    comparisons.append(comp)

# Accuracy summary
print("\n" + "=" * 70)
print("METHOD ACCURACIES")
print("=" * 70)

methods = ['baseline', 'schema_only', 'examples_only', 'dense_rag']
if bm25_data:
    methods.append('bm25_rag')
if cot_data:
    methods.append('cot_rag')

for method in methods:
    correct = sum(1 for c in comparisons if c[method])
    print(f"{method:<15}: {correct}/{num_examples} = {correct/num_examples:.1%}")

# Find unique successes
print("\n" + "=" * 70)
print("UNIQUE SUCCESS PATTERNS")
print("=" * 70)

for method in methods:
    # Find examples where only this method succeeded
    only_this = []
    for c in comparisons:
        if c[method]:  # This method succeeded
            others_failed = all(not c[m] for m in methods if m != method)
            if others_failed:
                only_this.append(c)
    
    if only_this:
        print(f"\n{method.upper()} - Only this method succeeded: {len(only_this)} examples")
        if only_this:
            ex = only_this[0]
            print(f"  Example #{ex['index']+1}:")
            print(f"    Question: {ex['question']}")
            print(f"    Database: {ex['db_id']}")

# Find hard examples (all methods fail)
all_failed = []
for c in comparisons:
    if all(not c[m] for m in methods):
        all_failed.append(c)

print(f"\n\nALL METHODS FAILED: {len(all_failed)} examples")
if all_failed[:3]:
    print("\nFirst 3 hard examples:")
    for ex in all_failed[:3]:
        print(f"\n  Example #{ex['index']+1}:")
        print(f"    Question: {ex['question']}")
        print(f"    Database: {ex['db_id']}")

# Find easy examples (all methods succeed)
all_succeeded = []
for c in comparisons:
    if all(c[m] for m in methods):
        all_succeeded.append(c)

print(f"\n\nALL METHODS SUCCEEDED: {len(all_succeeded)} examples")
if all_succeeded[:3]:
    print("\nFirst 3 easy examples:")
    for ex in all_succeeded[:3]:
        print(f"\n  Example #{ex['index']+1}:")
        print(f"    Question: {ex['question']}")
        print(f"    Database: {ex['db_id']}")

print("\n" + "=" * 70)
print("PAIRWISE COMPARISONS")
print("=" * 70)

def compare_methods(name_a, name_b):
    """Compare success patterns between two methods."""
    a_only = sum(1 for c in comparisons if c[name_a] and not c[name_b])
    b_only = sum(1 for c in comparisons if c[name_b] and not c[name_a])
    diff = a_only - b_only

    if diff > 0:
        winner = name_a
    elif diff < 0:
        winner = name_b
    else:
        winner = "tie"

    print(f"\n{name_a} vs {name_b}:")
    print(f"  {name_a} succeeds, {name_b} fails: {a_only}")
    print(f"  {name_b} succeeds, {name_a} fails: {b_only}")
    
    if winner == "tie":
        print("  → Result: tie")
    else:
        print(f"  → Winner: {winner} by {abs(diff)}")

# Run all comparisons
compare_methods('schema_only', 'examples_only')

if bm25_data:
    compare_methods('dense_rag', 'bm25_rag')

if cot_data:
    compare_methods('dense_rag', 'cot_rag')

# Method agreement analysis
print("\n" + "=" * 70)
print("METHOD AGREEMENT ANALYSIS")
print("=" * 70)

# How often do methods agree?
all_agree = sum(1 for c in comparisons 
    if len(set(c[m] for m in methods)) == 1)
print(f"\nAll methods agree: {all_agree}/{num_examples} = {all_agree/num_examples:.1%}")

# Save comprehensive analysis
output = {
    'num_examples': num_examples,
    'methods_compared': methods,
    'accuracies': {m: sum(1 for c in comparisons if c[m])/num_examples for m in methods},
    'all_failed': len(all_failed),
    'all_succeeded': len(all_succeeded),
    'unique_successes': {m: sum(1 for c in comparisons 
                                if c[m] and all(not c[o] for o in methods if o != m))
                         for m in methods},
    'comparisons': comparisons
}

output_path = results_dir / 'comprehensive_analysis.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n\nComprehensive analysis saved to {output_path}")
