"""
Comprehensive experiment to find optimal k value.

Testing: k = 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
"""

import sys
sys.path.append('src')

from evaluation.evaluator import Evaluator

# Initialize
evaluator = Evaluator(
    spider_path=r"F:\text2sql\spider_data",
    index_path=r"F:\text2sql\indices"
)

NUM_EXAMPLES = 100
K_VALUES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

print("=" * 70)
print("COMPREHENSIVE K VALUE EXPERIMENT")
print("=" * 70)
print(f"Evaluating {NUM_EXAMPLES} examples per k value")
print(f"Testing k values: {K_VALUES}\n")

results_summary = []

for k in K_VALUES:
    print("\n" + "=" * 70)
    print(f"TESTING K={k}")
    print("=" * 70)
    
    dev_examples = evaluator.data_loader.get_dev_examples()[:NUM_EXAMPLES]
    
    correct = 0
    total = 0
    
    for i, example in enumerate(dev_examples, 1):
        print(f"[{i}/{len(dev_examples)}] ", end='', flush=True)
        
        # Generate with custom k
        generated_sql = evaluator.pipeline.generate_sql_rag(
            example['question'],
            example['db_id'],
            k=k,
            include_schema=True,
            include_examples=True
        )
        
        # Execute
        gold_success, gold_result = evaluator.executor.execute_sql(
            example['query'], example['db_id']
        )
        gen_success, gen_result = evaluator.executor.execute_sql(
            generated_sql, example['db_id']
        )
        
        # Check match
        execution_match = False
        if gold_success and gen_success:
            execution_match = evaluator.executor.compare_results(gold_result, gen_result)
        
        total += 1
        if execution_match:
            correct += 1
            print("✓")
        else:
            print("✗")
    
    accuracy = correct / total if total > 0 else 0
    results_summary.append((k, accuracy))
    
    print(f"\nK={k} Results: {correct}/{total} = {accuracy:.1%}")

# Final Summary
print("\n\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Examples evaluated per k: {NUM_EXAMPLES}\n")
print(f"{'K Value':<10} {'Accuracy':>10} {'Correct/Total':>15}")
print("-" * 40)

for k, accuracy in results_summary:
    correct = int(accuracy * NUM_EXAMPLES)
    print(f"{k:<10} {accuracy:>9.1%} {correct:>7}/{NUM_EXAMPLES:<7}")

print("\n" + "=" * 70)
print("Analysis:")
best_k = max(results_summary, key=lambda x: x[1])
print(f"Best k value: {best_k[0]} with {best_k[1]:.1%} accuracy")

# Show trend
print("\nAccuracy trend:")
for k, accuracy in results_summary:
    bar = "█" * int(accuracy * 50)
    print(f"k={k:2d}: {bar} {accuracy:.1%}")

# Save results
import json
from pathlib import Path

output_path = Path(r"F:\text2sql\results\k_values_experiment.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

results_data = {
    'num_examples': NUM_EXAMPLES,
    'k_values_tested': K_VALUES,
    'results': [{'k': k, 'accuracy': accuracy} for k, accuracy in results_summary],
    'best_k': best_k[0],
    'best_accuracy': best_k[1]
}

with open(output_path, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"\n\nResults saved to {output_path}")
