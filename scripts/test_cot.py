"""
Test Chain-of-Thought prompting for SQL generation.
"""

import sys
sys.path.append('src')

from data_processing.load_spider import SpiderDataLoader
from text2sql_pipeline import Text2SQLPipeline
from evaluation.sql_executor import SQLExecutor
import json
from pathlib import Path

print("=" * 70)
print("TESTING CHAIN-OF-THOUGHT PROMPTING")
print("=" * 70)

# Initialize
loader = SpiderDataLoader(r"F:\text2sql\spider_data")
pipeline = Text2SQLPipeline(
    spider_path=r"F:\text2sql\spider_data",
    index_path=r"F:\text2sql\indices"
)
executor = SQLExecutor(r"F:\text2sql\spider_data")

# Test parameters
NUM_EXAMPLES = None  # All 1034 examples
K = 10

dev_examples = loader.get_dev_examples()
if NUM_EXAMPLES:
    dev_examples = dev_examples[:NUM_EXAMPLES]

print(f"\nTesting CoT with k={K} on {len(dev_examples)} examples")
print("=" * 70)

cot_correct = 0
total = len(dev_examples)
detailed_results = []  # Store per-example results

for i, example in enumerate(dev_examples, 1):
    print(f"[{i}/{total}] {example['question'][:50]}... ", end='', flush=True)
    
    # Generate with CoT
    generated_sql = pipeline.generate_sql_rag(
        example['question'],
        example['db_id'],
        k=K,
        include_schema=True,
        include_examples=True,
        use_cot=True  # Enable Chain-of-Thought
    )
    
    # Execute
    gold_success, gold_result = executor.execute_sql(example['query'], example['db_id'])
    gen_success, gen_result = executor.execute_sql(generated_sql, example['db_id'])
    
    execution_match = gold_success and gen_success and executor.compare_results(gold_result, gen_result)
    
    if execution_match:
        cot_correct += 1
        print("✓")
    else:
        print("✗")
    
    # Store detailed result
    detailed_results.append({
        'index': i-1,
        'question': example['question'],
        'db_id': example['db_id'],
        'gold_sql': example['query'],
        'generated_sql': generated_sql,
        'execution_match': execution_match,
        'gold_success': gold_success,
        'gen_success': gen_success
    })

cot_accuracy = cot_correct / total

# Results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Total examples: {total}")
print(f"Correct: {cot_correct}")
print(f"Accuracy: {cot_accuracy:.1%}")

# Save both summary and detailed results
results = {
    'method': 'Chain-of-Thought RAG',
    'num_examples': total,
    'k': K,
    'correct': cot_correct,
    'accuracy': cot_accuracy,
    'results': detailed_results  # Add detailed per-example results
}

output_path = Path(r"F:\text2sql\results\cot_results.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")

# Compare with previous results
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"Schema Only:           49.0%")
print(f"Dense RAG (k=10):      47.9%")
print(f"BM25 RAG (k=10):       48.0%")
print(f"CoT RAG (k=10):        {cot_accuracy:.1%}")
