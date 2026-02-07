"""
Test BM25 retrieval method only.
"""

import sys
sys.path.append('src')

from data_processing.load_spider import SpiderDataLoader
from retrieval.bm25_retriever import BM25Retriever
from generation.ollama_client import OllamaClient
from generation.prompt_builder import PromptBuilder
from evaluation.sql_executor import SQLExecutor
import pickle
import json
from pathlib import Path

print("=" * 70)
print("TESTING BM25 RETRIEVAL")
print("=" * 70)

# Load data
print("\nLoading components...")
loader = SpiderDataLoader(r"F:\text2sql\spider_data")
executor = SQLExecutor(r"F:\text2sql\spider_data")
ollama_client = OllamaClient()

# Load BM25 retriever
with open(r"F:\text2sql\indices\bm25_index.pkl", 'rb') as f:
    bm25_retriever = pickle.load(f)

print("Components loaded!")

# Test parameters
NUM_EXAMPLES = None  # All 1034 examples
K = 10

dev_examples = loader.get_dev_examples()
if NUM_EXAMPLES:
    dev_examples = dev_examples[:NUM_EXAMPLES]

print(f"\nTesting BM25 with k={K} on {len(dev_examples)} examples")
print("=" * 70)

bm25_correct = 0
total = len(dev_examples)
detailed_results = []  # Store per-example results

for i, example in enumerate(dev_examples, 1):
    print(f"[{i}/{total}] {example['question'][:50]}... ", end='', flush=True)
    
    # Get context with BM25 retrieval
    results = bm25_retriever.retrieve(example['question'], k=K, db_id=example['db_id'])
    
    # Format context (schema + examples)
    context_parts = []
    schema = loader.format_schema_text(example['db_id'])
    context_parts.append("DATABASE SCHEMA:")
    context_parts.append(schema)
    context_parts.append("")
    
    if results:
        context_parts.append("EXAMPLE QUERIES FROM THIS DATABASE:")
        context_parts.append("")
        for j, result in enumerate(results, 1):
            context_parts.append(f"Example {j}:")
            context_parts.append(f"Question: {result['question']}")
            context_parts.append(f"SQL: {result['query']}")
            context_parts.append("")
    
    context = "\n".join(context_parts)
    prompt = PromptBuilder.build_rag_prompt(example['question'], context)
    generated_sql = ollama_client.generate_sql(prompt)
    
    # Execute
    gold_success, gold_result = executor.execute_sql(example['query'], example['db_id'])
    gen_success, gen_result = executor.execute_sql(generated_sql, example['db_id'])
    
    execution_match = gold_success and gen_success and executor.compare_results(gold_result, gen_result)
    
    if execution_match:
        bm25_correct += 1
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

bm25_accuracy = bm25_correct / total

# Results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Total examples: {total}")
print(f"Correct: {bm25_correct}")
print(f"Accuracy: {bm25_accuracy:.1%}")

# Save both summary and detailed results
results = {
    'method': 'BM25',
    'num_examples': total,
    'k': K,
    'correct': bm25_correct,
    'accuracy': bm25_accuracy,
    'results': detailed_results  # Add detailed per-example results
}

output_path = Path(r"F:\text2sql\results\bm25_results.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
