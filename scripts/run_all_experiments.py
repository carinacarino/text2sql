"""
Comprehensive experiment script testing all Text-to-SQL methods.

Methods tested:
1. Baseline: No context at all
2. Schema Only: Database schema only (no examples)
3. Examples Only: Example queries only (no schema)
4. Full RAG (Dense): Schema + Examples with dense retrieval
5. BM25 RAG: Schema + Examples with BM25 retrieval
6. CoT RAG: Schema + Examples with Chain-of-Thought prompting
"""

import sys
sys.path.append('src')

from data_processing.load_spider import SpiderDataLoader
from text2sql_pipeline import Text2SQLPipeline
from evaluation.sql_executor import SQLExecutor
from retrieval.bm25_retriever import BM25Retriever
from generation.prompt_builder import PromptBuilder
from generation.ollama_client import OllamaClient
import pickle
import json
from pathlib import Path
import time

print("=" * 70)
print("COMPREHENSIVE TEXT-TO-SQL EXPERIMENTS")
print("=" * 70)

# Configuration
NUM_EXAMPLES = None  # None = all 1034, or set to smaller number for testing
K = 10  # Number of examples to retrieve

# Initialize components
print("\nInitializing components...")
loader = SpiderDataLoader(r"F:\text2sql\spider_data")
pipeline = Text2SQLPipeline(
    spider_path=r"F:\text2sql\spider_data",
    index_path=r"F:\text2sql\indices"
)
executor = SQLExecutor(r"F:\text2sql\spider_data")
ollama_client = OllamaClient()

# Load BM25 retriever
print("Loading BM25 index...")
with open(r"F:\text2sql\indices\bm25_index.pkl", 'rb') as f:
    bm25_retriever = pickle.load(f)

print("Components loaded!\n")

# Get dev examples
dev_examples = loader.get_dev_examples()
if NUM_EXAMPLES:
    dev_examples = dev_examples[:NUM_EXAMPLES]

total_examples = len(dev_examples)
print(f"Will evaluate {total_examples} examples for each method")
print("=" * 70)

# Store all results
all_results = {}
summary = {}
results_dir = Path(r"F:\text2sql\results")
results_dir.mkdir(exist_ok=True)

# Helper function to evaluate a configuration
def evaluate_config(examples, config_name, generate_fn):
    """
    Evaluate a specific configuration.
    
    Args:
        examples: List of examples to evaluate
        config_name: Name of the configuration
        generate_fn: Function that takes (question, db_id) and returns SQL
    
    Returns:
        Dict with results
    """
    print(f"\n{config_name}")
    print("-" * 70)
    
    correct = 0
    detailed_results = []
    start_time = time.time()
    
    for i, example in enumerate(examples, 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(examples)}...")
        
        # Generate SQL using the provided function
        generated_sql = generate_fn(example['question'], example['db_id'])
        
        # Execute and compare
        gold_success, gold_result = executor.execute_sql(example['query'], example['db_id'])
        gen_success, gen_result = executor.execute_sql(generated_sql, example['db_id'])
        
        execution_match = gold_success and gen_success and executor.compare_results(gold_result, gen_result)
        
        if execution_match:
            correct += 1
        
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
    
    elapsed = time.time() - start_time
    accuracy = correct / len(examples)
    
    print(f"  Correct: {correct}/{len(examples)}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Time: {elapsed:.1f}s")
    
    return {
        'method': config_name,
        'num_examples': len(examples),
        'correct': correct,
        'accuracy': accuracy,
        'time_seconds': elapsed,
        'results': detailed_results
    }

# 1. BASELINE (No Context)
print("\n" + "=" * 70)
print("EXPERIMENT 1: BASELINE (No Context)")
print("=" * 70)
baseline_results = evaluate_config(
    dev_examples,
    "Baseline",
    lambda q, db: pipeline.generate_sql_baseline(q)
)
all_results['baseline'] = baseline_results
summary['Baseline'] = baseline_results['accuracy']

# Save baseline results immediately
with open(results_dir / 'baseline.json', 'w') as f:
    json.dump(baseline_results, f, indent=2)
print("  ✓ Saved baseline.json")

# 2. SCHEMA ONLY
print("\n" + "=" * 70)
print("EXPERIMENT 2: SCHEMA ONLY")
print("=" * 70)
schema_results = evaluate_config(
    dev_examples,
    "Schema Only",
    lambda q, db: pipeline.generate_sql_rag(q, db, k=0, include_schema=True, include_examples=False)
)
all_results['schema_only'] = schema_results
summary['Schema Only'] = schema_results['accuracy']

# Save schema results immediately
with open(results_dir / 'schema_only.json', 'w') as f:
    json.dump(schema_results, f, indent=2)
print("  ✓ Saved schema_only.json")

# 3. EXAMPLES ONLY
print("\n" + "=" * 70)
print("EXPERIMENT 3: EXAMPLES ONLY")
print("=" * 70)
examples_results = evaluate_config(
    dev_examples,
    "Examples Only",
    lambda q, db: pipeline.generate_sql_rag(q, db, k=K, include_schema=False, include_examples=True)
)
all_results['examples_only'] = examples_results
summary['Examples Only'] = examples_results['accuracy']

# Save examples results immediately
with open(results_dir / 'examples_only.json', 'w') as f:
    json.dump(examples_results, f, indent=2)
print("  ✓ Saved examples_only.json")

# 4. FULL RAG (Dense Embeddings)
print("\n" + "=" * 70)
print("EXPERIMENT 4: FULL RAG (Dense Embeddings)")
print("=" * 70)
dense_rag_results = evaluate_config(
    dev_examples,
    "Dense RAG",
    lambda q, db: pipeline.generate_sql_rag(q, db, k=K, include_schema=True, include_examples=True)
)
all_results['dense_rag'] = dense_rag_results
summary['Dense RAG'] = dense_rag_results['accuracy']

# Save dense RAG results immediately
with open(results_dir / 'full_rag.json', 'w') as f:
    json.dump(dense_rag_results, f, indent=2)
print("  ✓ Saved full_rag.json")

# 5. BM25 RAG
print("\n" + "=" * 70)
print("EXPERIMENT 5: BM25 RAG")
print("=" * 70)

def generate_bm25(question, db_id):
    """Generate SQL using BM25 retrieval."""
    # Get BM25 results
    results = bm25_retriever.retrieve(question, k=K, db_id=db_id)
    
    # Build context
    context_parts = []
    schema = loader.format_schema_text(db_id)
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
    prompt = PromptBuilder.build_rag_prompt(question, context)
    return ollama_client.generate_sql(prompt)

bm25_results = evaluate_config(
    dev_examples,
    "BM25 RAG",
    generate_bm25
)
all_results['bm25_rag'] = bm25_results
summary['BM25 RAG'] = bm25_results['accuracy']

# Save BM25 results immediately
with open(results_dir / 'bm25_results.json', 'w') as f:
    json.dump(bm25_results, f, indent=2)
print("  ✓ Saved bm25_results.json")

# 6. CHAIN-OF-THOUGHT RAG
print("\n" + "=" * 70)
print("EXPERIMENT 6: CHAIN-OF-THOUGHT RAG")
print("=" * 70)
cot_results = evaluate_config(
    dev_examples,
    "CoT RAG",
    lambda q, db: pipeline.generate_sql_rag(q, db, k=K, include_schema=True, include_examples=True, use_cot=True)
)
all_results['cot_rag'] = cot_results
summary['CoT RAG'] = cot_results['accuracy']

# Save CoT results immediately
with open(results_dir / 'cot_results.json', 'w') as f:
    json.dump(cot_results, f, indent=2)
print("  ✓ Saved cot_results.json")

# FINAL SUMMARY
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Evaluated {total_examples} examples per method")
print(f"K value for retrieval: {K}\n")

print(f"{'Method':<20} {'Accuracy':>10} {'vs Baseline':>15}")
print("-" * 50)

baseline_acc = summary['Baseline']
for method, accuracy in summary.items():
    improvement = accuracy - baseline_acc
    if method == 'Baseline':
        print(f"{method:<20} {accuracy:>9.1%}")
    else:
        print(f"{method:<20} {accuracy:>9.1%} {improvement:>14.1%}")

# Find best method
best_method = max(summary.items(), key=lambda x: x[1])
print(f"\n🏆 Best Method: {best_method[0]} ({best_method[1]:.1%})")

# Save all results
print("\n" + "=" * 70)
print("Saving results...")
results_dir = Path(r"F:\text2sql\results")
results_dir.mkdir(exist_ok=True)

# Save individual method results
for key, result in all_results.items():
    output_path = results_dir / f"{key}.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {key} → {output_path}")

# Save summary
summary_data = {
    'num_examples': total_examples,
    'k_value': K,
    'methods': summary,
    'best_method': best_method[0],
    'best_accuracy': best_method[1]
}
summary_path = results_dir / "experiment_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary_data, f, indent=2)
print(f"  Saved summary → {summary_path}")

print("\n✅ All experiments complete!")
