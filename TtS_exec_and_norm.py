import json
import sqlite3
import os

# --- GLOBAL CONFIGURATION ---
INPUT_JSON = './results/pipeline1_text_to_sql.json'
OUTPUT_JSON = './results/pipeline1_executed_data.json'
DB_BASE_DIR = './database'

# Target databases selected for this evaluation
TARGET_DATABASES = ['world_1', 'orchestra']

def normalize_data(raw_data):
    normalized_list = []
    for row in raw_data:
        norm_row = []
        for item in row:
            if isinstance(item, str):
                norm_row.append(item.lower().strip())
            elif item is None:
                norm_row.append("null")
            else:
                norm_row.append(str(item))
        normalized_list.append(norm_row)
    
    return sorted(normalized_list)

def execute_query(db_id, query):
    if not query or query.strip() == "":
        return [], "Empty query"

    db_path = os.path.join(DB_BASE_DIR, db_id, f"{db_id}.sqlite")
    
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.cursor()
        cursor.execute(query)
        
        raw_results = cursor.fetchall()
        conn.close()
        
        return normalize_data(raw_results), "Success"
        
    except sqlite3.Error as e:
        return [], f"SQL Error: {str(e)}"
    except Exception as e:
        return [], f"Execution Error: {str(e)}"

def run_execution_step():
    print("Starting SQL query execution on local databases...\n")
    
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
        
    evaluation_tasks = [t for t in tasks if t['db_id'] in TARGET_DATABASES]
        
    results_with_data = []
    success_count = 0
    error_count = 0
    
    for i, task in enumerate(evaluation_tasks):
        db_id = task['db_id']
        
        truth_data, truth_status = execute_query(db_id, task['query'])
        
        pred_data, pred_status = execute_query(db_id, task['predicted_query'])
        
        if pred_status == "Success":
            success_count += 1
        else:
            error_count += 1
            print(f"Query Error [{i+1}] ({db_id}): {pred_status}")
            print(f"Malformed Query Syntax: {task['predicted_query']}\n" + "-"*40)
        
        task['ground_truth_data'] = truth_data
        task['predicted_data'] = pred_data
        task['execution_status'] = pred_status
        
        results_with_data.append(task)
        
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_with_data, f, indent=4)
        
    print(f"\nCompleted execution step!")
    print(f"Stats: {success_count} successful, {error_count} failed.")
    print(f"Extracted and normalized data saved in {OUTPUT_JSON}")

if __name__ == "__main__":
    run_execution_step()