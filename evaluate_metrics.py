import json
import sqlite3
import re

# --- CONFIGURATION ---
PIPELINE1_FILE = './results/pipeline1_text_to_sql.json'
PIPELINE2_L2_FILE = './results/pipeline2_table_qa.json'
PIPELINE2_L1_FILE = './results/pipeline2_table_qa_simple.json'
PIPELINE2_L3_FILE = './results/pipeline2_table_qa_COT.json'
DB_BASE_DIR = './database'

def execute_query(db_id, query):
    try:
        conn = sqlite3.connect(f"{DB_BASE_DIR}/{db_id}/{db_id}.sqlite")
        cursor = conn.cursor()
        cursor.execute(query)
        results = [list(r) for r in cursor.fetchall()]
        conn.close()
        return results
    except Exception:
        return []

def extract_json_from_text(raw_text):
    try:
        match = re.search(r'\[\s*\[.*\]\s*\]', raw_text, re.DOTALL)
        if match:
            clean_str = match.group(0)
            return json.loads(clean_str)
        return []
    except Exception:
        return []

def extract_json_from_cot(raw_text):
    try:
        match = re.search(r'<json>\s*(.*?)\s*</json>', raw_text, re.DOTALL | re.IGNORECASE)
        
        if match:
            clean_str = match.group(1).strip()
            return json.loads(clean_str)   
        return []
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return []

def calculate_metrics(ground_truth_data, predicted_data):
    """Calculates Cell Precision, Cell Recall, and Tuple Cardinality under the Qatch framework."""
    def flatten(table):
        return [str(cell).strip().lower() for row in table for cell in row] if table else []

    gt_cells = flatten(ground_truth_data)
    pred_cells = flatten(predicted_data)

    cardinality = 1.0 if len(ground_truth_data) == len(predicted_data) else 0.0

    matches = 0
    temp_gt = gt_cells.copy()
    for cell in pred_cells:
        if cell in temp_gt:
            matches += 1
            temp_gt.remove(cell)  

    precision = (matches / len(pred_cells)) if pred_cells else 0.0
    recall = (matches / len(gt_cells)) if gt_cells else 0.0

    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "cardinality": cardinality
    }

def run_evaluation():
    with open(PIPELINE1_FILE, 'r') as f: p1_tasks = json.load(f)
    with open(PIPELINE2_L1_FILE, 'r') as f: p2_l1_tasks = json.load(f)
    with open(PIPELINE2_L2_FILE, 'r') as f: p2_l2_tasks = json.load(f)
    with open(PIPELINE2_L3_FILE, 'r') as f: p2_l3_tasks = json.load(f)

    print("STARTING  EVALUATION PIPELINE\n")
    print(f"{'DB ID':<12} | {'P1 (SQL)':<20} | {'L1 (Baseline)':<14} | {'L2 (Constrained)':<14} | {'L3 (CoT)':<14}")
    print(f"{'':<12} | {'P':<5} {'R':<5} {'C':<5} | {'P':<6} {'R':<6} {'C':<6} | {'P':<6} {'R':<6} {'C':<6} | {'P':<6} {'R':<6} {'C':<6}")
    print("-" * 115)

    p1_prec, p1_rec, p1_card = 0.0, 0.0, 0.0
    p2_l1_prec, p2_l1_rec, p2_l1_card = 0.0, 0.0, 0.0
    p2_l2_prec, p2_l2_rec, p2_l2_card = 0.0, 0.0, 0.0
    p2_l3_prec, p2_l3_rec, p2_l3_card = 0.0, 0.0, 0.0

    for i in range(len(p1_tasks)):
        task = p1_tasks[i]
        db_id = task['db_id']
        question = task['question']
        
        gt_data = execute_query(db_id, task['query'])

        p1_sql = task.get('predicted_query', '')
        p1_data = execute_query(db_id, p1_sql)
        p1_metrics = calculate_metrics(gt_data, p1_data)

        p2_l1_raw = p2_l1_tasks[i].get('predicted_table_qa_data', '[]')
        p2_l1_data = extract_json_from_text(p2_l1_raw)
        p2_l1_metrics = calculate_metrics(gt_data, p2_l1_data)

        p2_l2_raw = p2_l2_tasks[i].get('predicted_table_qa_data', '[]')
        p2_l2_data = extract_json_from_text(p2_l2_raw)
        p2_l2_metrics = calculate_metrics(gt_data, p2_l2_data)

        p2_l3_raw = p2_l3_tasks[i].get('predicted_table_qa_data', '[]')
        p2_l3_data = extract_json_from_cot(p2_l3_raw)
        p2_l3_metrics = calculate_metrics(gt_data, p2_l3_data)

        p1_prec += p1_metrics['precision']; p1_rec += p1_metrics['recall']; p1_card += p1_metrics['cardinality']
        p2_l1_prec += p2_l1_metrics['precision']; p2_l1_rec += p2_l1_metrics['recall']; p2_l1_card += p2_l1_metrics['cardinality']
        p2_l2_prec += p2_l2_metrics['precision']; p2_l2_rec += p2_l2_metrics['recall']; p2_l2_card += p2_l2_metrics['cardinality']
        p2_l3_prec += p2_l3_metrics['precision']; p2_l3_rec += p2_l3_metrics['recall']; p2_l3_card += p2_l3_metrics['cardinality']

        print(f"Question: {question}")
        print(f"{db_id:<12} | {p1_metrics['precision']:<5} {p1_metrics['recall']:<5} {p1_metrics['cardinality']:<5} | {p2_l1_metrics['precision']:<6} {p2_l1_metrics['recall']:<6} {p2_l1_metrics['cardinality']:<6} | {p2_l2_metrics['precision']:<6} {p2_l2_metrics['recall']:<6} {p2_l2_metrics['cardinality']:<6} | {p2_l3_metrics['precision']:<6} {p2_l3_metrics['recall']:<6} {p2_l3_metrics['cardinality']:<6}")
        print("-" * 115)

    n = len(p1_tasks)
    print("\nFINAL AVERAGES")
    print("-" * 115)
    print(f"{'OVERALL AVG':<12} | {round(p1_prec/n, 2):<5} {round(p1_rec/n, 2):<5} {round(p1_card/n, 2):<5} | {round(p2_l1_prec/n, 2):<6} {round(p2_l1_rec/n, 2):<6} {round(p2_l1_card/n, 2):<6} | {round(p2_l2_prec/n, 2):<6} {round(p2_l2_rec/n, 2):<6} {round(p2_l2_card/n, 2):<6} | {round(p2_l3_prec/n, 2):<6} {round(p2_l3_rec/n, 2):<6} {round(p2_l3_card/n, 2):<6}")
    print("-" * 115)
if __name__ == "__main__":
    run_evaluation()