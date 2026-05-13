import json
import sqlite3
import re

# --- CONFIGURAZIONE ---
PIPELINE1_FILE = './results/pipeline1_text_to_sql.json'
PIPELINE2_FILE = './results/pipeline2_table_qa.json'
PIPELINE2S_FILE = './results/pipeline2_table_qa_simple.json'
PIPELINE2COT_FILE = './results/pipeline2_table_qa_COT.json'
DB_BASE_DIR = './database'

def execute_query(db_id, query):
    """Esegue una query SQLite e restituisce i dati. Se fallisce, restituisce lista vuota."""
    try:
        conn = sqlite3.connect(f"{DB_BASE_DIR}/{db_id}/{db_id}.sqlite")
        cursor = conn.cursor()
        cursor.execute(query)
        results = [list(r) for r in cursor.fetchall()]
        conn.close()
        return results
    except Exception as e:
        return [] # Se l'SQL di Granite è sbagliato, restituisce vuoto (Recall = 0 come da PDF)

def extract_json_from_text(raw_text):
    """Il nostro 'salvavita'. Cerca un array JSON dentro le chiacchiere di Granite."""
    try:
        # Cerca qualcosa che inizia con [ e finisce con ] (array di array)
        match = re.search(r'\[\s*\[.*\]\s*\]', raw_text, re.DOTALL)
        if match:
            clean_str = match.group(0)
            return json.loads(clean_str)
        return []
    except:
        return []

def extract_json_from_cot(raw_text):
    """Estrae in modo sicuro il JSON generato dopo il Chain-of-Thought."""
    try:
        # Cerca tutto quello che c'è ESATTAMENTE tra <json> e </json>
        match = re.search(r'<json>\s*(.*?)\s*</json>', raw_text, re.DOTALL | re.IGNORECASE)
        
        if match:
            clean_str = match.group(1).strip()
            return json.loads(clean_str)
            
        # Fallback di sicurezza: se il modello si dimentica i tag <json>,
        # proviamo a cercare un classico array 2D nel testo
        fallback_match = re.search(r'\[\s*\[.*\]\s*\]', raw_text, re.DOTALL)
        if fallback_match:
             return json.loads(fallback_match.group(0))
             
        return []
    except Exception as e:
        print(f"Errore di parsing JSON: {e}")
        return []

def calculate_metrics(ground_truth_data, predicted_data):
    """Calcola Cell Precision, Cell Recall e Tuple Cardinality (Logica Qatch)."""
    # 1. Appiattiamo le celle per confrontarle (tutto minuscolo e stringa)
    def flatten(table):
        return [str(cell).strip().lower() for row in table for cell in row] if table else []

    gt_cells = flatten(ground_truth_data)
    pred_cells = flatten(predicted_data)

    # 2. Tuple Cardinality (1 se il numero di righe è identico, 0 altrimenti)
    cardinality = 1.0 if len(ground_truth_data) == len(predicted_data) else 0.0

    # 3. Precision & Recall (sulle celle)
    matches = 0
    temp_gt = gt_cells.copy()
    for cell in pred_cells:
        if cell in temp_gt:
            matches += 1
            temp_gt.remove(cell) # Rimuove per evitare doppi conteggi

    precision = (matches / len(pred_cells)) if pred_cells else 0.0
    recall = (matches / len(gt_cells)) if gt_cells else 0.0

    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "cardinality": cardinality
    }

def run_evaluation():
    # Carichiamo i risultati delle due pipeline
    with open(PIPELINE1_FILE, 'r') as f: p1_tasks = json.load(f)
    with open(PIPELINE2_FILE, 'r') as f: p2_tasks = json.load(f)
    with open(PIPELINE2S_FILE, 'r') as f: p2S_tasks = json.load(f)
    with open(PIPELINE2COT_FILE, 'r') as f: p2COT_tasks = json.load(f)

    print("📊 INIZIO VALUTAZIONE DATA-CENTRIC...\n")
    print(f"{'DB':<15} | {'P1-Prec':<8} {'P1-Rec':<8} | {'P2-Prec':<8} {'P2-Rec':<8}| {'P2S-Prec':<8} {'P2S-Rec':<8}| {'P2COT-Prec':<8} {'P2COT-Rec':<8}")
    print("-" * 90)

    p1_avg_prec, p1_avg_rec = 0, 0
    p2_avg_prec, p2_avg_rec = 0, 0
    p2S_avg_prec, p2S_avg_rec = 0, 0
    p2COT_avg_prec, p2COT_avg_rec = 0, 0

    for i in range(len(p1_tasks)):
        task = p1_tasks[i]
        db_id = task['db_id']
        quest = task['question']
        
        # --- GROUND TRUTH ---
        gt_data = execute_query(db_id, task['query'])

        # --- PIPELINE 1 (Text-to-SQL) ---
        p1_sql = task.get('predicted_query', '')
        p1_data = execute_query(db_id, p1_sql)
        p1_metrics = calculate_metrics(gt_data, p1_data)

        # --- PIPELINE 2 (Table QA) ---
        p2_raw_text = p2_tasks[i].get('predicted_table_qa_data', '[]')
        p2_data = extract_json_from_text(p2_raw_text) # Applichiamo la Regex!
        p2_metrics = calculate_metrics(gt_data, p2_data)

        # --- PIPELINE 2 Simple(Table QA) ---
        p2S_raw_text = p2S_tasks[i].get('predicted_table_qa_data', '[]')
        p2S_data = extract_json_from_text(p2S_raw_text) # Applichiamo la Regex!
        p2S_metrics = calculate_metrics(gt_data, p2S_data)

        # --- PIPELINE 2 Chain of Thought(Table QA) ---
        p2COT_raw_text = p2COT_tasks[i].get('predicted_table_qa_data')
        p2COT_data = extract_json_from_cot(p2COT_raw_text)
        p2COT_metrics = calculate_metrics(gt_data, p2COT_data)

        # Somme per la media finale
        p1_avg_prec += p1_metrics['precision']; p1_avg_rec += p1_metrics['recall']
        p2_avg_prec += p2_metrics['precision']; p2_avg_rec += p2_metrics['recall']
        p2S_avg_prec += p2S_metrics['precision']; p2S_avg_rec += p2S_metrics['recall']
        p2COT_avg_prec += p2COT_metrics['precision']; p2COT_avg_rec += p2COT_metrics['recall']

        print("-"*120)
        print(f"{quest:<100}")
        print(f"{db_id:<15}  | {p1_metrics['precision']:<8} {p1_metrics['recall']:<8} | {p2_metrics['precision']:<8} {p2_metrics['recall']:<8}| {p2S_metrics['precision']:<8} {p2S_metrics['recall']:<8}| {p2COT_metrics['precision']:<8} {p2COT_metrics['recall']:<8}")

    # Medie
    n = len(p1_tasks)
    print("-" * 120)
    print(f"{'MEDIE TOTALI':<15} | {round(p1_avg_prec/n, 2):<8} {round(p1_avg_rec/n, 2):<8} | {round(p2_avg_prec/n, 2):<8} {round(p2_avg_rec/n, 2):<8}| {round(p2S_avg_prec/n, 2):<8} {round(p2S_avg_rec/n, 2):<8}| {round(p2COT_avg_prec/n, 2):<8} {round(p2COT_avg_rec/n, 2):<8}")

if __name__ == "__main__":
    run_evaluation()