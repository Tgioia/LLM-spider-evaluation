import json
import sqlite3
import os

# --- CONFIGURAZIONE ---
INPUT_JSON = './results/pipeline1_text_to_sql.json'
OUTPUT_JSON = './results/pipeline1_executed_data.json'
DB_BASE_DIR = './database'

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
                # Converte numeri e altri tipi in stringhe per un confronto standard
                norm_row.append(str(item))
        normalized_list.append(norm_row)
    
    # Ordiniamo la lista finale in modo che l'ordine delle righe 
    # non influenzi metriche come Cell Precision/Recall (a meno che non sia richiesto ORDER BY)
    return sorted(normalized_list)

def execute_query(db_id, query):
    if not query or query.strip() == "":
        return [], "Empty query"

    db_path = os.path.join(DB_BASE_DIR, db_id, f"{db_id}.sqlite")
    
    try:
        # Timeout breve per evitare query che vanno in loop infinito
        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Estraiamo i risultati grezzi
        raw_results = cursor.fetchall()
        conn.close()
        
        # Normalizziamo l'output
        return normalize_data(raw_results), "Success"
        
    except sqlite3.Error as e:
        # CATTURIAMO L'ERRORE! Il sistema non andrà in crash.
        # Restituiamo una lista vuota, che al momento della valutazione porterà a Recall = 0.
        return [], f"SQL Error: {str(e)}"
    except Exception as e:
        return [], f"Execution Error: {str(e)}"

def run_execution_step():
    print("Starting SQL query execution on local databases\n")
    
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
        
    results_with_data = []
    success_count = 0
    error_count = 0
    
    for i, task in enumerate(tasks):
        db_id = task['db_id']
        
        # 1. Eseguiamo la query vera (Ground Truth)
        truth_data, truth_status = execute_query(db_id, task['query'])
        
        # 2. Eseguiamo la query generata da Granite 4.1 (Predicted)
        pred_data, pred_status = execute_query(db_id, task['predicted_query'])
        
        if pred_status == "Success":
            success_count += 1
        else:
            error_count += 1
            print(f"Query Error {i+1} ({db_id}): {pred_status}")
            print(f"Poor Query syntax: {task['predicted_query']}")
        
        # Salviamo tutto il pacchetto per la valutazione finale
        task['ground_truth_data'] = truth_data
        task['predicted_data'] = pred_data
        task['execution_status'] = pred_status
        
        results_with_data.append(task)
        
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_with_data, f, indent=4)
        
    print(f"\nCompleted execution!")
    print(f"📊 Stats: {success_count} successfull, {error_count} failed.")
    print(f"💾 Extracted and normalized data saved in {OUTPUT_JSON}")

if __name__ == "__main__":
    run_execution_step()