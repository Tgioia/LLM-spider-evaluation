import json
import sqlite3
import ollama
import os

# --- CONFIGURAZIONE GLOBALE ---
MODEL_NAME = 'ibm/granite4.1:8b'
INPUT_JSON = './queries.json'
OUTPUT_JSON = './results/pipeline1_text_to_sql.json'
DB_BASE_DIR = './database'

# Impostiamo a mano i database come richiesto
TARGET_DATABASES = ['world_1', 'orchestra']

def load_schemas():
    schemas = {}
    for db in TARGET_DATABASES:
        conn = sqlite3.connect(f"{DB_BASE_DIR}/{db}/{db}.sqlite")
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        schemas[db] = "\n\n".join([row[0] for row in cursor.fetchall() if row[0]])
        conn.close()
    return schemas

def run_pipeline():
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
        
    print(f"Starting the Text-to-SQL pipeline with {MODEL_NAME}\n")
    
    schemas = load_schemas()
    
    for i, task in enumerate(tasks):
        db_id = task['db_id']
        print(f"[{i+1}/{len(tasks)}] DB: {db_id} | Question: {task['question']}")
        
        system_prompt = f"You are an expert SQL developer. Write a valid SQLite query to answer the user's question.\nSchema:\n{schemas[db_id]}\nRespond ONLY with the SQL query."

        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': task['question']}
        ])
        
        predicted_sql = response['message']['content'].strip()
        print(f"Predizione: {predicted_sql}\n" + "-"*40)
        
        task['predicted_query'] = predicted_sql

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=4)
        
    print(f"Done! Output saved in {OUTPUT_JSON}")

if __name__ == "__main__":
    run_pipeline()