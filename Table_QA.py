import json
import sqlite3
import ollama
import os
import re

# --- CONFIGURAZIONE GLOBALE ---
MODEL_NAME = 'ibm/granite4.1:8b'
INPUT_JSON = './queries.json'
OUTPUT_JSON = './results/pipeline2_table_qa.json'
DB_BASE_DIR = './database'

# I tuoi database scelti
TARGET_DATABASES = ['world_1', 'orchestra']

def get_oracle_tables(db_id, query):
    """
    Estrae i nomi di tutte le tabelle del DB e controlla quali 
    sono menzionate nella query corretta (Oracle Tables).
    """
    conn = sqlite3.connect(f"{DB_BASE_DIR}/{db_id}/{db_id}.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    # Identifica le tabelle menzionate nella query (case-insensitive)
    query_lower = query.lower()
    oracle_tables = [table for table in all_tables if table.lower() in query_lower]
    return oracle_tables

def serialize_tables_tapex(db_id, table_names):
    conn = sqlite3.connect(f"{DB_BASE_DIR}/{db_id}/{db_id}.sqlite")
    cursor = conn.cursor()
    
    serialized_text = ""
    for table in table_names:
        cursor.execute(f"SELECT * FROM {table} LIMIT 30") # Mantieni il limite per la Context Window
        rows = cursor.fetchall()
        
        col_names = [description[0] for description in cursor.description]
        
        # Formattazione in stile TAPEX (Slide 63)
        serialized_text += f"Table: {table}\n"
        serialized_text += "[HEAD] " + " | ".join(col_names) + "\n"
        
        for row in rows:
            clean_row = [str(item).replace('\n', ' ') if item is not None else "NULL" for item in row]
            serialized_text += "[ROW] " + " | ".join(clean_row) + "\n"
            
        serialized_text += "\n"
        
    conn.close()
    return serialized_text
    
def run_pipeline_2():
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
        
    print(f"🚀 Avvio Pipeline 2 (Direct Table-QA) con {MODEL_NAME}...\n")
    
    for i, task in enumerate(tasks):
        db_id = task['db_id']
        question = task['question']
        gt_query = task['query']
        
        print(f"[{i+1}/{len(tasks)}] DB: {db_id} | Domanda: {question}")
        
        # 1. Troviamo le Oracle Tables
        oracle_tables = get_oracle_tables(db_id, gt_query)
        if not oracle_tables:
            print("⚠️ Nessuna oracle table trovata. Salto questa domanda.")
            continue
            
        # 2. Serializziamo i dati (IL CUORE DELLA PIPELINE 2)
        serialized_data = serialize_tables_to_markdown(db_id, oracle_tables)
        
        # 3. Prompting strategico (Richiediamo output JSON)
        system_prompt = f"""You are a data analysis AI. 
Read the provided database tables and answer the user's question directly.
Output your answer STRICTLY as a JSON list of lists, representing rows and columns. 
Do not include any other text, markdown formatting, or explanations.

Example format:
[["Mario", 30], ["Luigi", 25]]
NO TEXT INTRODUCTION
Database Tables:
{serialized_data}"""

        # 4. Invocazione modello
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question}
        ], options={
            'num_ctx': 8192
        })
        
        predicted_answer = response['message']['content'].strip()
        print(f"Risposta Generata: {predicted_answer}\n" + "-"*40)
        
        task['predicted_table_qa_data'] = predicted_answer
        task['oracle_tables_used'] = oracle_tables

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=4)
        
    print(f"✅ Pipeline 2 completata! Risultati salvati in {OUTPUT_JSON}")

if __name__ == "__main__":
    run_pipeline_2()
