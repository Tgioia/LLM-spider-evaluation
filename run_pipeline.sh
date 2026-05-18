#!/bin/bash
set -e  # Se uno script va in errore, si ferma tutto automaticamente

echo "Running Text-to-SQL..."
python Text_to_SQL.py
python TtS_exec_and_norm.py
echo "Running Table-QA..."
python Table_QA.py
python Table_QA_simple.py
python Table_QA_COT.py

echo "Evaluating the metrics"
python evaluate_metrics.py

