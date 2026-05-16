"""# Neural Table Representations - Assignment

This repository contains the code for the course assignment "Models and Practice of Neural Table Representations". 

The project evaluates and compares two different paradigms using a local LLM (`ibm/granite4.1:8b`) on a subset of the **Spider** dataset (`world_1` and `orchestra`):
1. **Text-to-SQL:** The LLM generates an SQL query that is executed on the database.
2. **Direct Table-QA:** The LLM reads serialized tables (TAPEX format) and directly generates the answer.

## 🛠️ Setup Instructions

To run the code, you need Python and [Ollama](https://ollama.com/) installed on your machine.

**1. Install Python dependencies:**
```bash
pip install -r requirements.txt
