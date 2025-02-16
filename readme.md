# SQL Retriever Agent Demo
- A quick demo of using Llama-Index to build a SQL Retrieval Agent for Retrieval Augmented Generation (RAG).
- The intention is that there is either a human or another LLM Agent interacting with the SQL Retrieval Agent for various queries.
- For example, a human may interact with a Multi-Tool Agent ChatBot that has access to a tool that queries the SQL Agent for additional information from the database.

# Setup
1. Create a virtual environment and install the dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Create a .env file in the root of the folder with the following (changeable)
```.env
DB_USER="user"
DB_PASSWORD="password"
DB_NAME="citus"
OPENAI_API_KEY=...
```
3. Build the database and insert dummy data (ensure docker desktop is running first.)
```bash
make build_db
```
Or alternatively, for enhanced synthetic data use:
```bash
make build_enhanced_db
```
**WARNING: It uses GPT-4o-mini to generate 1000 fake meeting notes.**
4. Explore the demo.ipynb file in the root
- There are two implementations of TextToSQL retrieval
    - The first is the Llama-Index chat enginer
    - The second is a simplified custom implementation using the Llama-Index framework
5. Finally, to destroy everything
```bash
make destroy_db
```