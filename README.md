 # convpool-realtime

## Quickstart
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn --app-dir src convpool.main:app --reload --port 7860
# open /sender and /receiver
