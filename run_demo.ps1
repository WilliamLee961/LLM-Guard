param(
    [int]$Port = 8787
)

python -m uvicorn demo_api:app --host 0.0.0.0 --port $Port --reload
