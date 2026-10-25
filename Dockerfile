FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

CMD ["python", "-m", "crypto_paper_trader.main", "--mode", "live", "--auto-select", "--interval", "1h"]
