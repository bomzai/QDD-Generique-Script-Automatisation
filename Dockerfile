FROM python:3.11-slim

WORKDIR /app

# Optionnel : évite les .pyc, logs plus lisibles
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Installation des dépendances
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copie du code source dans le conteneur
COPY . /app

# Par défaut, on lance le script principal
CMD ["./run.sh"]