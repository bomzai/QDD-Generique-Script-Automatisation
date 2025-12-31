#!/bin/bash

set -e

echo "Démarrage du pipeline QDD"

echo "1/4 Génération et synchronisation des Testcases..."
python scripts/1_generate_testcases.py

echo "2/4 Génération des checks Soda..."
python scripts/2_generate_soda_checks.py

echo "3/4 Vérification de la connexion Snowflake..."
python scripts/3_generate_soda_config.py ./soda/configs/dev-config.yml

echo "Lancement du Scan Soda..."
soda scan -d snowflake -c ./soda/configs/dev-config.yml target/soda_checks/*.yml -srf target/results.json || true

echo "4/4 Génération de la configuration Soda..."
python scripts/4_push_results_to_snowflake.py ./target/results.json

echo "Pipeline terminé avec succès"