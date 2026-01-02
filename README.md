# Pipeline QDD - Guide Développeur

## Présentation

Ce pipeline automatise la génération des controles QDD techniques en une chaîne de 4 scripts Python qui :
1. Récupèrent la définition des données au format DBML.
2. Génèrent automatiquement les tests de qualité.
3. Configurent et exécutent Soda pour exécuter les tests dans le Datahub.
4. Sauvegardent les résultats dans une table T_TESTCASE_RESULT.

---

## Architecture du pipeline

```
DBML => Génération tests => Génération des Checks Soda => Exécution des tests => Sauvegarder les résultats dans Snowflake.
```

### Scripts principaux

| Script | Rôle | Entrées | Sorties |
|--------|------|---------|---------|
| `1_generate_testcases.py` | Crée les tests QDD | Ficher DBML, la table T_METRIQUE | Fichier SQL pour alimenter T_TESTCASE |
| `2_generate_soda_checks.py` | Génère les checks Soda | la table T_TESTCASE | fichiers checks_*.yml (un fichier par table)|
| `3_generate_soda_config.py` | Génère la configuration de la connexion Soda <> Snowflake | Variables d'env | dev-config.yml |
| `4_push_results_to_snowflake.py` | Sauvgrade les résultats renvoyés par Soda | results.json | alimentation de la table T_TESTCASE_RESULT |

---

## 1. Génération des tests

**Objectif** : Analyser le DBML et créer automatiquement les tests de qualité

### Principe

Le script parcourt le DBML et génère **4 types de tests** :

| Type | Détection | Test généré |
|------|-----------|-------------|
| **Complétude** | Colonnes PK ou NOT NULL | Vérifier qu'aucune valeur n'est manquante |
| **Unicité** | Clés primaires | Vérifier l'absence de doublons |
| **Intégrité** | Clés étrangères | Vérifier la cohérence des références |
| **Traçabilité** | Colonne `COMMENT` dans Snowflake via la table d'information `COLUMNS` dans le schema `INFORMATION_SCHEMA`  | Si cette colonne est présente, vérifier qu'elle contient une description |

→ Génère 4 tests :
- Complétude Ex : `Test de complétude - Table CUSTOMER_SCHEMA.CUSTOMER, Colonne C_CUSTKEY`
- Unicité Ex: `Test d'unicité (PK) - Table CUSTOMER_SCHEMA.CUSTOMER, Colonnes C_CUSTKEY`
- Intégrité Ex: `Test d'intégrité - Table CUSTOMER_SCHEMA.CUSTOMER, Colonne C_NATIONKEY sur CUSTOMER_SCHEMA.NATION.N_NATIONKEY de référence`
- Traçabilité Ex: `Test de traçabilité - Table CUSTOMER_SCHEMA.CUSTOMER, Colonne C_CUSTKEY`

```sql
INSERT INTO T_TESTCASE (TST_NOM_TEST, TST_TABLE_CIBLE, TST_CHAMP_CIBLE, ...)
SELECT ...
WHERE NOT EXISTS (
  -- Évite les doublons sur table+champ+métrique
);
```

---

## 2. Génération des checks Soda

**Objectif** : Transformer les tests QDD en fichiers YAML pour Soda

### Fonctionnement

1. Lit les testcases depuis `T_TESTCASE` (avec filtre sur dates de validité)
2. Convertit chaque test en check SodaCL selon son type
3. Regroupe par table et écrit un fichier `.yml` par table

### Mapping QDD → Soda

| Type QDD | Check Soda | Exemple |
|----------|------------|---------|
| Complétude | `missing_count` | `missing_count(CLI_NOM) = 0` |
| Unicité | `duplicate_count` | `duplicate_count(CLI_ID) = 0` |
| Intégrité | `invalid_percent` | `invalid_percent(CLI_PAYS_ID) <= 5` |
| Traçabilité | `traceability_ratio` | `traceability_ratio(CLI_DESC) = 0` |

### Sortie

```yaml
# customer_schema_customer.yml
table_name: CUSTOMER_SCHEMA.CUSTOMER
checks:
  - missing_count(C_CUSTKEY) = 0:
      name: COMPLETUDE-C_CUSTKEY
  - duplicate_count(C_CUSTKEY) = 0:
      name: UNICITE-C_CUSTKEY
```

---

## 3. Configuration Soda

**Objectif** : Créer le fichier de connexion Snowflake pour Soda

### Configuration minimale

```bash
SNOWFLAKE_ACCOUNT=xyz12345
SNOWFLAKE_USER=QDD_USER
SNOWFLAKE_ROLE=QDD_ROLE
SNOWFLAKE_WAREHOUSE=WH_QDD
SNOWFLAKE_DATABASE=QDD_DATABASE
SNOWFLAKE_SCHEMA=QDD
```

### Sortie

```yaml
# configuration.yml
data_source:
  snowflake_qdd:
    type: snowflake
    account: xyz12345
    user: QDD_USER
    role: QDD_ROLE
    warehouse: WH_QDD
    database: QDD_DATABASE
    schema: QDD
```

---

## 4. Remontée des résultats

**Objectif** : Insérer les résultats Soda dans Snowflake pour analyse

### Fonctionnement

1. Lit `results.json` produit par Soda
2. Normalise les données (statuts, noms de tables, déduplication)
3. Insère dans `T_TESTCASE_RESULT`

### Structure des résultats

```json
{
  "check_name": "COMPLETUDE-CLI_NOM",
  "table": "PRODUIT.CLIENT",
  "column": "CLI_NOM",
  "outcome": "pass",
  "failed_rows": 0,
  "total_rows": 15000
}
```

---

## Exécution du container Docker
Le projet contient le fichier run.sh qui execute les scripts
Le Dockerfile inclus toutes les dépendances nécessaires.

### Pré-requis
- Avoir Docker installés sur la machine.
- Modifier les variables d'environnement dans le fichier `.env.docker` pour configurer la connexion à Snowflake.
- Avoir créé la paire de clé SSH et ajouté la clé privé dans le dossier `secrets` à la racine du projet (voir le fichier KEY_PAIR.md pour plus d'infos).

### Commande d'exécution
La premiere fois, lancer la commande suivante pour build et exécuter le container Docker :
```sh
docker compose up --build
```

Ensuite, pour relancer le pipeline, il suffit d'exécuter :
```sh
docker compose up
```

---

## Dépendances Python

```txt
snowflake-connector-python
soda-core-snowflake
pydbml
PyYAML
requests
cryptography
```

---

## Points d'attention

### Idempotence
Tous les scripts sont conçus pour être **réexécutables**.