# Snowflake Key Pair Authentication  
_Génération et configuration des clés pour Soda & Framework QDD_

## Objectif

Ce document explique comment :
- Générer une paire de clés RSA compatible Snowflake
- Configurer l’authentification par clé (key-pair)

---

## 1. Génération de la paire de clés (OpenSSL)

### Pré-requis
- OpenSSL ≥ 1.1
- macOS / Linux / WSL (Windows via WSL recommandé)

### 1.1 Générer la clé privée (PKCS8)

```bash
openssl genpkey \
  -algorithm RSA \
  -out snowflake_key.p8 \
  -pkeyopt rsa_keygen_bits:2048
```

### 1.2 Générer la clé publique

```bash
openssl rsa \
  -in snowflake_key.p8 \
  -pubout \
  -out snowflake_key.pub
```

## 2. Déclaration de la clé publique dans Snowflake

### 2.1 Extraire la clé publique au format Snowflake

```bash
cat snowflake_key.pub | sed '1d;$d' | tr -d '\n'
```
➡️ Copier la chaîne Base64 obtenue

### 2.2 Associer la clé à un utilisateur Snowflake 

```sql
ALTER USER QDD_USER
SET RSA_PUBLIC_KEY='MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A...';
```