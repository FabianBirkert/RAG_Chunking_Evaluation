# Bachelorarbeit: RAG Chunking & Retrieval Evaluation

Dieses Repository enthält die Experiment-Pipeline der Bachelorarbeit zur Wirkung verschiedener Chunking-Strategien auf Retrieval-Augmented-Generation (RAG). Untersucht wird ein 3x3-Design aus drei Chunkern (heuristic, semantic, agentic) und drei Retrievern (sparse, dense, hybrid), bewertet auf Retrieval- und Antwortqualität.

## Schnellstart

```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. Umgebungsvariablen konfigurieren
cp .env.template .env
# .env mit eigenen Werten befüllen

# 3. Pipeline ausführen
python main.py
```

## Voraussetzungen

- Python 3.10+
- OpenAI-kompatibler API-Zugang (siehe `.env.template`)

## Projektstruktur

```
BA_RAG_System/
├── config.py                  # Zentrale Parameter und Defaults
├── main.py                    # Orchestriert Schritte 1-4
├── requirements.txt           # Abhängigkeiten
├── .env.template              # Template für Umgebungsvariablen
├── src/
│   ├── step1_cleaning.py      # Datenaufbereitung (NQ -> Markdown, Filter)
│   ├── step2_chunking.py      # Chunking + Indexing (BM25, Chroma)
│   ├── step3_retrieval.py     # Retrieval (sparse/dense/hybrid) + Generation
│   └── step4_evaluation.py    # Evaluation (Retrieval + Antwortqualität)
└── data/
    ├── input/                 # Bereinigte CSV (Step 1)
    ├── indices/               # BM25 + Chroma Stores (Step 2)
    ├── output/                # Generation/Evaluation-Ergebnisse (Step 3/4)
    └── cache/                 # Embedding/LLM Cache
```

## Forschungsfragen

**Hauptforschungsfrage:**
> Wie beeinflussen unterschiedliche Chunking-Strategien die Effektivität von RAG-Systemen und welche Interdependenzen bestehen dabei mit der eingesetzten Retrieval-Methode?

**Unterforschungsfragen:**
1. Führen kontextsensitive Segmentierungsverfahren zu einer signifikanten Steigerung der Retrieval-Qualität im Vergleich zu heuristischen Baseline-Verfahren?
2. Inwieweit verbessert die Wahl des Chunkings die inhaltliche Korrektheit der Antworten und reduziert Halluzinationen?
3. Zeigen sich systematische qualitative Leistungsunterschiede der Chunking-Strategien in Abhängigkeit von der verwendeten Retrieval-Logik?

## Datensatz & Aufbereitung (Step 1)

| Aspekt | Beschreibung |
|--------|--------------|
| Quelle | Google Natural Questions (simplified) |
| Konvertierung | HTML -> Markdown, Entfernung von Wikipedia-Boilerplate |
| Filter | Genau eine Short Answer, Länge 10-100 Zeichen |
| Sampling | Deterministisch mit Seed 42 |
| Stichprobe | N=250 (anpassbar via `SAMPLE_SIZE` in [config.py](config.py)) |


**Automatischer Download:** Das Natural Questions-Dataset (~4.4 GB) wird beim ersten Pipeline-Start automatisch heruntergeladen und entpackt. Ein manueller Download ist nicht nötig. Der Fortschritt wird im Terminal angezeigt. Sollte der Download fehlschlagen, gibt das System eine klare Anleitung für den manuellen Download aus.

## Experimentelles Design (Step 2-3)

**Vollfaktoriell 3x3:** Chunking x Retrieval

### Chunking-Strategien

| Strategie | Beschreibung |
|-----------|--------------|
| **Heuristic** | Sliding Window, 512 Tokens, 10% Overlap, Satzgrenzen bevorzugt |
| **Semantic** | Satzfenster a 2 Sätze, neuer Chunk bei Cosine Similarity < 0.6 |
| **Agentic** | LLM extrahiert atomare Propositions, gruppiert max. 15 pro Chunk |

### Retrieval-Methoden

| Methode | Beschreibung |
|---------|--------------|
| **Sparse** | BM25 (invertierter Index) |
| **Dense** | ChromaDB mit `text-embedding-3-small` |
| **Hybrid** | Reciprocal Rank Fusion (RRF, k=60) |

**Weitere Parameter:**
- Top-k Retrieval: 10 Treffer pro Query
- Generation: `gpt-4o-mini`, Temperatur 0


## Evaluation (Step 4)

| Ebene | Metriken |
|-------|----------|
| **Retrieval** | Recall@10, MRR@10, nDCG@10 (via LLM-as-a-Judge) |
| **Generation** | RAGAS Faithfulness, RAGAS Factual Correctness, F1-Score |


**Ergebnisdateien:**
- [data/output/generation_results.csv](data/output/generation_results.csv): Alle generierten Antworten und Retrieval-Ergebnisse
- [data/output/evaluation_results.csv](data/output/evaluation_results.csv): Auswertung aller Metriken pro Sample
- [data/output/evaluation_summary.json](data/output/evaluation_summary.json): Aggregierte Metriken und Zusammenfassungen

## Nutzung

### Gesamte Pipeline

```bash
python main.py
```

### Einzelne Schritte

```bash
python main.py --step 1          # Cleaning
python main.py --step 2          # Chunking + Indexing
python main.py --step 3          # Retrieval + Generation
python main.py --step 4          # Evaluation
```

### Bereich ausführen

```bash
python main.py --from-step 2 --to-step 4
```

### Verbose-Modus

```bash
python main.py --verbose
```

## Konfiguration


Alle Parameter sind zentral in [config.py](config.py) definiert und können dort für eigene Experimente angepasst werden:

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| `SAMPLE_SIZE` | 250 | Anzahl der Dokumente (Stichprobengröße) |
| `RANDOM_SEED` | 42 | Reproduzierbarkeit der Stichprobe |
| `HEURISTIC_CHUNK_TOKENS` | 512 | Zielgröße heuristischer Chunks (Tokens) |
| `HEURISTIC_OVERLAP_TOKENS` | 51 | Überlappung zwischen Chunks (Tokens) |
| `SEMANTIC_SIMILARITY_THRESHOLD` | 0.6 | Schwellenwert für semantische Chunk-Trennung |
| `RETRIEVAL_TOP_K` | 10 | Anzahl abgerufener Chunks pro Query |
| `RRF_K` | 60 | Glättungsparameter für Hybrid Retrieval (RRF) |
| `LLM_MODEL` | gpt-4o-mini | Modell für Antwortgenerierung/Evaluation |
| `EMBEDDING_MODEL` | text-embedding-3-small | Embedding-Modell für Dense Retrieval |
