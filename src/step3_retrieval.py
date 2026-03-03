# =============================================================================
# STEP 3: RETRIEVAL & ANSWER-GENERATION
# =============================================================================
"""
step3_retrieval.py - Retrieval und Answer-Generation

Dieses Modul implementiert den dritten Schritt der RAG-Pipeline:
das Abrufen relevanter Chunks aus den Indizes und die Generierung
von Antworten mittels LLM. Es unterstützt drei Retrieval-Methoden
(Sparse, Dense, Hybrid) und kombiniert diese mit allen verfügbaren
Chunking-Strategien.

Hauptkomponenten:
    - IndexManager: Verwaltet das Laden und Cachen von Indizes
    - Retrieval-Funktionen: sparse_retrieve, dense_retrieve, hybrid_retrieve
    - Generation: Antwortgenerierung auf Basis der abgerufenen Chunks
    - Ergebnis-Export: Serialisierung der Ergebnisse als CSV

Typische Verwendung:
    from src.step3_retrieval import run_step3
    output_path = run_step3()
"""

import os
import json
import pickle
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import pandas as pd
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# Lade Environment-Variablen
load_dotenv()

# Logger für dieses Modul
logger = logging.getLogger("RAG-Pipeline.Step3")

# Fallback-Logging für direkten Aufruf
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for lib in ["httpx", "openai", "chromadb"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

# Import config und utils
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR,
    RETRIEVAL_TOP_K,
    RETRIEVAL_METHODS,
    RRF_K,
    CHUNKING_STRATEGIES,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_PROMPT,
    LLM_MODEL,
    TEMPERATURE,
    get_chroma_path,
    get_bm25_path,
    get_chunk_ids_path,
)
from src.utils import (
    ensure_parent_directory,
    async_llm_call,
    async_get_single_embedding,
    sync_get_embeddings,
    tokenize_for_bm25,
    run_sync_in_executor,
    get_semaphore,
)


OUTPUT_CSV_PATH = DATA_DIR / "input" / "nq_validation_cleaned.csv"
CSV_ENCODING = "utf-8"
GENERATION_OUTPUT_CSV_PATH = DATA_DIR / "output" / "generation_results.csv"


@dataclass
class RetrievedChunk:
    """
    Repräsentiert einen abgerufenen Text-Chunk.

    Attributes:
        rank: Die Position im Ranking (1-basiert).
        chunk_id: Die eindeutige ID des Chunks.
        source_id: Die ID des Quelldokuments.
        chunk_text: Der Textinhalt des Chunks.
        chunk_length: Die Länge des Chunks in Zeichen.
        score: Der Relevanz-Score (optional).
    """
    rank: int
    chunk_id: str
    source_id: str
    chunk_text: str
    chunk_length: int
    score: Optional[float] = None


@dataclass
class TokenUsage:
    """
    Speichert die Token-Nutzung eines LLM-Aufrufs.

    Attributes:
        input_tokens: Anzahl der Eingabe-Tokens.
        output_tokens: Anzahl der Ausgabe-Tokens.
        total_tokens: Gesamtanzahl der verbrauchten Tokens.
    """
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class GenerationResult:
    """
    Enthält das vollständige Ergebnis einer Frage-Antwort-Generierung.

    Attributes:
        chunking_strategy: Die verwendete Chunking-Strategie.
        retrieval_method: Die verwendete Retrieval-Methode.
        question_id: Die ID der Frage.
        document_id: Die ID des zugehörigen Dokuments.
        question: Der Fragetext.
        short_answer: Die erwartete Kurzantwort.
        generated_answer: Die vom LLM generierte Antwort.
        retrieved_chunks: Die abgerufenen Chunks.
        token_usage: Die Token-Nutzungsstatistik.
    """
    chunking_strategy: str
    retrieval_method: str
    question_id: str
    document_id: str
    question: str
    short_answer: str
    generated_answer: str
    retrieved_chunks: List[RetrievedChunk]
    token_usage: TokenUsage


class IndexManager:
    """
    Verwaltet das Laden und Cachen von Retrieval-Indizes.

    Diese Klasse stellt eine zentrale Schnittstelle für den Zugriff auf
    ChromaDB-Collections und BM25-Indizes bereit und implementiert
    Caching um wiederholte I/O-Operationen zu vermeiden.

    Attributes:
        _chroma_collections: Cache für geladene ChromaDB-Collections.
        _bm25_indices: Cache für geladene BM25-Indizes.
        _chunk_ids: Mapping von Strategie zu Chunk-ID-Listen.
        _chunk_texts: Mapping von Strategie zu Chunk-Texten.
        _chunk_metadata: Mapping von Strategie zu Chunk-Metadaten.
    """
    
    def __init__(self):
        self._chroma_collections: Dict[str, Any] = {}
        self._bm25_indices: Dict[str, BM25Okapi] = {}
        self._chunk_ids: Dict[str, List[str]] = {}
        self._chunk_texts: Dict[str, Dict[str, str]] = {}  # strategy -> {chunk_id: text}
        self._chunk_metadata: Dict[str, Dict[str, Dict]] = {}  # strategy -> {chunk_id: metadata}
    
    def preload(self, strategy: str):
        """
        Lädt alle Indizes für eine Strategie vor.

        Stellt sicher, dass sowohl ChromaDB als auch BM25-Index
        geladen und gecached sind bevor Anfragen verarbeitet werden.

        Args:
            strategy: Die Chunking-Strategie (heuristic, semantic, agentic).
        """
        logger.info(f"Preloading indices for {strategy}...")
        self.load_chroma_collection(strategy)
        self.load_bm25_index(strategy)

    def load_chroma_collection(self, strategy: str):
        """
        Lädt eine ChromaDB-Collection für eine Strategie.

        Prüft zunächst den Cache und lädt die Collection nur bei Bedarf.
        Cached zusätzlich alle Chunk-Texte und Metadaten.

        Args:
            strategy: Die Chunking-Strategie.

        Returns:
            Die geladene ChromaDB-Collection.

        Raises:
            FileNotFoundError: Wenn der ChromaDB-Pfad nicht existiert.
        """
        if strategy in self._chroma_collections:
            return self._chroma_collections[strategy]
        
        chroma_path = get_chroma_path(strategy)
        # Stelle sicher, dass das Verzeichnis existiert
        if not os.path.exists(chroma_path):
             logger.error(f"Chroma path does not exist: {chroma_path}")
             raise FileNotFoundError(f"Chroma path not found: {chroma_path}")

        client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection_name = f"{strategy}_chunks"
        try:
            collection = client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"Failed to get collection {collection_name}: {e}")
            raise

        self._chroma_collections[strategy] = collection
        
        # Cache für Chunk-Texte und Metadaten
        all_data = collection.get(include=['documents', 'metadatas'])
        self._chunk_texts[strategy] = dict(zip(all_data['ids'], all_data['documents']))
        self._chunk_metadata[strategy] = dict(zip(all_data['ids'], all_data['metadatas']))
        
        return collection
    
    def load_bm25_index(self, strategy: str) -> Tuple[BM25Okapi, List[str]]:
        """
        Lädt einen BM25-Index mit den zugehörigen Chunk-IDs.

        Prüft zunächst den Cache und deserialisiert die Index-Dateien
        nur bei Bedarf.

        Args:
            strategy: Die Chunking-Strategie.

        Returns:
            Ein Tuple aus (BM25-Index, Liste der Chunk-IDs).
        """
        if strategy in self._bm25_indices:
            return self._bm25_indices[strategy], self._chunk_ids[strategy]
        
        bm25_path = get_bm25_path(strategy)
        chunk_ids_path = get_chunk_ids_path(strategy)
        
        with open(bm25_path, 'rb') as f:
            bm25 = pickle.load(f)
        
        with open(chunk_ids_path, 'rb') as f:
            chunk_ids = pickle.load(f)
        
        self._bm25_indices[strategy] = bm25
        self._chunk_ids[strategy] = chunk_ids
        
        return bm25, chunk_ids
    
    def get_chunk_text(self, strategy: str, chunk_id: str) -> str:
        """
        Gibt den Textinhalt eines Chunks zurück.

        Args:
            strategy: Die Chunking-Strategie.
            chunk_id: Die eindeutige Chunk-ID.

        Returns:
            Der Chunk-Text oder ein leerer String.
        """
        if strategy not in self._chunk_texts:
            self.load_chroma_collection(strategy)
        return self._chunk_texts[strategy].get(chunk_id, "")
    
    def get_chunk_metadata(self, strategy: str, chunk_id: str) -> Dict:
        """
        Gibt die Metadaten eines Chunks zurück.

        Args:
            strategy: Die Chunking-Strategie.
            chunk_id: Die eindeutige Chunk-ID.

        Returns:
            Ein Dictionary mit den Chunk-Metadaten.
        """
        if strategy not in self._chunk_metadata:
            self.load_chroma_collection(strategy)
        return self._chunk_metadata[strategy].get(chunk_id, {})


# Globaler Index-Manager
index_manager = IndexManager()


def sparse_retrieve(
    query: str,
    strategy: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> List[RetrievedChunk]:
    """
    Führt lexikalisches Retrieval mittels BM25 durch.

    Tokenisiert die Anfrage und berechnet BM25-Scores gegen
    alle Chunks der angegebenen Strategie.

    Args:
        query: Die Suchanfrage.
        strategy: Die Chunking-Strategie.
        top_k: Anzahl der zurückzugebenden Top-Ergebnisse.

    Returns:
        Eine nach Relevanz sortierte Liste von RetrievedChunk-Objekten.
    """
    bm25, chunk_ids = index_manager.load_bm25_index(strategy)
    
    # Anfrage tokenisieren
    query_tokens = tokenize_for_bm25(query)
    
    # BM25-Scores berechnen
    scores = bm25.get_scores(query_tokens)
    
    # Top-k Indizes sortieren
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        chunk_id = chunk_ids[idx]
        chunk_text = index_manager.get_chunk_text(strategy, chunk_id)
        metadata = index_manager.get_chunk_metadata(strategy, chunk_id)
        
        results.append(RetrievedChunk(
            rank=rank,
            chunk_id=chunk_id,
            source_id=metadata.get('source_id', ''),
            chunk_text=chunk_text,
            chunk_length=len(chunk_text),
            score=float(scores[idx]),
        ))
    
    return results


async def dense_retrieve(
    query: str,
    strategy: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> List[RetrievedChunk]:
    """
    Führt semantisches Retrieval mittels Vektorähnlichkeit durch.

    Berechnet ein Embedding für die Anfrage und sucht die ähnlichsten
    Chunks in der ChromaDB-Collection.

    Args:
        query: Die Suchanfrage.
        strategy: Die Chunking-Strategie.
        top_k: Anzahl der zurückzugebenden Top-Ergebnisse.

    Returns:
        Eine nach Similarity sortierte Liste von RetrievedChunk-Objekten.
    """
    collection = index_manager.load_chroma_collection(strategy)
    
    # Anfrage-Embedding berechnen
    query_embedding = await async_get_single_embedding(query)
    
    # ChromaDB-Abfrage (blockierende I/O) - via Executor
    def do_query():
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
    
    query_result = await run_sync_in_executor(do_query)
    
    results = []
    ids = query_result['ids'][0]
    documents = query_result['documents'][0]
    metadatas = query_result['metadatas'][0]
    distances = query_result['distances'][0]
    
    for rank, (chunk_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), 1):
        results.append(RetrievedChunk(
            rank=rank,
            chunk_id=chunk_id,
            source_id=meta.get('source_id', ''),
            chunk_text=doc,
            chunk_length=len(doc),
            score=float(1 - dist) if dist else None,  # Konvertiere Distanz zu Ähnlichkeit
        ))
    
    return results


async def hybrid_retrieve(
    query: str,
    strategy: str,
    top_k: int = RETRIEVAL_TOP_K,
    rrf_k: int = RRF_K,
) -> List[RetrievedChunk]:
    """
    Führt hybrides Retrieval mit Reciprocal Rank Fusion durch.

    Kombiniert die Ergebnisse aus Sparse- und Dense-Retrieval mittels
    RRF-Algorithmus zu einem einheitlichen Ranking.

    Args:
        query: Die Suchanfrage.
        strategy: Die Chunking-Strategie.
        top_k: Anzahl der zurückzugebenden Top-Ergebnisse.
        rrf_k: Der k-Parameter für die RRF-Berechnung.

    Returns:
        Eine nach RRF-Score sortierte Liste von RetrievedChunk-Objekten.
    """
    # Sparse- und Dense-Retrieval parallel ausführen
    sparse_task = run_sync_in_executor(sparse_retrieve, query, strategy, top_k)
    dense_task = dense_retrieve(query, strategy, top_k)
    sparse_results, dense_results = await asyncio.gather(sparse_task, dense_task)
    
    # RRF-Fusion: Score(d) = sum(1 / (k + rank_m(d))) für m in {sparse, dense}
    rrf_scores: Dict[str, float] = defaultdict(float)
    chunk_data: Dict[str, RetrievedChunk] = {}
    
    # Kombiniere Scores aus beiden Quellen (Inline statt lokaler Funktion)
    for result in sparse_results:
        rrf_scores[result.chunk_id] += 1.0 / (rrf_k + result.rank)
        if result.chunk_id not in chunk_data:
            chunk_data[result.chunk_id] = result
    
    for result in dense_results:
        rrf_scores[result.chunk_id] += 1.0 / (rrf_k + result.rank)
        if result.chunk_id not in chunk_data:
            chunk_data[result.chunk_id] = result
    
    # Sortiere nach RRF-Score (heapq.nlargest wäre für sehr große Listen effizienter)
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    
    # Erstelle Ergebnisliste
    return [
        RetrievedChunk(
            rank=rank,
            chunk_id=chunk_id,
            source_id=chunk_data[chunk_id].source_id,
            chunk_text=chunk_data[chunk_id].chunk_text,
            chunk_length=chunk_data[chunk_id].chunk_length,
            score=rrf_scores[chunk_id],
        )
        for rank, chunk_id in enumerate(sorted_ids, 1)
    ]


async def retrieve(
    query: str,
    strategy: str,
    method: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> List[RetrievedChunk]:
    """
    Dispatcher-Funktion für Retrieval-Operationen.

    Wählt basierend auf der angegebenen Methode die entsprechende
    Retrieval-Implementierung aus und führt diese aus.

    Args:
        query: Die Suchanfrage.
        strategy: Die Chunking-Strategie.
        method: Die Retrieval-Methode (sparse, dense, hybrid).
        top_k: Anzahl der zurückzugebenden Top-Ergebnisse.

    Returns:
        Eine sortierte Liste von RetrievedChunk-Objekten.

    Raises:
        ValueError: Bei unbekannter Retrieval-Methode.
    """
    if method == "sparse":
        return await run_sync_in_executor(sparse_retrieve, query, strategy, top_k)
    elif method == "dense":
        return await dense_retrieve(query, strategy, top_k)
    elif method == "hybrid":
        return await hybrid_retrieve(query, strategy, top_k)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")


def build_context_string(chunks: List[RetrievedChunk]) -> str:
    """
    Erstellt einen formatierten Kontext-String für die LLM-Generation.

    Formatiert die abgerufenen Chunks in XML-ähnliche Document-Tags
    für die Einbettung im Generation-Prompt.

    Args:
        chunks: Die abgerufenen Chunks in Ranking-Reihenfolge.

    Returns:
        Der formatierte Kontext-String.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f'<document index="{i}">\n{chunk.chunk_text}\n</document>')
    return '\n\n'.join(parts)


async def generate_answer(
    question: str,
    chunks: List[RetrievedChunk],
) -> Tuple[str, TokenUsage]:
    """
    Generiert eine Antwort auf eine Frage basierend auf Kontext-Chunks.

    Konstruiert einen Prompt aus den abgerufenen Chunks und der Frage,
    sendet diesen an das LLM und gibt die Antwort zurück.

    Args:
        question: Die zu beantwortende Frage.
        chunks: Die als Kontext zu verwendenden Chunks.

    Returns:
        Ein Tuple aus (generierte_Antwort, Token-Nutzungsstatistik).
    """
    context_string = build_context_string(chunks)
    
    system_prompt = GENERATION_SYSTEM_PROMPT
    user_prompt = GENERATION_USER_PROMPT.format(
        context_string=context_string,
        question=question
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    response = await async_llm_call(messages)
    
    token_usage = TokenUsage(
        input_tokens=response['input_tokens'],
        output_tokens=response['output_tokens'],
        total_tokens=response['total_tokens'],
    )
    
    return response['content'], token_usage


async def process_question(
    row: Dict,
    strategy: str,
    method: str,
) -> GenerationResult:
    """
    Verarbeitet eine einzelne Frage für eine Strategie-Methode-Kombination.

    Orchestriert den vollständigen Retrieval-Generation-Workflow für
    eine Frage und sammelt alle relevanten Ergebnisdaten.

    Args:
        row: Ein Dictionary mit den Fragedaten aus der CSV.
        strategy: Die anzuwendende Chunking-Strategie.
        method: Die anzuwendende Retrieval-Methode.

    Returns:
        Ein GenerationResult-Objekt mit allen Ergebnissen.
    """
    question = row['Question']
    question_id = str(row['ID'])
    document_id = str(row['DocumentID'])
    short_answer = str(row['short_answer'])
    
    # Retrieval
    chunks = await retrieve(question, strategy, method)
    
    # Generation
    generated_answer, token_usage = await generate_answer(question, chunks)
    
    return GenerationResult(
        chunking_strategy=strategy,
        retrieval_method=method,
        question_id=question_id,
        document_id=document_id,
        question=question,
        short_answer=short_answer,
        generated_answer=generated_answer,
        retrieved_chunks=chunks,
        token_usage=token_usage,
    )


async def safe_process_question(
    row_idx: int, 
    row: Dict, 
    strategy: str, 
    method: str
) -> Tuple[int, Any]:
    """
    Wrapper für process_question mit Exception-Handling.

    Fängt Fehler bei der Verarbeitung ab und gibt sie zusammen
    mit dem Index zurück für spätere Fehlerbehandlung.

    Args:
        row_idx: Der Index der Zeile in der Datenliste.
        row: Die Zeilendaten.
        strategy: Die Chunking-Strategie.
        method: Die Retrieval-Methode.

    Returns:
        Ein Tuple aus (Zeilenindex, Ergebnis oder Exception).
    """
    try:
        result = await process_question(row, strategy, method)
        return (row_idx, result)
    except Exception as e:
        return (row_idx, e)


async def run_step3_async() -> List[GenerationResult]:
    """
    Führt die asynchrone Retrieval- und Generierungspipeline aus.

    Orchestriert die Verarbeitung aller Fragen über alle Kombinationen
    von Chunking-Strategien und Retrieval-Methoden mit Fortschrittsanzeige.

    Returns:
        Eine Liste aller GenerationResult-Objekte.
    """
    from tqdm import tqdm
    from tqdm.asyncio import tqdm_asyncio
    
    logger.info("STEP 3: Retrieval + Generation")
    
    # Lade Step-1 CSV
    csv_path = str(OUTPUT_CSV_PATH)
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, encoding=CSV_ENCODING)
    rows = df.to_dict('records')
    
    total_combinations = len(CHUNKING_STRATEGIES) * len(RETRIEVAL_METHODS)
    logger.info(f"Processing {len(rows)} questions × {total_combinations} combinations")
    
    results = []
    
    # Verarbeite alle Kombinationen mit Fortschrittsbalken
    combinations = [
        (strategy, method)
        for strategy in CHUNKING_STRATEGIES
        for method in RETRIEVAL_METHODS
    ]
    
    with tqdm(total=len(combinations), desc="Strategy/Method", position=0) as pbar:
        current_strategy = None
        
        for strategy, method in combinations:
            # Lade Indizes vor wenn neue Strategie
            if strategy != current_strategy:
                try:
                    index_manager.preload(strategy)
                    current_strategy = strategy
                except Exception as e:
                    logger.error(f"Failed to preload indices for {strategy}: {e}")
                    pbar.update(1)
                    continue
            
            pbar.set_postfix_str(f"{strategy}/{method}")
            
            tasks = [
                safe_process_question(i, row, strategy, method) 
                for i, row in enumerate(rows)
            ]
            
            indexed_results = await tqdm_asyncio.gather(
                *tasks,
                desc=f"  Questions",
                total=len(tasks),
                position=1,
                leave=False,
            )
            
            for row_idx, result in indexed_results:
                if isinstance(result, Exception):
                    logger.warning(f"Error for question {rows[row_idx]['ID']}: {result}")
                    # Erstelle Fallback-Ergebnis
                    results.append(GenerationResult(
                        chunking_strategy=strategy,
                        retrieval_method=method,
                        question_id=str(rows[row_idx]['ID']),
                        document_id=str(rows[row_idx]['DocumentID']),
                        question=rows[row_idx]['Question'],
                        short_answer=str(rows[row_idx]['short_answer']),
                        generated_answer="Fehler bei der Generierung",
                        retrieved_chunks=[],
                        token_usage=TokenUsage(0, 0, 0),
                    ))
                else:
                    results.append(result)
            
            pbar.update(1)
    
    return results


def results_to_csv(results: List[GenerationResult], output_path: str) -> None:
    """
    Serialisiert die Generierungsergebnisse als CSV-Datei.

    Konvertiert alle GenerationResult-Objekte in ein tabellarisches Format
    mit JSON-serialisierten Chunk- und Token-Informationen.

    Args:
        results: Die zu exportierenden GenerationResult-Objekte.
        output_path: Der Zielpfad für die CSV-Datei.
    """
    ensure_parent_directory(output_path)
    
    data = []
    for result in results:
        # retrieved_chunks_json
        chunks_list = [
            {
                'rank': c.rank,
                'chunk_id': c.chunk_id,
                'source_id': c.source_id,
                'chunk_text': c.chunk_text,
                'chunk_length': c.chunk_length,
                'score': c.score,
            }
            for c in result.retrieved_chunks
        ]
        
        # token_json
        token_dict = {
            'input_tokens': result.token_usage.input_tokens,
            'output_tokens': result.token_usage.output_tokens,
            'total_tokens': result.token_usage.total_tokens,
        }
        
        data.append({
            'chunking_strategy': result.chunking_strategy,
            'retrieval_method': result.retrieval_method,
            'question_id': result.question_id,
            'document_id': result.document_id,
            'question': result.question,
            'short_answer': result.short_answer,
            'generated_answer': result.generated_answer,
            'retrieved_chunks_json': json.dumps(chunks_list, ensure_ascii=False),
            'token_json': json.dumps(token_dict),
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding=CSV_ENCODING)
    logger.info(f"Wrote {len(results)} results to {output_path}")


def validate_output(csv_path: str, num_questions: int) -> bool:
    """
    Validiert die erzeugte Output-CSV auf Vollständigkeit und Korrektheit.

    Prüft die Anzahl der Zeilen und die Struktur der JSON-Felder
    für alle erwarteten Strategie-Methode-Kombinationen.

    Args:
        csv_path: Der Pfad zur zu validierenden CSV-Datei.
        num_questions: Die erwartete Anzahl von Fragen.

    Returns:
        True wenn alle Validierungskriterien erfüllt sind.
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV file does not exist: {csv_path}")
        return False
    
    df = pd.read_csv(csv_path, encoding=CSV_ENCODING)
    
    expected_rows = num_questions * len(CHUNKING_STRATEGIES) * len(RETRIEVAL_METHODS)
    if len(df) != expected_rows:
        logger.error(f"Expected {expected_rows} rows, got {len(df)}")
        return False
    
    # Vektorisierte JSON-Parsing und Validierung
    def validate_chunks_json(json_str: str) -> bool:
        """Validiert ein einzelnes JSON-Feld."""
        try:
            chunks = json.loads(json_str)
            if len(chunks) != RETRIEVAL_TOP_K:
                return False
            expected_ranks = list(range(1, RETRIEVAL_TOP_K + 1))
            actual_ranks = [c['rank'] for c in chunks]
            return actual_ranks == expected_ranks[:len(chunks)]
        except (json.JSONDecodeError, KeyError, TypeError):
            return False
    
    # apply ist hier angemessen da JSON-Parsing nicht vektorisierbar ist
    validation_results = df['retrieved_chunks_json'].apply(validate_chunks_json)
    
    if not validation_results.all():
        failed_indices = validation_results[~validation_results].index.tolist()
        logger.error(f"Validation failed for rows: {failed_indices[:5]}...")
        return False
    
    logger.info("Validation passed!")
    return True


def run_step3() -> str:
    """
    Führt die vollständige Retrieval- und Generierungspipeline aus.

    Einstiegspunkt für Schritt 3, der die asynchrone Pipeline startet,
    die Ergebnisse exportiert und die Ausgabe validiert.

    Returns:
        Der Pfad zur erzeugten CSV-Datei.
    """
    results = asyncio.run(run_step3_async())
    
    output_path = str(GENERATION_OUTPUT_CSV_PATH)
    results_to_csv(results, output_path)
    
    # Validiere Output
    df = pd.read_csv(str(OUTPUT_CSV_PATH), encoding=CSV_ENCODING)
    num_questions = len(df)
    
    if validate_output(output_path, num_questions):
        logger.info(f"Step 3 completed successfully. Output: {output_path}")
    else:
        logger.error("Step 3 validation failed!")
    
    return output_path


if __name__ == "__main__":
    run_step3()
