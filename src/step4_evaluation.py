# =============================================================================
# STEP 4: EVALUIERUNG
# =============================================================================
"""
step4_evaluation.py - Evaluierung der RAG-Pipeline

Dieses Modul implementiert den vierten und letzten Schritt der RAG-Pipeline:
die umfassende Evaluierung der Retrieval- und Generierungsqualität.
Es berechnet Standard-IR-Metriken (Recall, MRR, nDCG) sowie
generierungsbezogene Metriken (F1, RAGAS Faithfulness).

Hauptkomponenten:
    - LLM Judge: Automatische Relevanzbewertung von Chunks
    - Retrieval-Metriken: Recall@k, MRR@k, nDCG@k mit Pooling-Ansatz
    - Generation-Metriken: Token-F1, RAGAS Faithfulness und Factual Correctness
    - Ergebnisaggregation: Zusammenfassung nach Strategie, Methode und Kombination

Typische Verwendung:
    from src.step4_evaluation import run_step4
    results_path, summary_path = run_step4()
"""

import os
# Deaktiviere RAGAS-Analytics um blockierende synchrone Aufrufe zu verhindern
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import json
import asyncio
import logging
import math
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Lade Environment-Variablen
load_dotenv()

# Logger für dieses Modul
logger = logging.getLogger("RAG-Pipeline.Step4")

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
    CHUNKING_STRATEGIES,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
    LLM_MODEL,
    EMBEDDING_MODEL,
    MAX_CONCURRENT_REQUESTS,
    MAX_CONCURRENT_RAGAS,
    USE_OPENROUTER,
    OPENROUTER_BASE_URL,
)
from src.utils import (
    ensure_parent_directory,
    async_llm_call,
    get_semaphore,
)


CSV_ENCODING = "utf-8"
GENERATION_OUTPUT_CSV_PATH = DATA_DIR / "output" / "generation_results.csv"
EVALUATION_RESULTS_CSV_PATH = DATA_DIR / "output" / "evaluation_results.csv"
EVALUATION_SUMMARY_JSON_PATH = DATA_DIR / "output" / "evaluation_summary.json"


@dataclass
class RetrievalMetrics:
    """
    Speichert Retrieval-Qualitätsmetriken für eine Konfiguration.

    Attributes:
        relative_recall_at_k: Anteil der relevanten Chunks in den Top-k Ergebnissen.
        mrr_at_k: Mean Reciprocal Rank des ersten relevanten Chunks.
        ndcg_at_k: Normalized Discounted Cumulative Gain.
    """
    relative_recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


@dataclass
class GenerationMetrics:
    """
    Speichert Generierungsqualitätsmetriken für eine Antwort.

    Attributes:
        f1_score: Token-basierter F1-Score gegen die Referenzantwort.
        ragas_faithfulness: RAGAS Faithfulness Score.
        ragas_factual_correctness: RAGAS Factual Correctness Score.
    """
    f1_score: float
    ragas_faithfulness: float
    ragas_factual_correctness: float


async def judge_chunk_relevance(
    question: str,
    short_answer: str,
    chunk_text: str,
) -> bool:
    """
    Bewertet die Relevanz eines Chunks für eine Frage mittels LLM.

    Verwendet einen LLM als Judge um zu bestimmen, ob der Chunk-Text
    die notwendigen Informationen zur Beantwortung der Frage enthält.

    Args:
        question: Die zu beantwortende Frage.
        short_answer: Die erwartete Kurzantwort (Ground Truth).
        chunk_text: Der zu bewertende Chunk-Text.

    Returns:
        True wenn der Chunk als relevant bewertet wird, sonst False.
    """
    user_prompt = JUDGE_USER_PROMPT.format(
        question=question,
        short_answer=short_answer,
        chunk_text=chunk_text,
    )
    
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        response = await async_llm_call(messages, context="evaluation")
        answer = response['content'].strip().upper()
        return answer == 'YES'
    except Exception as e:
        logger.warning(f"Judge call failed: {e}")
        return False


async def judge_pool(
    question: str,
    short_answer: str,
    pool: Dict[str, str],
) -> Dict[str, bool]:
    """
    Bewertet alle Chunks eines Pools parallel auf Relevanz.

    Führt für jeden Chunk im Pool eine LLM-basierte Relevanzbewertung
    durch und sammelt die Ergebnisse in einem Mapping.

    Args:
        question: Die zu beantwortende Frage.
        short_answer: Die erwartete Kurzantwort.
        pool: Ein Dictionary von Chunk-ID zu Chunk-Text.

    Returns:
        Ein Dictionary von Chunk-ID zu Relevanz-Boolean.
    """
    tasks = []
    chunk_ids = list(pool.keys())
    
    for chunk_id in chunk_ids:
        chunk_text = pool[chunk_id]
        task = judge_chunk_relevance(question, short_answer, chunk_text)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    relevance_map = {}
    for chunk_id, result in zip(chunk_ids, results):
        if isinstance(result, Exception):
            logger.warning(f"Judge failed for chunk {chunk_id}: {result}")
            relevance_map[chunk_id] = False
        else:
            relevance_map[chunk_id] = result
    
    return relevance_map


def compute_relative_recall_at_k(
    retrieved_chunk_ids: List[str],
    relevance_map: Dict[str, bool],
    total_relevant_in_pool: int,
) -> float:
    """
    Berechnet den Relative Recall@k für eine Retrieval-Ergebnisliste.

    Bestimmt den Anteil der relevanten Chunks aus dem Pool,
    die in den abgerufenen Top-k Ergebnissen enthalten sind.

    Args:
        retrieved_chunk_ids: Die Chunk-IDs in Ranking-Reihenfolge.
        relevance_map: Mapping von Chunk-ID zu Relevanz.
        total_relevant_in_pool: Gesamtzahl relevanter Chunks im Pool.

    Returns:
        Der Relative Recall@k-Wert zwischen 0.0 und 1.0.
    """
    if total_relevant_in_pool == 0:
        return 0.0
    
    relevant_in_top_k = sum(
        1 for chunk_id in retrieved_chunk_ids
        if relevance_map.get(chunk_id, False)
    )
    
    return relevant_in_top_k / total_relevant_in_pool


def compute_mrr_at_k(
    retrieved_chunk_ids: List[str],
    relevance_map: Dict[str, bool],
) -> float:
    """
    Berechnet den Mean Reciprocal Rank für eine Ergebnisliste.

    Ermittelt den Kehrwert des Rangs des ersten relevanten Chunks
    in der Ergebnisliste.

    Args:
        retrieved_chunk_ids: Die Chunk-IDs in Ranking-Reihenfolge.
        relevance_map: Mapping von Chunk-ID zu Relevanz.

    Returns:
        Der MRR@k-Wert (1/rank oder 0.0 wenn kein relevanter Chunk).
    """
    for rank, chunk_id in enumerate(retrieved_chunk_ids, 1):
        if relevance_map.get(chunk_id, False):
            return 1.0 / rank
    
    return 0.0


def compute_ndcg_at_k(
    retrieved_chunk_ids: List[str],
    relevance_map: Dict[str, bool],
    total_relevant_in_pool: int,
) -> float:
    """
    Berechnet den Normalized Discounted Cumulative Gain.

    Bewertet die Ranking-Qualität unter Berücksichtigung der
    Position relevanter Dokumente mit logarithmischer Diskontierung.

    Args:
        retrieved_chunk_ids: Die Chunk-IDs in Ranking-Reihenfolge.
        relevance_map: Mapping von Chunk-ID zu Relevanz.
        total_relevant_in_pool: Gesamtzahl relevanter Chunks im Pool.

    Returns:
        Der nDCG@k-Wert zwischen 0.0 und 1.0.
    """
    k = len(retrieved_chunk_ids)
    
    if total_relevant_in_pool == 0:
        return 0.0
    
    # DCG@k berechnen
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved_chunk_ids, 1):
        rel = 1.0 if relevance_map.get(chunk_id, False) else 0.0
        dcg += rel / math.log2(i + 1)
    
    # IDCG@k berechnen (ideale Sortierung)
    ideal_relevant_count = min(k, total_relevant_in_pool)
    idcg = 0.0
    for i in range(1, ideal_relevant_count + 1):
        idcg += 1.0 / math.log2(i + 1)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_f1_score(generated_answer: str, short_answer: str) -> float:
    """
    Berechnet den Token-F1-Score zwischen Antwort und Referenz.

    Bestimmt Precision und Recall basierend auf dem Token-Overlap
    und kombiniert diese zum harmonischen Mittel.

    Args:
        generated_answer: Die generierte Antwort.
        short_answer: Die Referenz-Kurzantwort.

    Returns:
        Der F1-Score zwischen 0.0 und 1.0.
    """
    # Früher Rücksprung für leere Strings (häufiger Edge Case)
    if not generated_answer or not short_answer:
        return 0.0
    
    # Tokenisiere (lowercase, whitespace split)
    gen_tokens = set(generated_answer.lower().split())
    ref_tokens = set(short_answer.lower().split())
    
    # Früher Rücksprung für leere Token-Sets
    if not gen_tokens or not ref_tokens:
        return 0.0
    
    # Schnittmenge direkt berechnen (O(min(m,n)))
    overlap_count = len(gen_tokens & ref_tokens)
    
    if overlap_count == 0:
        return 0.0
    
    precision = overlap_count / len(gen_tokens)
    recall = overlap_count / len(ref_tokens)
    
    return 2.0 * (precision * recall) / (precision + recall)


async def compute_ragas_metrics(
    results_df: pd.DataFrame,
) -> Dict[int, Dict[str, float]]:
    """
    Berechnet RAGAS-Metriken für alle Generierungsergebnisse.

    Führt die RAGAS-Evaluierung asynchron aus um den Event Loop
    nicht zu blockieren und gibt Faithfulness sowie Factual Correctness zurück.

    Args:
        results_df: DataFrame mit den Generierungsergebnissen.

    Returns:
        Ein Dictionary von Zeilen-Index zu Metrik-Werten.
    """
    num_rows = len(results_df)
    num_metrics = 2  # faithfulness + factual_correctness
    logger.info(f"Computing RAGAS metrics for {num_rows} samples ({num_rows * num_metrics} metric evaluations)...")
    
    ragas_results = {}
    
    # Definiere die synchrone Ragas-Ausführung als innere Funktion
    def run_ragas_sync():
        try:
            from ragas import evaluate, RunConfig
            from ragas.metrics import faithfulness, FactualCorrectness
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from datasets import Dataset
            
            # Instanziiere FactualCorrectness-Metrik (Klasse, keine vorgefertigte Instanz)
            factual_correctness = FactualCorrectness()
            
            # Bereite Daten für RAGAS vor
            # RAGAS 0.4.1 erwartet: 'user_input', 'response', 'reference', 'retrieved_contexts'
            # für FactualCorrectness: benötigt 'response' und 'reference'
            # für Faithfulness: benötigt 'user_input', 'response', 'retrieved_contexts'
            data = {
                'user_input': results_df['question'].tolist(),
                'response': results_df['generated_answer'].tolist(),
                'reference': results_df['short_answer'].tolist(),
                'retrieved_contexts': results_df['retrieved_chunks_json'].apply(
                    lambda x: [c['chunk_text'] for c in json.loads(x)]
                ).tolist()
            }
            
            # Erstelle RAGAS-Dataset
            dataset = Dataset.from_dict(data)
            
            # Initialisiere LLM und Embeddings für RAGAS
            # Verwende OpenRouter falls aktiviert (gleiche API, andere base_url)
            if USE_OPENROUTER:
                llm = ChatOpenAI(
                    model=LLM_MODEL,
                    temperature=0,
                    base_url=OPENROUTER_BASE_URL,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
                # OpenRouter unterstützt Embeddings über openai/text-embedding-3-small
                embeddings = OpenAIEmbeddings(
                    model="openai/text-embedding-3-small",
                    base_url=OPENROUTER_BASE_URL,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
                logger.info("RAGAS using OpenRouter for LLM and Embedding calls")
            else:
                llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
                embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            
            # Wrapper für RAGAS erstellen
            wrapped_llm = LangchainLLMWrapper(llm)
            wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

            # --- PROMPT-ANPASSUNG (System/User-Trennung) ---
            # Erzwinge strikte System/User-Trennung für Faithfulness (Statement Generator)
            faithfulness.statement_generator_prompt.instruction = (
                "Given a question and an answer, analyze the complexity of each sentence in the answer. "
                "Break down each sentence into one or more fully understandable statements. "
                "Ensure that no pronouns are used in any statement. "
                "Format the outputs in JSON. "
                "Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:\n"
                "{schema}\n"
                "Do not use single quotes in your response but double quotes, properly escaped with a backslash."
            )

            # Evaluierung durchführen
            # max_workers=MAX_CONCURRENT_RAGAS für konservative RAGAS-Parallelisierung
            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, factual_correctness],
                llm=wrapped_llm,
                embeddings=wrapped_embeddings,
                run_config=RunConfig(
                    timeout=180, 
                    max_retries=10, 
                    max_wait=300, 
                    max_workers=MAX_CONCURRENT_RAGAS
                ),
                raise_exceptions=False,
            )
            
            # Ergebnisse extrahieren
            result_df = result.to_pandas()
            
            # Finde die factual_correctness-Spalte (kann Mode-Suffix wie 'factual_correctness(mode=f1)' enthalten)
            fc_col = None
            for col in result_df.columns:
                if col.startswith('factual_correctness'):
                    fc_col = col
                    break
            
            local_results = {}
            for idx, row in result_df.iterrows():
                local_results[idx] = {
                    'faithfulness': row.get('faithfulness', 0.0),
                    'factual_correctness': row.get(fc_col, 0.0) if fc_col else 0.0,
                }
            
            logger.info("RAGAS metrics computed successfully")
            return local_results
            
        except ImportError as e:
            logger.warning(f"RAGAS not available: {e}. Using fallback values.")
            return {}
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}. Using fallback values.")
            return {}

    # Führe die synchrone Funktion im Executor aus
    try:
        loop = asyncio.get_running_loop()
        ragas_results = await loop.run_in_executor(None, run_ragas_sync)
    except Exception as e:
        logger.error(f"Async execution of RAGAS failed: {e}")
        ragas_results = {}

    # Fallback-Werte auffüllen falls leer oder Fehler
    if not ragas_results:
        for idx in range(len(results_df)):
            ragas_results[idx] = {
                'faithfulness': 0.0,
                'factual_correctness': 0.0,
            }
            
    return ragas_results


async def evaluate_retrieval_for_question(
    question: str,
    short_answer: str,
    question_id: str,
    strategy: str,
    results_for_strategy: Dict[str, List[Dict]],
) -> Dict[str, RetrievalMetrics]:
    """
    Evaluiert Retrieval-Qualität für eine Frage mittels Pooling-Ansatz.

    Vereinigt die Top-k Ergebnisse aller Retrieval-Methoden zu einem Pool,
    bewertet diesen einmalig mit dem LLM-Judge und berechnet dann die
    Metriken für jede Methode basierend auf dem gemeinsamen Relevanz-Mapping.

    Args:
        question: Die Fragestellung.
        short_answer: Die erwartete Kurzantwort.
        question_id: Die eindeutige Frage-ID.
        strategy: Die verwendete Chunking-Strategie.
        results_for_strategy: Mapping von Retrieval-Methode zu Chunk-Listen.

    Returns:
        Ein Dictionary von Methoden-Namen zu RetrievalMetrics-Objekten.
    """
    # 1. Pool: Vereinigung aller Chunks mit Dict Comprehension (O(n))
    # Innere Dict Comprehension wird pro Methode ausgeführt, äußere merged
    pool: Dict[str, str] = {
        chunk['chunk_id']: chunk['chunk_text']
        for chunks in results_for_strategy.values()
        for chunk in chunks
    }
    
    logger.debug(f"Pool size for {question_id}/{strategy}: {len(pool)}")
    
    # 2. Judge den Pool einmalig
    relevance_map = await judge_pool(question, short_answer, pool)
    
    # 3. total_relevant aus dem gesamten Pool (Generator statt List)
    total_relevant = sum(1 for v in relevance_map.values() if v)
    
    logger.debug(f"Total relevant in pool: {total_relevant}/{len(pool)}")
    
    # 4. Berechne Metriken für jede Methode mit Dict Comprehension
    return {
        method: RetrievalMetrics(
            relative_recall_at_k=compute_relative_recall_at_k(
                [c['chunk_id'] for c in chunks], relevance_map, total_relevant
            ),
            mrr_at_k=compute_mrr_at_k(
                [c['chunk_id'] for c in chunks], relevance_map
            ),
            ndcg_at_k=compute_ndcg_at_k(
                [c['chunk_id'] for c in chunks], relevance_map, total_relevant
            ),
        )
        for method, chunks in results_for_strategy.items()
    }


async def run_step4_async() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Führt die asynchrone Evaluierungspipeline aus.

    Orchestriert die Retrieval- und Generierungsevaluierung für alle
    Fragen und Konfigurationen und erstellt eine Ergebniszusammenfassung.

    Returns:
        Ein Tuple aus (erweitertem DataFrame mit Metriken, Summary-Dictionary).
    """
    from tqdm import tqdm
    from tqdm.asyncio import tqdm_asyncio
    
    logger.info("STEP 4: Evaluation")
    
    # Lade Step-3 CSV
    csv_path = str(GENERATION_OUTPUT_CSV_PATH)
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, encoding=CSV_ENCODING)
    
    # Organisiere Daten für Pooling
    # Struktur: question_id -> strategy -> method -> [chunks]
    organized_data: Dict[str, Dict[str, Dict[str, List[Dict]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    
    question_data: Dict[str, Dict[str, str]] = {}  # question_id -> {question, short_answer}
    
    for idx, row in df.iterrows():
        qid = row['question_id']
        strategy = row['chunking_strategy']
        method = row['retrieval_method']
        
        chunks = json.loads(row['retrieved_chunks_json'])
        organized_data[qid][strategy][method] = chunks
        
        if qid not in question_data:
            question_data[qid] = {
                'question': row['question'],
                'short_answer': row['short_answer'],
            }
    
    num_questions = len(organized_data)
    logger.info(f"Loaded {len(df)} rows ({num_questions} questions × {len(CHUNKING_STRATEGIES)} strategies × {len(RETRIEVAL_METHODS)} methods)")
    
    # 1. Retrieval-Evaluierung (mit Pooling pro Strategie)
    logger.info("Evaluating retrieval quality...")
    
    # Struktur: (question_id, strategy, method) -> RetrievalMetrics
    retrieval_metrics: Dict[Tuple[str, str, str], RetrievalMetrics] = {}
    
    # Hilfsfunktion für parallele Ausführung
    async def process_question_evaluation(qid_val: str):
        q_text = question_data[qid_val]['question']
        s_answer = question_data[qid_val]['short_answer']
        
        results = []
        for strat in organized_data[qid_val]:
            res_for_strat = organized_data[qid_val][strat]
            
            metrics_map = await evaluate_retrieval_for_question(
                question=q_text,
                short_answer=s_answer,
                question_id=qid_val,
                strategy=strat,
                results_for_strategy=res_for_strat,
            )
            results.append((qid_val, strat, metrics_map))
        return results

    # Wrapper für Exception-Handling
    async def safe_process_evaluation(qid: str) -> Tuple[str, Any]:
        try:
            result = await process_question_evaluation(qid)
            return (qid, result)
        except Exception as e:
            return (qid, e)

    # Erstelle Tasks für alle Fragen
    tasks = [safe_process_evaluation(qid) for qid in organized_data]
    
    # Führe alle Fragen parallel mit Fortschrittsbalken aus
    indexed_results = await tqdm_asyncio.gather(
        *tasks,
        desc="Retrieval Eval",
        total=len(tasks),
    )
    
    for qid, res_batch in indexed_results:
        if isinstance(res_batch, Exception):
            logger.warning(f"Evaluation task failed for {qid}: {res_batch}")
            continue
            
        for qid_val, strat, metrics_map in res_batch:
            for meth, metrics in metrics_map.items():
                retrieval_metrics[(qid_val, strat, meth)] = metrics
    
    # 2. Generierungsmetriken
    logger.info("Computing generation metrics (F1 + RAGAS)...")
    
    # F1-Scores (vektorisiert)
    f1_scores = df.apply(
        lambda row: compute_f1_score(row['generated_answer'], row['short_answer']), 
        axis=1
    )
    
    # RAGAS-Metriken (asynchron)
    ragas_metrics = await compute_ragas_metrics(df)
    
    # 3. Erweitere DataFrame
    logger.info("Extending DataFrame with metrics...")
    
    # Erstelle Lookup-DataFrame aus retrieval_metrics für effizienten Merge
    metrics_data = [
        {
            'question_id': key[0],
            'chunking_strategy': key[1],
            'retrieval_method': key[2],
            f'RelativeRecall@{RETRIEVAL_TOP_K}': metrics.relative_recall_at_k,
            f'MRR@{RETRIEVAL_TOP_K}': metrics.mrr_at_k,
            f'nDCG@{RETRIEVAL_TOP_K}': metrics.ndcg_at_k,
        }
        for key, metrics in retrieval_metrics.items()
    ]
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        # Merge auf Schlüsselspalten (effizient via Index-basiertem Join)
        df = df.merge(
            metrics_df,
            on=['question_id', 'chunking_strategy', 'retrieval_method'],
            how='left'
        )
        # Fülle NaN-Werte mit 0.0
        for col in [f'RelativeRecall@{RETRIEVAL_TOP_K}', f'MRR@{RETRIEVAL_TOP_K}', f'nDCG@{RETRIEVAL_TOP_K}']:
            df[col] = df[col].fillna(0.0)
    else:
        # Fallback: Leere Spalten
        df[f'RelativeRecall@{RETRIEVAL_TOP_K}'] = 0.0
        df[f'MRR@{RETRIEVAL_TOP_K}'] = 0.0
        df[f'nDCG@{RETRIEVAL_TOP_K}'] = 0.0
    
    df['F1_Score'] = f1_scores.values  # .values um Index-Alignment sicherzustellen
    
    # RAGAS-Metriken mappen (Index-Matching, da ragas_metrics nach Index geschlüsselt ist)
    df['Ragas_Faithfulness'] = df.index.map(lambda idx: ragas_metrics.get(idx, {}).get('faithfulness', 0.0))
    df['Ragas_FactualCorrectness'] = df.index.map(lambda idx: ragas_metrics.get(idx, {}).get('factual_correctness', 0.0))
    
    # 4. Zusammenfassung erstellen mit Helper-Funktion (DRY)
    logger.info("Creating evaluation summary...")
    
    def compute_mean_metrics(subset_df: pd.DataFrame) -> Dict[str, float]:
        """Helper: Berechnet Mittelwerte aller Metriken für ein DataFrame-Subset."""
        return {
            f'mean_RelativeRecall@{RETRIEVAL_TOP_K}': float(subset_df[f'RelativeRecall@{RETRIEVAL_TOP_K}'].mean()),
            f'mean_MRR@{RETRIEVAL_TOP_K}': float(subset_df[f'MRR@{RETRIEVAL_TOP_K}'].mean()),
            f'mean_nDCG@{RETRIEVAL_TOP_K}': float(subset_df[f'nDCG@{RETRIEVAL_TOP_K}'].mean()),
            'mean_F1_Score': float(subset_df['F1_Score'].mean()),
            'mean_Ragas_Faithfulness': float(subset_df['Ragas_Faithfulness'].mean()),
            'mean_Ragas_FactualCorrectness': float(subset_df['Ragas_FactualCorrectness'].mean()),
        }
    
    summary = {
        'by_strategy': {
            strategy: compute_mean_metrics(df[df['chunking_strategy'] == strategy])
            for strategy in CHUNKING_STRATEGIES
        },
        'by_method': {
            method: compute_mean_metrics(df[df['retrieval_method'] == method])
            for method in RETRIEVAL_METHODS
        },
        'by_combination': {
            f"{strategy}_{method}": compute_mean_metrics(
                df[(df['chunking_strategy'] == strategy) & (df['retrieval_method'] == method)]
            )
            for strategy in CHUNKING_STRATEGIES
            for method in RETRIEVAL_METHODS
        },
    }
    
    return df, summary


def run_step4() -> Tuple[str, str]:
    """
    Führt die vollständige Evaluierungspipeline aus.

    Einstiegspunkt für Schritt 4, der die asynchrone Pipeline startet,
    die Ergebnisse als CSV und die Zusammenfassung als JSON exportiert.

    Returns:
        Ein Tuple aus (Pfad zur Ergebnis-CSV, Pfad zur Summary-JSON).
    """
    df, summary = asyncio.run(run_step4_async())
    
    # Schreibe erweiterte CSV
    results_path = str(EVALUATION_RESULTS_CSV_PATH)
    ensure_parent_directory(results_path)
    df.to_csv(results_path, index=False, encoding=CSV_ENCODING)
    logger.info(f"Results saved to {results_path}")
    
    # Schreibe Summary JSON
    summary_path = str(EVALUATION_SUMMARY_JSON_PATH)
    ensure_parent_directory(summary_path)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")
    
    # Kompakte Zusammenfassung
    logger.info("EVALUATION RESULTS")
    
    # Top-Level-Metriken pro Strategie (kompakt)
    logger.info("By Strategy (mean values):")
    for strategy, metrics in summary['by_strategy'].items():
        recall = metrics.get('mean_RelativeRecall@10', 0)
        f1 = metrics.get('mean_F1_Score', 0)
        faith = metrics.get('mean_Ragas_Faithfulness', 0)
        logger.info(f"  {strategy:12s} | Rel.Recall@10: {recall:.3f} | F1: {f1:.3f} | Faithfulness: {faith:.3f}")
    
    logger.info("By Method (mean values):")
    for method, metrics in summary['by_method'].items():
        recall = metrics.get('mean_RelativeRecall@10', 0)
        f1 = metrics.get('mean_F1_Score', 0)
        faith = metrics.get('mean_Ragas_Faithfulness', 0)
        logger.info(f"  {method:12s} | Rel.Recall@10: {recall:.3f} | F1: {f1:.3f} | Faithfulness: {faith:.3f}")
    
    return results_path, summary_path


if __name__ == "__main__":
    run_step4()
