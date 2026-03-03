# =============================================================================
# STEP 2: CHUNKING & INDEXIERUNG
# =============================================================================
"""
Chunking und Indexierung für die RAG-Pipeline.

Dieses Modul implementiert den zweiten Schritt der RAG-Pipeline und ist
verantwortlich für die Segmentierung von Dokumenten in Chunks sowie deren
Indexierung für effizientes Retrieval. Es stellt drei verschiedene
Chunking-Strategien bereit, die unterschiedliche Ansätze zur
Dokumentsegmentierung verfolgen.

Die implementierten Strategien sind:
    - Heuristisches Chunking: Sliding-Window-Ansatz mit Token-Zählung
    - Semantisches Chunking: Ähnlichkeitsbasierte Segmentierung
    - Agentisches Chunking: LLM-gestützte Proposition-Extraktion

Für jede Strategie werden zwei Indextypen erstellt:
    - Chroma Vector Store für Dense Retrieval
    - BM25 Inverted Index für Sparse Retrieval
"""

import os
import re
import json
import pickle
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

import pandas as pd
import numpy as np
import nltk
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("RAG-Pipeline.Step2")

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for lib in ["httpx", "openai", "chromadb"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except Exception as e:
        logger.warning(f"NLTK data download failed: {e}")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR,
    HEURISTIC_CHUNK_TOKENS,
    HEURISTIC_OVERLAP_TOKENS,
    SEMANTIC_SENTENCE_MIN_CHARS,
    SEMANTIC_WINDOW_SENTENCE_COUNT,
    SEMANTIC_SIMILARITY_THRESHOLD,
    CHUNKING_STRATEGIES,
    CHROMA_PERSIST_ROOT,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TEMPERATURE,
    AGENTIC_PROPOSITION_SYSTEM_PROMPT,
    AGENTIC_PROPOSITION_USER_PROMPT,
    AGENTIC_GROUPING_SYSTEM_PROMPT,
    AGENTIC_GROUPING_USER_PROMPT,
    get_chroma_path,
    get_bm25_path,
    get_chunk_ids_path,
    get_stats_path,
)
from src.utils import (
    ensure_parent_directory,
    ensure_directory,
    encode_text,
    decode_tokens,
    count_tokens,
    generate_chunk_id,
    async_llm_call,
    async_get_embeddings,
    sync_get_embeddings,
    safe_json_parse,
    tokenize_for_bm25,
    get_proposition_format_instructions,
    get_grouping_format_instructions,
    get_semaphore,
)


OUTPUT_CSV_PATH = DATA_DIR / "input" / "nq_validation_cleaned.csv"
CSV_ENCODING = "utf-8"


_RE_MARKDOWN_HEADER = re.compile(r'(^#+\s+.*$)', re.MULTILINE)
_RE_HEADER_CHECK = re.compile(r'^#+\s+')
_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
_RE_AGENTIC_HEADER = re.compile(r'(^#+ .*$)', re.MULTILINE)


@dataclass
class Chunk:
    """
    Repräsentiert einen Textchunk aus einem Dokument.

    Attributes:
        chunk_id: Eindeutige Kennung des Chunks.
        strategy: Die verwendete Chunking-Strategie.
        source_id: Die ID des Quelldokuments.
        source_document_id: Die Dokument-ID aus der Datenquelle.
        chunk_index: Fortlaufender Index des Chunks im Dokument.
        text: Der Textinhalt des Chunks.
        metadata: Zusätzliche Metadaten zum Chunk.
    """
    chunk_id: str
    strategy: str
    source_id: str
    source_document_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ergänzt die Metadaten um Basis-Informationen."""
        self.metadata.update({
            'strategy': self.strategy,
            'source_id': self.source_id,
            'source_document_id': self.source_document_id,
            'chunk_index': self.chunk_index,
        })


@dataclass
class IndexingStats:
    """
    Statistiken über die Indexierung einer Chunking-Strategie.

    Attributes:
        strategy: Die Chunking-Strategie.
        num_documents: Anzahl der verarbeiteten Dokumente.
        num_chunks: Gesamtzahl der erzeugten Chunks.
        chunk_len_chars_min: Minimale Chunk-Länge in Zeichen.
        chunk_len_chars_mean: Durchschnittliche Chunk-Länge in Zeichen.
        chunk_len_chars_max: Maximale Chunk-Länge in Zeichen.
        chunk_len_tokens_min: Minimale Chunk-Länge in Tokens.
        chunk_len_tokens_mean: Durchschnittliche Chunk-Länge in Tokens.
        chunk_len_tokens_max: Maximale Chunk-Länge in Tokens.
        embedding_model: Das verwendete Embedding-Modell.
        created_at: Zeitstempel der Erstellung.
    """
    strategy: str
    num_documents: int
    num_chunks: int
    chunk_len_chars_min: int
    chunk_len_chars_mean: float
    chunk_len_chars_max: int
    chunk_len_tokens_min: int
    chunk_len_tokens_mean: float
    chunk_len_tokens_max: int
    embedding_model: str
    created_at: str


def heuristic_chunk(
    document_text: str,
    source_id: str,
    source_document_id: str,
    chunk_size: int = HEURISTIC_CHUNK_TOKENS,
    overlap: int = HEURISTIC_OVERLAP_TOKENS,
) -> List[Chunk]:
    """
    Segmentiert ein Dokument mittels heuristischem Chunking.

    Verwendet einen Sliding-Window-Ansatz mit Satzrespektierung. Das
    Dokument wird in Chunks einer Zielgröße unterteilt, wobei Satzgrenzen
    nach Möglichkeit respektiert werden. Überlappende Bereiche zwischen
    aufeinanderfolgenden Chunks gewährleisten Kontextkontinuität.

    Args:
        document_text: Der zu segmentierende Dokumenttext.
        source_id: Die ID des Quelldokuments.
        source_document_id: Die Dokument-ID aus der Datenquelle.
        chunk_size: Zielgröße eines Chunks in Tokens.
        overlap: Überlappung zwischen Chunks in Tokens.

    Returns:
        Eine Liste von Chunk-Objekten.
    """
    strategy = "heuristic"
    chunks = []
    
    if not document_text:
        return chunks

    try:
        raw_sentences = nltk.sent_tokenize(document_text, language='english')
    except Exception as e:
        logger.warning(f"NLTK tokenization failed: {e}. Fallback to simple split.")
        raw_sentences = document_text.split('. ')

    sentences = [s.replace('\n', ' ').strip() for s in raw_sentences if s.strip()]
    
    units = []
    for sentence in sentences:
        cnt = count_tokens(sentence)
        if cnt <= chunk_size:
            units.append((sentence, cnt))
        else:
            words = sentence.split(' ')
            for word in words:
                if not word: continue
                units.append((word, count_tokens(word)))
    
    current_chunk_units = []
    current_chunk_tokens = 0
    chunk_index = 0
    
    i = 0
    while i < len(units):
        unit_text, unit_tokens = units[i]
        
        current_chunk_units.append((unit_text, unit_tokens))
        current_chunk_tokens += unit_tokens
        i += 1
        
        if current_chunk_tokens >= chunk_size and i < len(units):
            chunk_text = " ".join(u[0] for u in current_chunk_units)
            
            chunk_id = generate_chunk_id(strategy, source_id, chunk_index)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                strategy=strategy,
                source_id=source_id,
                source_document_id=source_document_id,
                chunk_index=chunk_index,
                text=chunk_text,
                metadata={'token_count': current_chunk_tokens}
            ))
            chunk_index += 1
            
            overlap_units = []
            overlap_tokens_current = 0
            
            for u_text, u_tokens in reversed(current_chunk_units):
                if overlap_tokens_current + u_tokens <= overlap:
                    overlap_units.append((u_text, u_tokens))
                    overlap_tokens_current += u_tokens
                else:
                    remaining = overlap - overlap_tokens_current
                    if remaining > 0:
                        words = u_text.split(' ')
                        suffix_words = words[-remaining:]
                        suffix_text = " ".join(suffix_words)
                        suffix_tokens = count_tokens(suffix_text)
                        
                        overlap_units.append((suffix_text, suffix_tokens))
                        overlap_tokens_current += suffix_tokens
                    break
            
            overlap_units.reverse()
            current_chunk_units = overlap_units
            current_chunk_tokens = overlap_tokens_current
            
    # Letzten Chunk hinzufügen
    if current_chunk_units:
        chunk_text = " ".join(u[0] for u in current_chunk_units)
        chunk_id = generate_chunk_id(strategy, source_id, chunk_index)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            strategy=strategy,
            source_id=source_id,
            source_document_id=source_document_id,
            chunk_index=chunk_index,
            text=chunk_text,
            metadata={'token_count': current_chunk_tokens}
        ))
    
    return chunks


def split_by_headers(text: str) -> List[Dict[str, str]]:
    """
    Splittet einen Text anhand von Markdown-Überschriften.

    Führt einen Macro-Split durch, bei dem der Text an Überschriften
    aufgeteilt wird. Jeder Abschnitt enthält die Überschrift und den
    zugehörigen Fließtext.

    Args:
        text: Der zu splittende Text.

    Returns:
        Eine Liste von Dictionaries mit 'header' und 'body' Schlüsseln.
    """
    parts = _RE_MARKDOWN_HEADER.split(text)
    
    sections = []
    current_header = None
    current_body = []
    
    for part in parts:
        if _RE_HEADER_CHECK.match(part):
            # Neuer Header gefunden
            if current_header is not None or current_body:
                sections.append({
                    'header': current_header or '',
                    'body': '\n'.join(current_body).strip()
                })
            current_header = part.strip()
            current_body = []
        else:
            if part.strip():
                current_body.append(part)
    
    # Letzte Section hinzufügen
    if current_header is not None or current_body:
        sections.append({
            'header': current_header or '',
            'body': '\n'.join(current_body).strip()
        })
    
    return sections


def split_into_sentences(text: str) -> List[str]:
    """
    Splittet einen Text in einzelne Sätze.

    Verwendet Satzendzeichen (., !, ?) als Trennkriterium.

    Args:
        text: Der zu splittende Text.

    Returns:
        Eine Liste von Satzstrings.
    """
    # Split bei Satzendzeichen, behalte den Delimiter
    sentences = _RE_SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]


def merge_short_sentences(
    sentences: List[str],
    min_chars: int = SEMANTIC_SENTENCE_MIN_CHARS
) -> List[str]:
    """
    Merged kurze Sätze mit dem vorherigen.
    
    Args:
        sentences: Liste von Sätzen.
        min_chars: Minimale Zeichenanzahl.
        
    Returns:
        Liste von gemergten Sätzen.
    """
    if not sentences:
        return []
    
    merged = [sentences[0]]
    
    for sentence in sentences[1:]:
        if len(sentence) < min_chars and merged:
            merged[-1] = merged[-1] + ' ' + sentence
        else:
            merged.append(sentence)
    
    return merged


def create_windows(
    sentences: List[str],
    window_size: int = SEMANTIC_WINDOW_SENTENCE_COUNT
) -> List[str]:
    """
    Bildet Fenster aus Sätzen.
    
    Args:
        sentences: Liste von Sätzen.
        window_size: Anzahl Sätze pro Fenster.
        
    Returns:
        Liste von Fenster-Texten.
    """
    windows = []
    
    for i in range(0, len(sentences), window_size):
        window_sentences = sentences[i:i + window_size]
        windows.append(' '.join(window_sentences))
    
    return windows


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Berechnet die Kosinus-Ähnlichkeit zwischen zwei Vektoren.

    Args:
        vec1: Der erste Embedding-Vektor.
        vec2: Der zweite Embedding-Vektor.

    Returns:
        Ein Ähnlichkeitswert zwischen -1 und 1.
    """
    # Konvertiere einmal zu numpy (O(n))
    arr1 = np.asarray(vec1)
    arr2 = np.asarray(vec2)
    
    # Berechne Normen inline (vermeidet temporäre Variablen)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    
    # Früher Rücksprung für Nullvektoren (häufiger Edge Case)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    
    # Skalarprodukt und Division in einem Schritt
    return float(np.dot(arr1, arr2) / (norm1 * norm2))


def semantic_chunk(
    document_text: str,
    source_id: str,
    source_document_id: str,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> List[Chunk]:
    """
    Segmentiert ein Dokument mittels semantischem Chunking.

    Dieser Ansatz identifiziert thematische Brüche im Text durch
    Analyse der semantischen Ähnlichkeit zwischen Textfenstern.
    Bei niedrigen Ähnlichkeitswerten wird ein neuer Chunk begonnen.

    Args:
        document_text: Der zu segmentierende Dokumenttext.
        source_id: Die ID des Quelldokuments.
        source_document_id: Die Dokument-ID aus der Datenquelle.
        similarity_threshold: Schwellenwert für die Ähnlichkeit.

    Returns:
        Eine Liste von Chunk-Objekten.
    """
    strategy = "semantic"
    chunks = []
    chunk_index = 0
    
    # 1. Macro-Split nach Headers
    sections = split_by_headers(document_text)
    
    for section in sections:
        body = section['body']
        header = section['header']
        
        if not body.strip():
            continue
        
        # 2. Satzsegmentierung
        sentences = split_into_sentences(body)
        
        if not sentences:
            continue
        
        # 3. Kurze Sätze zusammenführen
        sentences = merge_short_sentences(sentences)
        
        if len(sentences) <= 1:
            # Zu wenig Sätze für ähnlichkeitsbasiertes Splitting
            chunk_id = generate_chunk_id(strategy, source_id, chunk_index)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                strategy=strategy,
                source_id=source_id,
                source_document_id=source_document_id,
                chunk_index=chunk_index,
                text=body,
                metadata={'section_header': header}
            ))
            chunk_index += 1
            continue
        
        # 4. Fensterbildung
        windows = create_windows(sentences)
        
        if len(windows) <= 1:
            chunk_id = generate_chunk_id(strategy, source_id, chunk_index)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                strategy=strategy,
                source_id=source_id,
                source_document_id=source_document_id,
                chunk_index=chunk_index,
                text=body,
                metadata={'section_header': header}
            ))
            chunk_index += 1
            continue
        
        # 5. Embeddings berechnen
        try:
            embeddings = sync_get_embeddings(windows)
        except Exception as e:
            logger.warning(f"Embedding failed for section: {e}")
            chunk_id = generate_chunk_id(strategy, source_id, chunk_index)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                strategy=strategy,
                source_id=source_id,
                source_document_id=source_document_id,
                chunk_index=chunk_index,
                text=body,
                metadata={'section_header': header}
            ))
            chunk_index += 1
            continue
        
        # 6. Ähnlichkeit berechnen und Split-Punkte finden
        split_indices = []  # Indizes in der Satz-Liste, wo gesplittet wird
        
        for i in range(len(windows) - 1):
            sim = compute_cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < similarity_threshold:
                # Split nach dem aktuellen Fenster
                # Fenster i enthält Sätze i*2 bis (i+1)*2-1
                split_after_sentence = (i + 1) * SEMANTIC_WINDOW_SENTENCE_COUNT
                if split_after_sentence < len(sentences):
                    split_indices.append(split_after_sentence)
        
        # 7. Chunks aus Sätzen zwischen Split-Punkten erstellen
        split_indices = [0] + sorted(set(split_indices)) + [len(sentences)]
        
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]
            chunk_sentences = sentences[start_idx:end_idx]
            
            if chunk_sentences:
                chunk_text = ' '.join(chunk_sentences)
                chunk_id = generate_chunk_id(strategy, source_id, chunk_index)
                
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    strategy=strategy,
                    source_id=source_id,
                    source_document_id=source_document_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    metadata={'section_header': header}
                ))
                chunk_index += 1
    
    return chunks


def macro_split_for_agentic(document_text: str) -> List[str]:
    """
    Führt einen Macro-Split für das agentische Chunking durch.

    Teilt das Dokument an Markdown-Überschriften und entfernt
    abschließende Abschnitte, die nur aus Überschriften bestehen.

    Args:
        document_text: Der zu splittende Dokumenttext.

    Returns:
        Eine Liste von Section-Texten.
    """
    parts = _RE_AGENTIC_HEADER.split(document_text)
    
    sections = []
    
    # Preamble (Text vor erstem Header)
    if parts and parts[0].strip():
        sections.append(parts[0].strip())
    
    # Header + Content Paare
    i = 1
    while i < len(parts):
        header = parts[i] if i < len(parts) else ""
        content = parts[i + 1] if i + 1 < len(parts) else ""
        full_section = f"{header}\n{content}".strip()
        
        if full_section:
            sections.append(full_section)
        
        i += 2
    
    # Entferne abschließende Abschnitte, die nur Überschriften enthalten
    while sections:
        last = sections[-1]
        lines = last.strip().split('\n')
        # Prüfe, ob der Abschnitt Nicht-Überschriften-Inhalt hat
        has_content = any(line.strip() and not line.strip().startswith('#') for line in lines)
        if not has_content:
            sections.pop()  # Verwerfe Nur-Überschrift-Abschnitt am Ende
        else:
            break
    
    return sections


async def extract_propositions(section_text: str) -> List[str]:
    """
    Extrahiert atomare Propositionen aus einem Textabschnitt.

    Verwendet ein LLM um den Text in semantisch eigenständige,
    atomare Aussagen zu zerlegen.

    Args:
        section_text: Der zu analysierende Section-Text.

    Returns:
        Eine Liste von Proposition-Strings.
    """
    format_instructions = get_proposition_format_instructions()
    
    system_prompt = AGENTIC_PROPOSITION_SYSTEM_PROMPT.format(
        format_instructions=format_instructions
    )
    user_prompt = AGENTIC_PROPOSITION_USER_PROMPT.format(text=section_text)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        response = await async_llm_call(messages, context="agentic")
        content = response['content']
        
        parsed = safe_json_parse(content)
        
        if parsed and 'propositions' in parsed:
            props = [p.get('statement', '') for p in parsed['propositions']]
            return [p for p in props if p]
        
        logger.warning("Proposition extraction returned invalid format, retrying...")
        raise ValueError("Ungültiges JSON-Format vom LLM")
        
    except Exception as e:
        logger.warning(f"Proposition extraction failed: {e}, retrying...")
        # Exception erneut werfen um Retry im Aufrufer auszulösen oder Task fehlschlagen zu lassen
        raise e


async def group_propositions(propositions: List[str]) -> List[Dict[str, Any]]:
    """
    Gruppiert Propositionen thematisch mittels LLM.

    Ordnet die extrahierten Propositionen zu semantisch kohärenten
    Gruppen zusammen und generiert für jede Gruppe einen Titel.

    Args:
        propositions: Liste der zu gruppierenden Proposition-Strings.

    Returns:
        Eine Liste von Gruppen mit 'indices' und 'title' Schlüsseln.
    """
    if len(propositions) <= 1:
        return [{'indices': list(range(len(propositions))), 'title': 'Single Topic'}]
    
    # Nummerierte Propositions erstellen
    numbered = '\n'.join([f"{i}: {prop}" for i, prop in enumerate(propositions)])
    
    format_instructions = get_grouping_format_instructions()
    
    system_prompt = AGENTIC_GROUPING_SYSTEM_PROMPT.format(
        format_instructions=format_instructions
    )
    user_prompt = AGENTIC_GROUPING_USER_PROMPT.format(propositions=numbered)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        response = await async_llm_call(messages, context="agentic")
        content = response['content']
        
        parsed = safe_json_parse(content)
        
        if parsed and 'groups' in parsed:
            groups = []
            used_indices = set()
            
            for group in parsed['groups']:
                indices = group.get('group_indices', [])
                title = group.get('title', 'Untitled')
                
                # Filtere ungültige Indizes
                valid_indices = [i for i in indices if 0 <= i < len(propositions)]
                
                # Entferne bereits verwendete Indizes
                valid_indices = [i for i in valid_indices if i not in used_indices]
                
                if valid_indices:
                    groups.append({'indices': valid_indices, 'title': title})
                    used_indices.update(valid_indices)
            
            # Füge nicht zugewiesene Propositions als einzelne Gruppe hinzu
            missing_indices = [i for i in range(len(propositions)) if i not in used_indices]
            if missing_indices:
                groups.append({'indices': missing_indices, 'title': 'Miscellaneous'})
            
            return groups if groups else [{'indices': list(range(len(propositions))), 'title': 'Single Topic'}]
        
        logger.warning("Grouping returned invalid format, retrying...")
        raise ValueError("Ungültiges JSON-Format vom LLM")
        
    except Exception as e:
        logger.warning(f"Grouping failed: {e}, retrying...")
        raise e


async def agentic_chunk_document(
    document_text: str,
    source_id: str,
    source_document_id: str,
) -> List[Chunk]:
    """
    Segmentiert ein Dokument mittels agentischem Chunking.

    Dieser LLM-gestützte Ansatz extrahiert zunächst atomare Propositionen
    aus dem Text und gruppiert diese dann thematisch zu kohärenten Chunks.

    Args:
        document_text: Der zu segmentierende Dokumenttext.
        source_id: Die ID des Quelldokuments.
        source_document_id: Die Dokument-ID aus der Datenquelle.

    Returns:
        Eine Liste von Chunk-Objekten.
    """
    strategy = "agentic"
    chunks = []
    chunk_index = 0
    
    # 1. Macro-Split
    sections = macro_split_for_agentic(document_text)
    
    if not sections:
        return chunks
    
    # 2. Verarbeite jede Section
    for section in sections:
        if not section.strip():
            continue
        
        # 3. Proposition Extraction
        propositions = await extract_propositions(section)
        
        if not propositions:
            continue
        
        # 4. Grouping
        groups = await group_propositions(propositions)
        
        # 5. Chunk-Erzeugung
        for group in groups:
            indices = group['indices']
            title = group['title']
            
            group_props = [propositions[i] for i in indices if i < len(propositions)]
            
            if group_props:
                # Titel als erste Zeile hinzufügen
                chunk_text = f"{title}\n{' '.join(group_props)}"
                chunk_id = generate_chunk_id(strategy, source_id, chunk_index)
                
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    strategy=strategy,
                    source_id=source_id,
                    source_document_id=source_document_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    metadata={'group_title': title}
                ))
                chunk_index += 1
    
    return chunks


async def agentic_chunk_all_documents(rows: List[Dict]) -> List[Chunk]:
    """
    Führt agentisches Chunking für alle Dokumente parallel durch.

    Verarbeitet alle Dokumente mit Fortschrittsanzeige und implementiert
    Fehlerbehandlung mit Fallback für fehlgeschlagene Dokumente.

    Args:
        rows: Liste von Row-Dictionaries aus der CSV-Datei.

    Returns:
        Eine Liste aller erzeugten Chunk-Objekte.
    """
    from tqdm.asyncio import tqdm_asyncio
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    async def safe_chunk_document(idx: int, row: Dict) -> Tuple[int, Any]:
        """Wrapper, der Exceptions fängt und Index + Ergebnis/Fehler zurückgibt."""
        try:
            result = await agentic_chunk_document(
                document_text=row['Document_text'],
                source_id=row['ID'],
                source_document_id=row['DocumentID'],
            )
            return (idx, result)
        except Exception as e:
            # Falls alle Retries fehlschlagen, wollen wir die Exception trotzdem an die Hauptschleife zurückgeben
            # aber der Retry-Decorator behandelt die Wiederholungen bevor wir hier ankommen
            raise e

    async def process_with_fallback(idx, row):
        try:
            return await safe_chunk_document(idx, row)
        except Exception as e:
            return (idx, e)
    
    # enumerate() gibt Index direkt mit - vermeidet O(n) list.index() Suche
    tasks = [process_with_fallback(idx, row) for idx, row in enumerate(rows)]
    
    # Parallel ausführen mit Fortschrittsbalken
    indexed_results = await tqdm_asyncio.gather(
        *tasks,
        desc="Agentic Chunking",
        total=len(tasks),
    )
    
    all_chunks = []
    for idx, result in indexed_results:
        if isinstance(result, Exception):
            logger.error(f"Agentic chunking failed for document {rows[idx]['ID']}: {result}")
            # Fallback: Ganzes Dokument als ein Chunk
            chunk_id = generate_chunk_id("agentic", rows[idx]['ID'], 0)
            all_chunks.append(Chunk(
                chunk_id=chunk_id,
                strategy="agentic",
                source_id=rows[idx]['ID'],
                source_document_id=rows[idx]['DocumentID'],
                chunk_index=0,
                text=rows[idx]['Document_text'],
                metadata={'fallback': True}
            ))
        else:
            all_chunks.extend(result)
    
    return all_chunks


def build_chroma_index(chunks: List[Chunk], strategy: str) -> None:
    """
    Erstellt einen ChromaDB-Vektorindex für eine Chunking-Strategie.

    Speichert alle Chunks mit ihren Embeddings in einer persistenten
    ChromaDB-Collection für spätere Similarity-Suche.

    Args:
        chunks: Die zu indexierenden Chunk-Objekte.
        strategy: Die Chunking-Strategie (heuristic, semantic, agentic).
    """
    chroma_path = get_chroma_path(strategy)
    ensure_directory(str(chroma_path))
    
    # Chroma Client mit Persistenz
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Collection erstellen/abrufen
    collection_name = f"{strategy}_chunks"
    
    # Lösche existierende Collection falls vorhanden
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Batch-weise Embeddings und Hinzufügen
    batch_size = 100
    
    from tqdm import tqdm
    
    # Progress bar zeigt Chunks statt Batches für bessere Verständlichkeit
    with tqdm(total=len(chunks), desc=f"Indexing {strategy} (Chroma)", unit="chunks") as pbar:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [c.metadata for c in batch]
            
            # Embeddings berechnen
            try:
                embeddings = sync_get_embeddings(documents)
            except Exception as e:
                logger.error(f"Embedding failed for batch {i}: {e}")
                pbar.update(len(batch))
                continue
            
            # Zur Collection hinzufügen
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            pbar.update(len(batch))


def build_bm25_index(chunks: List[Chunk], strategy: str) -> None:
    """
    Erstellt einen BM25-Index für lexikalische Suche.

    Tokenisiert alle Chunk-Texte und erstellt einen invertierten
    Index basierend auf dem BM25-Algorithmus.

    Args:
        chunks: Die zu indexierenden Chunk-Objekte.
        strategy: Die Chunking-Strategie (heuristic, semantic, agentic).
    """
    bm25_path = get_bm25_path(strategy)
    chunk_ids_path = get_chunk_ids_path(strategy)
    
    ensure_parent_directory(str(bm25_path))
    
    # Tokenisiere alle Chunk-Texte
    tokenized_corpus = [tokenize_for_bm25(c.text) for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    
    # BM25 Index erstellen
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Speichere BM25 Index
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)
    
    # Speichere Chunk-IDs Mapping
    with open(chunk_ids_path, 'wb') as f:
        pickle.dump(chunk_ids, f)
    
    logger.info(f"BM25 index for {strategy} created with {len(chunks)} chunks")

def compute_stats(chunks: List[Chunk], strategy: str) -> IndexingStats:
    """
    Berechnet deskriptive Statistiken für indexierte Chunks.

    Ermittelt Anzahl, Längenverteilung (Zeichen und Tokens) sowie
    Metadaten über den erstellten Index.

    Args:
        chunks: Die zu analysierenden Chunk-Objekte.
        strategy: Die Chunking-Strategie.

    Returns:
        Ein IndexingStats-Objekt mit den berechneten Statistiken.
    """
    if not chunks:
        return IndexingStats(
            strategy=strategy,
            num_documents=0,
            num_chunks=0,
            chunk_len_chars_min=0,
            chunk_len_chars_mean=0.0,
            chunk_len_chars_max=0,
            chunk_len_tokens_min=0,
            chunk_len_tokens_mean=0.0,
            chunk_len_tokens_max=0,
            embedding_model=EMBEDDING_MODEL,
            created_at=datetime.now().isoformat(),
        )
    
    # Berechne Längen als numpy-Arrays für vektorisierte Statistiken
    char_lengths = np.array([len(c.text) for c in chunks])
    token_lengths = np.array([count_tokens(c.text) for c in chunks])
    
    # Eindeutige Dokumente mit Set (O(n) statt O(n²))
    unique_docs = len({c.source_id for c in chunks})
    
    return IndexingStats(
        strategy=strategy,
        num_documents=unique_docs,
        num_chunks=len(chunks),
        chunk_len_chars_min=int(char_lengths.min()),
        chunk_len_chars_mean=float(char_lengths.mean()),
        chunk_len_chars_max=int(char_lengths.max()),
        chunk_len_tokens_min=int(token_lengths.min()),
        chunk_len_tokens_mean=float(token_lengths.mean()),
        chunk_len_tokens_max=int(token_lengths.max()),
        embedding_model=EMBEDDING_MODEL,
        created_at=datetime.now().isoformat(),
    )


def write_stats(stats: IndexingStats, strategy: str) -> None:
    """
    Persistiert Indexing-Statistiken als JSON-Datei.

    Args:
        stats: Das zu speichernde IndexingStats-Objekt.
        strategy: Die Chunking-Strategie für den Dateipfad.
    """
    stats_path = get_stats_path(strategy)
    ensure_parent_directory(str(stats_path))
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(stats), f, indent=2)


def chunk_document(
    document_text: str,
    source_id: str,
    source_document_id: str,
    strategy: str,
) -> List[Chunk]:
    """
    Segmentiert ein Dokument mit der angegebenen Chunking-Strategie.

    Dispatcher-Funktion, die je nach Strategie die entsprechende
    Chunking-Implementierung aufruft.

    Args:
        document_text: Der zu segmentierende Dokumenttext.
        source_id: Die eindeutige ID des Quelldokuments.
        source_document_id: Die Dokument-ID aus der Datenquelle.
        strategy: Die anzuwendende Strategie (heuristic, semantic).

    Returns:
        Eine Liste von Chunk-Objekten.

    Raises:
        ValueError: Bei unbekannter Strategie.
    """
    if strategy == "heuristic":
        return heuristic_chunk(document_text, source_id, source_document_id)
    elif strategy == "semantic":
        return semantic_chunk(document_text, source_id, source_document_id)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


async def run_step2_async() -> Dict[str, List[Chunk]]:
    """
    Führt die asynchrone Chunking-Pipeline aus.

    Orchestriert alle drei Chunking-Strategien (heuristic, semantic, agentic)
    und erstellt anschließend die entsprechenden Indizes.

    Returns:
        Ein Dictionary mit Strategie-Namen als Schlüssel und Chunk-Listen als Werte.
    """
    logger.info("STEP 2: Chunking & Indexing")
    
    # Lade Step-1 CSV
    csv_path = str(OUTPUT_CSV_PATH)
    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path, encoding=CSV_ENCODING)
    rows = df.to_dict('records')
    
    all_chunks = {}
    
    # 1. Heuristic Chunking (synchron)
    heuristic_chunks = []
    # Verwende tqdm für Fortschrittsanzeige
    from tqdm import tqdm
    for row in tqdm(rows, desc="Heuristic Chunking"):
        chunks = heuristic_chunk(
            document_text=row['Document_text'],
            source_id=str(row['ID']),
            source_document_id=str(row['DocumentID']),
        )
        heuristic_chunks.extend(chunks)
    all_chunks['heuristic'] = heuristic_chunks
    logger.info(f"Heuristic: {len(heuristic_chunks)} chunks")
    
    # 2. Semantic Chunking (synchron, aber mit Embeddings)
    semantic_chunks = []
    for row in tqdm(rows, desc="Semantic Chunking"):
        chunks = semantic_chunk(
            document_text=row['Document_text'],
            source_id=str(row['ID']),
            source_document_id=str(row['DocumentID']),
        )
        semantic_chunks.extend(chunks)
    all_chunks['semantic'] = semantic_chunks
    logger.info(f"Semantic: {len(semantic_chunks)} chunks")
    
    # 3. Agentic Chunking (asynchron mit Fortschrittsbalken)
    agentic_chunks = await agentic_chunk_all_documents(rows)
    all_chunks['agentic'] = agentic_chunks
    logger.info(f"Agentic: {len(agentic_chunks)} chunks")
    
    # 4. Indexing für jede Strategie
    for strategy in CHUNKING_STRATEGIES:
        chunks = all_chunks[strategy]
        
        # Chroma Index
        build_chroma_index(chunks, strategy)
        
        # BM25 Index
        build_bm25_index(chunks, strategy)
        
        # Statistiken
        stats = compute_stats(chunks, strategy)
        write_stats(stats, strategy)
    
    return all_chunks


def run_step2() -> Dict[str, List[Chunk]]:
    """
    Führt die Chunking- und Indexierungspipeline aus.

    Einstiegspunkt für Schritt 2, der die asynchrone Pipeline
    in einem synchronen Kontext ausführt.

    Returns:
        Ein Dictionary mit Strategie-Namen als Schlüssel und Chunk-Listen als Werte.
    """
    return asyncio.run(run_step2_async())


if __name__ == "__main__":
    run_step2()
