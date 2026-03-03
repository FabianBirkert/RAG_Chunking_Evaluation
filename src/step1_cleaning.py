# =============================================================================
# STEP 1: DATENBEREINIGUNG & VORBEREITUNG
# =============================================================================
"""
Datenbereinigung und Vorbereitung für die RAG-Pipeline.

Dieses Modul implementiert den ersten Schritt der RAG-Pipeline und ist
verantwortlich für das Laden, Filtern und Bereinigen des Natural Questions
Datasets. Es transformiert die Rohdaten in ein für die nachfolgenden
Chunking-Strategien geeignetes Format.

Die Hauptfunktionalitäten umfassen:
    - Automatischer Download des Datasets bei Bedarf
    - Filterung nach Qualitätskriterien (Short Answer Länge, Eindeutigkeit)
    - HTML-zu-Markdown Konvertierung mit anschließender Bereinigung
    - Reservoir Sampling für reproduzierbare Stichprobenziehung
    - Export in ein standardisiertes CSV-Format
"""

import os
import re
import gzip
import json
import random
import logging
import time
import urllib.request
from typing import Optional, Dict, Any, List, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from markdownify import markdownify as md
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("RAG-Pipeline.Step1")

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR,
    NQ_SIMPLIFIED_DIR,
    NQ_DATASET_PATH,
    NQ_DOWNLOAD_URL,
    SAMPLE_SIZE,
    RANDOM_SEED,
    MAX_STREAM_EXAMPLES_TO_SCAN,
    SHORT_ANSWER_MIN_CHARS,
    SHORT_ANSWER_MAX_CHARS,
)
from src.utils import ensure_parent_directory, normalize_whitespace


# =============================================================================
# LOKALE KONSTANTEN
# =============================================================================
OUTPUT_CSV_PATH = DATA_DIR / "input" / "nq_validation_cleaned.csv"
CSV_ENCODING = "utf-8"


_RE_NAV_BOILERPLATE = re.compile(
    r'^(?:Jump to :|For other uses ,|Not to be confused with).*?$',
    re.MULTILINE | re.IGNORECASE
)
_RE_TOC_BLOCK = re.compile(r'## Contents.*?(?=\n## )', re.DOTALL | re.IGNORECASE)
_RE_EDIT_PARENS = re.compile(r'\s*\(\s*edit\s*\)', re.IGNORECASE)
_RE_HATNOTES = re.compile(
    r'^\s*(?:Main article|Further information|See also)\s*:.*?$',
    re.MULTILINE | re.IGNORECASE
)
_RE_FOOTER = re.compile(
    r'^##\s*(?:References|External links|See also|Sources|Notes|Bibliography)\s*$',
    re.MULTILINE | re.IGNORECASE
)
_RE_WIKIPEDIA_RETRIEVED = re.compile(r'Retrieved from `` https://en\.wikipedia\.org.*', re.DOTALL)
_RE_CATEGORIES = re.compile(r'Categories\s*:.*', re.DOTALL)
_RE_TRIPLE_NEWLINES = re.compile(r'\n{3,}')
_RE_NQ_HTML_FIX = re.compile(r'<(Th|Td)_colspan="(\d+)">', re.IGNORECASE)


@dataclass
class CleanedExample:
    """
    Repräsentiert ein bereinigtes Beispiel aus dem Natural Questions Dataset.

    Attributes:
        id: Eindeutige Kennung des Beispiels.
        question: Die Frage als Textstring.
        document_id: Identifikator des zugehörigen Wikipedia-Dokuments.
        document_text: Der bereinigte Dokumenttext im Markdown-Format.
        document_length: Zeichenlänge des bereinigten Dokuments.
        short_answer: Die Ground-Truth-Kurzantwort.
    """
    id: str
    question: str
    document_id: str
    document_text: str
    document_length: int
    short_answer: str


def ensure_dataset_exists() -> Path:
    """
    Stellt die Verfügbarkeit des Datasets sicher.

    Prüft, ob das Natural Questions Dataset lokal vorhanden ist.
    Falls nicht, wird es automatisch von Google Cloud Storage
    heruntergeladen (~4.4 GB, dauert einige Minuten).

    Returns:
        Der Pfad zum lokalen Dataset.

    Raises:
        RuntimeError: Wenn der Download fehlschlägt.
    """
    if NQ_DATASET_PATH.exists():
        return NQ_DATASET_PATH
    
    # Verzeichnis erstellen falls nicht vorhanden
    NQ_SIMPLIFIED_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Dataset nicht gefunden. Starte automatischen Download...")
    logger.info(f"URL: {NQ_DOWNLOAD_URL}")
    logger.info("Dies kann einige Minuten dauern (~4.4 GB)...")
    
    try:
        # Download mit Fortschrittsanzeige
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                # Nur alle 5% loggen um Spam zu vermeiden
                if int(percent) % 5 == 0 and int(percent) != getattr(report_progress, 'last_percent', -1):
                    logger.info(f"Download: {percent:.0f}% ({mb_downloaded:.0f}/{mb_total:.0f} MB)")
                    report_progress.last_percent = int(percent)
        
        urllib.request.urlretrieve(NQ_DOWNLOAD_URL, NQ_DATASET_PATH, report_progress)
        logger.info("Download abgeschlossen!")
        
    except Exception as e:
        # Falls Download fehlschlägt, Anleitung zum manuellen Download
        if NQ_DATASET_PATH.exists():
            NQ_DATASET_PATH.unlink()  # Teilweise heruntergeladene Datei löschen
        raise RuntimeError(
            f"\nDataset-Download fehlgeschlagen: {e}\n\n"
            f"Bitte manuell herunterladen:\n"
            f"  1. Öffne im Browser: {NQ_DOWNLOAD_URL}\n"
            f"  2. Speichere die Datei als: {NQ_DATASET_PATH}\n"
        )
    
    return NQ_DATASET_PATH


def stream_dataset(dataset_path: Path) -> Iterator[Dict]:
    """
    Streamt das Dataset zeilenweise aus einer gzip-komprimierten JSONL-Datei.

    Diese Funktion ermöglicht die speichereffiziente Verarbeitung großer
    Datasets durch zeilenweises Lesen und Parsing.

    Args:
        dataset_path: Der Pfad zur .jsonl.gz Datei.

    Yields:
        Ein Dictionary für jede gültige JSON-Zeile im Dataset.
    """
    with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def clean_text(text: str) -> str:
    """
    Bereinigt Markdown-Text von Wikipedia-spezifischen Elementen.

    Diese Funktion entfernt typische Störelemente aus Wikipedia-Dokumenten:
    Navigationselemente, Inhaltsverzeichnisse, Bearbeitungslinks,
    Hinweisboxen und Footer-Bereiche.

    Args:
        text: Der zu bereinigende Markdown-Text.

    Returns:
        Der bereinigte Textstring.
    """
    text = _RE_NAV_BOILERPLATE.sub('', text)
    text = _RE_TOC_BLOCK.sub('', text)
    text = text.replace(' ( edit )', '')
    text = _RE_EDIT_PARENS.sub('', text)
    text = _RE_HATNOTES.sub('', text)
    
    footer_match = _RE_FOOTER.search(text)
    if footer_match:
        text = text[:footer_match.start()]
    
    text = _RE_WIKIPEDIA_RETRIEVED.sub('', text)
    text = _RE_CATEGORIES.sub('', text)
    text = _RE_TRIPLE_NEWLINES.sub('\n\n', text)
    
    return text.strip()


def html_to_markdown(html: str) -> str:
    """
    Konvertiert HTML zu Markdown mit der markdownify-Bibliothek.

    Diese Funktion korrigiert zunächst NQ-spezifische HTML-Fehler
    und führt dann die Konvertierung zu Markdown durch.

    Hinweis:
        Komplexe Tabellen mit colspan/rowspan werden von markdownify
        nicht vollständig unterstützt. Die Zellinhalte bleiben erhalten,
        aber die Spaltenausrichtung kann bei merged cells verrutschen.
        Für RAG-Anwendungen ist der Textinhalt wichtiger als die
        exakte Tabellenformatierung.

    Args:
        html: Der zu konvertierende HTML-String.

    Returns:
        Der konvertierte Markdown-String.
    """
    try:
        html = _RE_NQ_HTML_FIX.sub(
            lambda m: f'<{m.group(1)} colspan="{m.group(2)}">',
            html
        )
        return md(html, heading_style="ATX", strip=['script', 'style'])
    except Exception as e:
        logger.warning(f"markdownify failed: {e}")
        return html


def tokens_to_text(tokens: List[str], is_html_list: List[bool], exclude_html: bool = True) -> str:
    """
    Konvertiert eine Token-Liste zu einem zusammenhängenden Text.

    Verarbeitet das NQ Simplified Format, bei dem Dokumente als
    tokenisierte Strings mit optionalen HTML-Markierungen vorliegen.

    Args:
        tokens: Liste der Token-Strings.
        is_html_list: Liste der HTML-Flags für jeden Token.
        exclude_html: Ob HTML-markierte Tokens ausgeschlossen werden sollen.

    Returns:
        Der zusammengefügte und normalisierte Textstring.
    """
    if not tokens:
        return ""
    
    from itertools import zip_longest
    
    if exclude_html:
        text_parts = [
            token_text 
            for token_text, is_html in zip_longest(tokens, is_html_list, fillvalue=False)
            if token_text and not is_html
        ]
    else:
        text_parts = [token_text for token_text in tokens if token_text]
    
    return normalize_whitespace(' '.join(text_parts))


def count_short_answers(example: Dict) -> int:
    """
    Zählt die Anzahl der Short Answers in einem Beispiel.

    Iteriert über alle Annotationen eines Beispiels und summiert
    die Anzahl der Short Answer Spans.

    Args:
        example: Das Beispiel-Dictionary aus dem Dataset.

    Returns:
        Die Gesamtanzahl der Short Answer Spans.
    """
    annotations = example.get('annotations', [])
    if not annotations:
        return 0
    
    total = 0
    for annotation in annotations:
        short_answers = annotation.get('short_answers', [])
        total += len(short_answers)
    
    return total


def extract_short_answer(example: Dict) -> Optional[str]:
    """
    Extrahiert die Short Answer aus einem Beispiel.

    Verwendet die Token-Positionen aus den Annotationen, um den
    Antworttext aus dem Dokument zu extrahieren.

    Args:
        example: Das Beispiel-Dictionary aus dem Dataset.

    Returns:
        Der extrahierte Antworttext oder None bei Fehler.
    """
    annotations = example.get('annotations', [])
    if not annotations:
        return None
    
    for annotation in annotations:
        short_answers = annotation.get('short_answers', [])
        if short_answers:
            sa = short_answers[0]
            start_token = sa.get('start_token', 0)
            end_token = sa.get('end_token', 0)
            
            document_text = example.get('document_text', '')
            if document_text and end_token > start_token:
                tokens = document_text.split()
                answer_tokens = tokens[start_token:end_token]
                return normalize_whitespace(' '.join(answer_tokens))
    
    return None


def passes_filter(example: Dict) -> bool:
    """
    Prüft, ob ein Beispiel alle Filterkriterien erfüllt.

    Die Kriterien sind:
    - Vorhandensein von Annotationen
    - Exakt eine Short Answer über alle Annotationen
    - Short Answer Länge zwischen 10 und 100 Zeichen

    Args:
        example: Das Beispiel-Dictionary aus dem Dataset.

    Returns:
        True wenn alle Kriterien erfüllt sind, sonst False.
    """
    annotations = example.get('annotations')
    if not annotations:
        return False
    
    total_short_answers = count_short_answers(example)
    if total_short_answers != 1:
        return False
    
    short_answer = extract_short_answer(example)
    if short_answer is None:
        return False
    
    answer_len = len(short_answer.strip())
    if not (SHORT_ANSWER_MIN_CHARS <= answer_len <= SHORT_ANSWER_MAX_CHARS):
        return False
    
    return True


def process_document(example: Dict) -> str:
    """
    Verarbeitet ein Dokument zu bereinigtem Markdown.

    Konvertiert den HTML-Inhalt des Dokuments zu Markdown und
    wendet anschließend die Bereinigungsfunktion an.

    Args:
        example: Das Beispiel-Dictionary aus dem Dataset.

    Returns:
        Der bereinigte Dokumenttext als String.
    """
    document_text = example.get('document_text', '')
    
    if not document_text:
        logger.warning(f"No document content found for {example.get('example_id', 'unknown')}")
        return ""
    
    try:
        markdown = html_to_markdown(document_text)
        return clean_text(markdown)
    except Exception as e:
        logger.warning(f"HTML processing failed for {example.get('example_id', 'unknown')}: {e}")
        # Fallback: Nur Bereinigung ohne Markdown-Konvertierung
        return clean_text(document_text)


def get_document_id(example: Dict) -> str:
    """
    Bestimmt die Dokument-ID für ein Beispiel.

    Verwendet eine Prioritätsreihenfolge: URL, Titel, Example-ID.

    Args:
        example: Das Beispiel-Dictionary aus dem Dataset.

    Returns:
        Die ermittelte Dokument-ID als String.
    """
    url = example.get('document_url', '')
    if url:
        return url
    
    title = example.get('document_title', '')
    if title:
        return title
    
    return str(example.get('example_id', 'unknown'))


def transform_example(example: Dict) -> CleanedExample:
    """
    Transformiert ein Roh-Beispiel in ein bereinigtes CleanedExample.

    Extrahiert alle relevanten Felder, verarbeitet das Dokument
    und erstellt das finale Datenobjekt.

    Args:
        example: Das Roh-Beispiel aus dem Dataset.

    Returns:
        Ein CleanedExample-Objekt mit allen bereinigten Daten.
    """
    example_id = str(example.get('example_id', ''))
    question = example.get('question_text', '')
    document_text = process_document(example)
    short_answer = extract_short_answer(example)
    
    return CleanedExample(
        id=example_id,
        question=question,
        document_id=get_document_id(example),
        document_text=document_text,
        document_length=len(document_text),
        short_answer=short_answer or "",
    )


def reservoir_sample(
    stream: Iterator[Dict],
    sample_size: int,
    seed: int,
    max_to_scan: int,
) -> List[CleanedExample]:
    """
    Führt Reservoir Sampling auf einem Stream von Beispielen durch.

    Das Reservoir Sampling ermöglicht die Ziehung einer gleichverteilten
    Stichprobe aus einem Datenstrom unbekannter Größe. Der Seed
    gewährleistet die Reproduzierbarkeit der Ergebnisse.

    Args:
        stream: Iterator über die Dataset-Beispiele.
        sample_size: Die gewünschte Stichprobengröße.
        seed: Der Random Seed für Reproduzierbarkeit.
        max_to_scan: Maximale Anzahl zu scannender Beispiele.

    Returns:
        Eine Liste von CleanedExample-Objekten.

    Raises:
        ValueError: Wenn nicht genügend passende Beispiele gefunden wurden.
    """
    rng = random.Random(seed)
    reservoir: List[CleanedExample] = []
    seen_filtered = 0
    scanned = 0
    target_matches = sample_size * 100
    logger.info(f"Reservoir Sampling: seed={seed}, target={sample_size} samples from {target_matches} matches")
    logger.info(f"Scanning up to {max_to_scan:,} documents...")
    
    pbar = tqdm(
        total=target_matches,
        desc="Finding matches",
        unit="matches",
        ncols=80,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )
    
    for example in stream:
        scanned += 1
        
        if not passes_filter(example):
            continue
        
        seen_filtered += 1
        pbar.update(1)
        
        if len(reservoir) < sample_size:
            transformed = transform_example(example)
            reservoir.append(transformed)
        else:
            j = rng.randint(1, seen_filtered)
            if j <= sample_size:
                transformed = transform_example(example)
                reservoir[j - 1] = transformed
        
        if scanned >= max_to_scan:
            break
        
        # Frühzeitiger Abbruch wenn genug Samples gesehen
        if len(reservoir) >= sample_size and seen_filtered >= target_matches:
            break
    
    pbar.close()
    
    logger.info(f"Result: {scanned:,} scanned → {seen_filtered:,} matches → {len(reservoir)} samples selected")
    
    if len(reservoir) < sample_size:
        raise ValueError(
            f"Not enough matching examples found. "
            f"Needed {sample_size}, got {len(reservoir)} after scanning {scanned} examples."
        )
    
    return reservoir


def write_csv(examples: List[CleanedExample], output_path: str) -> None:
    """
    Schreibt die bereinigten Beispiele in eine CSV-Datei.

    Exportiert die CleanedExample-Objekte in das standardisierte
    CSV-Format für die nachfolgenden Pipeline-Schritte.

    Args:
        examples: Liste der zu exportierenden CleanedExample-Objekte.
        output_path: Der Zielpfad für die CSV-Datei.
    """
    ensure_parent_directory(output_path)
    
    data = [asdict(ex) for ex in examples]
    df = pd.DataFrame(data)
    
    column_mapping = {
        'id': 'ID',
        'question': 'Question',
        'document_id': 'DocumentID',
        'document_text': 'Document_text',
        'document_length': 'Documentlength',
        'short_answer': 'short_answer'
    }
    
    df = df.rename(columns=column_mapping)
    df = df[list(column_mapping.values())]
    
    df.to_csv(output_path, index=False, encoding=CSV_ENCODING)
    logger.info(f"Wrote {len(examples)} examples to CSV")


def validate_output(csv_path: str) -> bool:
    """
    Validiert die erzeugte CSV-Datei.

    Prüft die Vollständigkeit und Korrektheit der Ausgabedatei
    anhand der definierten Qualitätskriterien.

    Args:
        csv_path: Der Pfad zur zu validierenden CSV-Datei.

    Returns:
        True wenn alle Validierungskriterien erfüllt sind, sonst False.
    """
    if not os.path.exists(csv_path):
        return False
    
    df = pd.read_csv(csv_path, encoding=CSV_ENCODING)
    
    if len(df) != SAMPLE_SIZE:
        return False
    
    required_columns = {'ID', 'Question', 'DocumentID', 'Document_text', 'Documentlength', 'short_answer'}
    if not required_columns.issubset(df.columns):
        return False
    
    answer_lengths = df['short_answer'].astype(str).str.strip().str.len()
    valid_lengths = answer_lengths.between(SHORT_ANSWER_MIN_CHARS, SHORT_ANSWER_MAX_CHARS)
    
    return valid_lengths.all()


def run_step1() -> str:
    """
    Führt den ersten Schritt der RAG-Pipeline aus.

    Dieser Schritt umfasst das Laden des Datasets, die Filterung
    nach Qualitätskriterien, die Datenbereinigung und den Export
    in das standardisierte CSV-Format.

    Returns:
        Der Pfad zur erzeugten CSV-Datei.
    """
    logger.info("STEP 1: Data Cleaning")
    
    dataset_path = ensure_dataset_exists()
    logger.info(f"Dataset: {dataset_path.name}")
    
    stream = stream_dataset(dataset_path)
    
    examples = reservoir_sample(
        stream=stream,
        sample_size=SAMPLE_SIZE,
        seed=RANDOM_SEED,
        max_to_scan=MAX_STREAM_EXAMPLES_TO_SCAN,
    )
    
    output_path = str(OUTPUT_CSV_PATH)
    write_csv(examples, output_path)
    
    if validate_output(output_path):
        logger.info(f"Step 1 completed successfully!")
        logger.info(f"Output: {output_path}")
    else:
        logger.error(f"Step 1 validation failed!")
    
    return output_path


if __name__ == "__main__":
    run_step1()
