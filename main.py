"""
main.py - Haupteinstiegspunkt der RAG-Pipeline

Dieses Modul stellt den zentralen Einstiegspunkt für die Ausführung
der vollständigen RAG-Pipeline bereit. Es orchestriert die vier
Verarbeitungsschritte und bietet flexible Kommandozeilenoptionen
für die selektive Ausführung einzelner Schritte.

Pipeline-Schritte:
    1. Datenbereinigung und -vorbereitung
    2. Chunking und Indexierung
    3. Retrieval und Antwortgenerierung
    4. Evaluierung und Metriken

Kommandozeilenoptionen:
    --step N: Führt nur Schritt N aus
    --from-step N: Beginnt bei Schritt N
    --to-step N: Endet bei Schritt N
    --verbose: Aktiviert Debug-Logging

Typische Verwendung:
    python main.py                    # Vollständige Pipeline
    python main.py --step 2           # Nur Chunking
    python main.py --from-step 3      # Ab Retrieval
"""

import argparse
import logging
import sys
import os
from pathlib import Path


def check_environment() -> None:
    """
    Prüft, ob alle Requirements installiert und die .env Datei konfiguriert ist.
    
    Bricht das Programm ab, falls:
        - Kritische Pakete nicht installiert sind
        - Die .env Datei fehlt oder keine gültigen API-Keys enthält
    """
    # Prüfe kritische Pakete
    critical_packages = [
        ("dotenv", "python-dotenv"),
        ("pandas", "pandas"),
        ("chromadb", "chromadb"),
        ("langchain", "langchain"),
        ("openai", "openai"),
        ("tiktoken", "tiktoken"),
        ("rank_bm25", "rank_bm25"),
        ("tqdm", "tqdm"),
        ("ragas", "ragas"),
        ("nltk", "nltk"),
    ]
    
    missing_packages = []
    for import_name, pip_name in critical_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("=" * 60)
        print("FEHLER: Fehlende Abhängigkeiten!")
        print("=" * 60)
        print(f"\nFolgende Pakete sind nicht installiert: {', '.join(missing_packages)}")
        print("\nBitte installiere alle Abhängigkeiten mit:")
        print()
        print("    pip install -r requirements.txt")
        print()
        sys.exit(1)
    
    # Prüfe .env Datei und API Keys
    base_dir = Path(__file__).parent
    env_path = base_dir / ".env"
    env_template_path = base_dir / ".env.template"
    
    if not env_path.exists():
        print("=" * 60)
        print("FEHLER: .env Datei nicht gefunden!")
        print("=" * 60)
        print("\nDie .env Datei mit den API-Keys fehlt.")
        print()
        if env_template_path.exists():
            print("Erstelle eine .env Datei basierend auf dem Template:")
            print()
            print("    cp .env.template .env       (Unix/Mac)")
            print("    copy .env.template .env     (Windows)")
            print()
            print("Dann trage deine API-Keys in die .env Datei ein.")
        else:
            print("Erstelle eine .env Datei mit folgendem Inhalt:")
            print()
            print("    OPENAI_API_KEY=sk-your-key-here")
            print("    OPENROUTER_API_KEY=sk-or-v1-your-key-here")
        print()
        sys.exit(1)
    
    # Lade .env temporär um Keys zu prüfen
    from dotenv import dotenv_values
    env_values = dotenv_values(env_path)
    
    # Prüfe ob mindestens ein API Key gesetzt ist
    openai_key = env_values.get("OPENAI_API_KEY", "")
    openrouter_key = env_values.get("OPENROUTER_API_KEY", "")
    
    # Platzhalter erkennen (nur exakte Platzhalter-Muster)
    placeholder_patterns = ["your-key-here", "sk-your-key-here", "sk-or-v1-your-key-here"]
    
    def is_valid_key(key: str) -> bool:
        if not key or not key.strip():
            return False
        key = key.strip()
        # Prüfe auf Platzhalter
        if any(p in key for p in placeholder_patterns):
            return False
        # Mindestlänge für echte Keys
        if len(key) < 20:
            return False
        return True
    
    openai_valid = is_valid_key(openai_key)
    openrouter_valid = is_valid_key(openrouter_key)
    
    if not openai_valid and not openrouter_valid:
        print("=" * 60)
        print("FEHLER: Keine gültigen API-Keys in .env!")
        print("=" * 60)
        print("\nBitte trage mindestens einen gültigen API-Key in die .env Datei ein:")
        print()
        print("  Für OpenAI:     OPENAI_API_KEY=sk-...")
        print("  Für OpenRouter: OPENROUTER_API_KEY=sk-or-v1-...")
        print()
        print("Die aktuellen Werte sind Platzhalter oder leer.")
        print()
        sys.exit(1)


# Prüfe Umgebung bevor irgendetwas anderes passiert
check_environment()

from dotenv import load_dotenv

# Lade Environment-Variablen
load_dotenv()


def setup_logging(verbose: bool = False) -> None:
    """
    Konfiguriert das zentrale Logging-System für die Pipeline.

    Initialisiert den Root-Logger mit einheitlichem Format und setzt
    externe Bibliotheken auf reduzierte Log-Level.

    Args:
        verbose: Bei True wird DEBUG-Level aktiviert, sonst INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Root Logger konfigurieren
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Externe Libraries auf WARNING setzen
    for lib in ["httpx", "openai", "chromadb", "httpcore", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


# Setup Logging (wird beim Import ausgeführt)
setup_logging()
logger = logging.getLogger("RAG-Pipeline")

# Füge src zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent))

from config import ensure_directories


def print_step_header(step: int) -> None:
    """
    Gibt eine formatierte Überschrift für einen Pipeline-Schritt aus.

    Args:
        step: Die Schrittnummer (1-4).
    """
    step_titles = {
        1: "STEP 1 - DATA CLEANING & PREPARATION",
        2: "STEP 2 - CHUNKING & INDEXING",
        3: "STEP 3 - RETRIEVAL & GENERATION",
        4: "STEP 4 - EVALUATION",
    }
    title = step_titles.get(step, f"STEP {step}")
    separator = "=" * 77
    logger.info("")
    logger.info(separator)
    logger.info(f"# {title}")
    logger.info(separator)


def run_step1():
    """
    Führt die Datenbereinigung und -vorbereitung aus.

    Returns:
        Der Pfad zur erzeugten CSV-Datei.
    """
    from src.step1_cleaning import run_step1 as step1
    return step1()


def run_step2():
    """
    Führt das Chunking und die Indexierung aus.

    Returns:
        Ein Dictionary mit Chunk-Listen pro Strategie.
    """
    from src.step2_chunking import run_step2 as step2
    return step2()


def run_step3():
    """
    Führt das Retrieval und die Antwortgenerierung aus.

    Returns:
        Der Pfad zur erzeugten Ergebnis-CSV-Datei.
    """
    from src.step3_retrieval import run_step3 as step3
    return step3()


def run_step4():
    """
    Führt die Evaluierung und Metrikberechnung aus.

    Returns:
        Ein Tuple aus (Ergebnis-CSV-Pfad, Summary-JSON-Pfad).
    """
    from src.step4_evaluation import run_step4 as step4
    return step4()


def main():
    """
    Haupteinstiegspunkt für die Pipeline-Ausführung.

    Parst Kommandozeilenargumente und führt die gewählten
    Pipeline-Schritte in der definierten Reihenfolge aus.

    Returns:
        Ein Dictionary mit den Ergebnissen jedes ausgeführten Schritts.
    """
    parser = argparse.ArgumentParser(
        description="RAG Chunking & Evaluation Framework"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run only a specific step (1-4). If not specified, runs all steps."
    )
    parser.add_argument(
        "--from-step",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Start from a specific step (default: 1)"
    )
    parser.add_argument(
        "--to-step",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="End at a specific step (default: 4)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    args = parser.parse_args()
    
    # Rekonfiguriere Logging falls verbose aktiviert
    if args.verbose:
        setup_logging(verbose=True)
    
    # Stelle sicher, dass alle Verzeichnisse existieren
    ensure_directories()
    
    logger.info("RAG Chunking & Evaluation Framework")
    
    # Bestimme welche Schritte ausgeführt werden
    if args.step:
        steps_to_run = [args.step]
    else:
        steps_to_run = list(range(args.from_step, args.to_step + 1))
    
    logger.info(f"Running steps: {steps_to_run}")
    
    # Führe Schritte aus
    step_functions = {
        1: run_step1,
        2: run_step2,
        3: run_step3,
        4: run_step4,
    }
    
    results = {}
    
    for step in steps_to_run:
        try:
            print_step_header(step)
            result = step_functions[step]()
            results[step] = result
        except Exception as e:
            logger.error(f"Step {step} failed: {e}")
            raise
    
    logger.info("")
    logger.info("=" * 77)
    logger.info("# PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 77)
    
    return results


if __name__ == "__main__":
    main()
