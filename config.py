"""
Zentrale Konfigurationsdatei für die RAG-Pipeline.

Dieses Modul definiert alle Parameter und Konstanten, die in der gesamten
Retrieval-Augmented Generation (RAG) Pipeline verwendet werden. Es dient als
zentrale Anlaufstelle für die Konfiguration aller vier Verarbeitungsschritte:
Datenbereinigung, Chunking, Retrieval und Evaluation.

Die Konfiguration umfasst:
    - Verzeichnisstrukturen und Dateipfade
    - Parameter für das Natural Questions Dataset
    - Chunking-Strategien (heuristisch, semantisch, agentisch)
    - Retrieval-Methoden (sparse, dense, hybrid)
    - LLM- und Embedding-Modellkonfiguration
    - Prompt-Templates für die Generierung und Evaluation
"""

import os
from pathlib import Path


# =============================================================================
# VERZEICHNISSTRUKTUR
# =============================================================================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"

# =============================================================================
# NATURAL QUESTIONS DATASET
# =============================================================================
NQ_SIMPLIFIED_DIR = DATA_DIR / "nq_simplified"
NQ_DATASET_FILENAME = "simplified-nq-train.jsonl.gz"
NQ_DATASET_PATH = NQ_SIMPLIFIED_DIR / NQ_DATASET_FILENAME
NQ_DOWNLOAD_URL = "https://storage.googleapis.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz"

# =============================================================================
# SAMPLING & FILTERKRITERIEN
# =============================================================================
SAMPLE_SIZE = 250
RANDOM_SEED = 42
MAX_STREAM_EXAMPLES_TO_SCAN = 200000

SHORT_ANSWER_MIN_CHARS = 10
SHORT_ANSWER_MAX_CHARS = 100
REQUIRE_EXACTLY_ONE_SHORT_ANSWER = True

# =============================================================================
# HEURISTISCHES CHUNKING
# =============================================================================
HEURISTIC_CHUNK_TOKENS = 512
HEURISTIC_OVERLAP_TOKENS = 51

# =============================================================================
# SEMANTISCHES CHUNKING
# =============================================================================
SEMANTIC_SENTENCE_MIN_CHARS = 50
SEMANTIC_WINDOW_SENTENCE_COUNT = 2
SEMANTIC_SIMILARITY_THRESHOLD = 0.6

# =============================================================================
# RETRIEVAL
# =============================================================================
RETRIEVAL_TOP_K = 10
RETRIEVAL_METHODS = ["sparse", "dense", "hybrid"]
RRF_K = 60

# =============================================================================
# PARALLELISIERUNG & RATE LIMITING
# =============================================================================
MAX_CONCURRENT_REQUESTS = 10
MAX_CONCURRENT_AGENTIC = 10
MAX_CONCURRENT_EVALUATION = 10
MAX_CONCURRENT_RAGAS = 5
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60

# =============================================================================
# API-PROVIDER & MODELLE
# =============================================================================
USE_OPENROUTER = True
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
PROMPT_VERSION = "step3_v1_context_only"

EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"

# =============================================================================
# INDEXIERUNG & SPEICHERUNG
# =============================================================================
VECTOR_STORE = "chroma"
CHROMA_PERSIST_ROOT = DATA_DIR / "indices"

INVERTED_INDEX_TYPE = "bm25"

CACHE_ROOT = DATA_DIR / "cache"

# =============================================================================
# PIPELINE-KONFIGURATION
# =============================================================================
CHUNKING_STRATEGIES = ["heuristic", "semantic", "agentic"]

TIKTOKEN_ENCODING = "cl100k_base"

# =============================================================================
# PROMPTS - AGENTISCHES CHUNKING
# =============================================================================
AGENTIC_PROPOSITION_SYSTEM_PROMPT = """You are an expert in Natural Language Processing. Your task is to decompose the given text into atomic, self-contained propositions. 

Guidelines:
1. Each proposition must be a complete sentence.
2. Resolve all coreferences (e.g., replace 'he', 'it', 'they' with the specific entity names they refer to).
3. Each proposition should contain exactly one distinct fact or idea.
4. The propositions should be understandable in isolation, without needing the surrounding context.

Example:
Input: "Greg went to the park. He played football there."
Output: ["Greg went to the park.", "Greg played football at the park."]

{format_instructions}"""

AGENTIC_PROPOSITION_USER_PROMPT = """Text to decompose:
\"\"\"{text}\"\"\""""

AGENTIC_GROUPING_SYSTEM_PROMPT = """You are an expert in semantic grouping. Your task is to group the following propositions into semantically coherent chunks.

Guidelines:
1. Group propositions that discuss the same specific topic or event.
2. You may group non-sequential propositions if they are strongly related to the same specific topic.
3. CRITICAL: You MUST include EVERY proposition index in exactly one group. Do not skip any index.
4. CRITICAL: Check your output. If there are N propositions, indices 0 to N-1 must all appear.
5. CRITICAL: Each group must contain a MAXIMUM of 15 propositions. If a topic has more, split it into Part 1 and Part 2.
6. Generate a short, descriptive title for each group.
7. Return a list of groups, where each group contains the indices and the title.

{format_instructions}"""

AGENTIC_GROUPING_USER_PROMPT = """Propositions to group:
\"\"\"{propositions}\"\"\""""

# =============================================================================
# PROMPTS - ANTWORTGENERIERUNG
# =============================================================================
GENERATION_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided context.

Guidelines:
1. Answer the question using ONLY the provided context.
2. Provide only the direct answer in a brief phrase or a few words.
3. Do not use complete sentences.
4. If the answer cannot be found in the context, say "I cannot answer this question from the given context."

Example:
Question: Who wrote Romeo and Juliet?
Answer: William Shakespeare

Question: When was the Eiffel Tower built?
Answer: 1887-1889

Question: What is the capital of France?
Answer: Paris"""

GENERATION_USER_PROMPT = """Context:
\"\"\"{context_string}\"\"\"

Question: {question}"""

# =============================================================================
# PROMPTS - LLM JUDGE (EVALUIERUNG)
# =============================================================================
JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating retrieval quality for a Question Answering system.

Guidelines:
1. Determine if the retrieved chunk contains the information necessary to answer the question.
2. The chunk must contain the semantic meaning of the ground truth answer.
3. Different wording is acceptable as long as the meaning is preserved.
4. If the chunk discusses the same topic but misses the specific answer, it is IRRELEVANT.
5. Respond ONLY with 'YES' or 'NO'."""

JUDGE_USER_PROMPT = """Question: {question}
Ground Truth Answer: {short_answer}

Retrieved Chunk:
\"\"\"{chunk_text}\"\"\"

Does this chunk contain the answer?"""


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================
def ensure_directories() -> None:
    """
    Erstellt alle erforderlichen Verzeichnisse der Pipeline.

    Diese Funktion stellt sicher, dass die Verzeichnisstruktur für Input-Daten,
    Indizes, Output-Dateien und Cache-Speicher existiert. Sie wird beim Start
    der Pipeline aufgerufen.
    """
    directories = (
        DATA_DIR / "input",
        DATA_DIR / "indices",
        DATA_DIR / "output",
        DATA_DIR / "cache",
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_chroma_path(strategy: str) -> Path:
    """
    Ermittelt den Pfad zum Chroma Vector Store für eine Chunking-Strategie.

    Args:
        strategy: Die Chunking-Strategie (heuristic, semantic, agentic).

    Returns:
        Der absolute Pfad zum Chroma-Datenbankverzeichnis.
    """
    return CHROMA_PERSIST_ROOT / strategy / "chroma_db"


def get_bm25_path(strategy: str) -> Path:
    """
    Ermittelt den Pfad zur BM25-Indexdatei für eine Chunking-Strategie.

    Args:
        strategy: Die Chunking-Strategie (heuristic, semantic, agentic).

    Returns:
        Der absolute Pfad zur BM25-Pickle-Datei.
    """
    return CHROMA_PERSIST_ROOT / strategy / "inverted_index" / "bm25.pkl"


def get_chunk_ids_path(strategy: str) -> Path:
    """
    Ermittelt den Pfad zur Chunk-ID-Mapping-Datei für eine Chunking-Strategie.

    Args:
        strategy: Die Chunking-Strategie (heuristic, semantic, agentic).

    Returns:
        Der absolute Pfad zur Chunk-IDs-Pickle-Datei.
    """
    return CHROMA_PERSIST_ROOT / strategy / "inverted_index" / "chunk_ids.pkl"


def get_stats_path(strategy: str) -> Path:
    """
    Ermittelt den Pfad zur Statistikdatei für eine Chunking-Strategie.

    Args:
        strategy: Die Chunking-Strategie (heuristic, semantic, agentic).

    Returns:
        Der absolute Pfad zur JSON-Statistikdatei.
    """
    return CHROMA_PERSIST_ROOT / strategy / "indexing_stats.json"
