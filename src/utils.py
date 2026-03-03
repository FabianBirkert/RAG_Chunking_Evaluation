"""
Hilfsfunktionen und Utilities für die RAG-Pipeline.

Dieses Modul stellt gemeinsam genutzte Funktionalitäten für alle Schritte
der Retrieval-Augmented Generation Pipeline bereit. Es kapselt die
Kommunikation mit externen APIs, Tokenisierung, Dateisystemoperationen
und asynchrone Verarbeitungslogik.

Die Kernkomponenten umfassen:
    - ClientManager: Singleton für API-Client-Verwaltung
    - Asynchrone LLM-Aufrufe mit Retry-Logik und Rate-Limiting
    - Embedding-Funktionen für Vektorrepräsentationen
    - Tokenisierungswerkzeuge für BM25 und Token-Zählung
    - Pydantic-Modelle für strukturierte LLM-Ausgaben
"""

import os
import re
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from functools import partial

import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, RateLimitError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

load_dotenv()

logger = logging.getLogger("RAG-Pipeline.Utils")

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

_RE_JSON_MARKDOWN_BLOCK = re.compile(r'^```(?:json|JSON)?\s*\n?(.*?)\n?```$', re.DOTALL)
_RE_MULTI_SPACES = re.compile(r' +')
_RE_MULTI_NEWLINES = re.compile(r'\n{3,}')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MAX_CONCURRENT_REQUESTS,
    MAX_CONCURRENT_AGENTIC,
    MAX_CONCURRENT_EVALUATION,
    MAX_CONCURRENT_RAGAS,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    TIKTOKEN_ENCODING,
    LLM_MODEL,
    TEMPERATURE,
    EMBEDDING_MODEL,
    USE_OPENROUTER,
    OPENROUTER_BASE_URL,
)


class ClientManager:
    """
    Singleton-Manager für API-Clients und gemeinsam genutzte Ressourcen.

    Diese Klasse verwaltet die Initialisierung und den Zugriff auf API-Clients
    für OpenAI und OpenRouter sowie auf Semaphore für die Parallelitätssteuerung.
    Das Singleton-Pattern stellt sicher, dass nur eine Instanz existiert.

    Attributes:
        async_client: Asynchroner OpenAI-Client für LLM-Aufrufe.
        sync_client: Synchroner OpenAI-Client für Embedding-Aufrufe.
        tiktoken_encoder: Tokenizer für die Token-Zählung.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClientManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._async_client: Optional[AsyncOpenAI] = None
        self._sync_client: Optional[OpenAI] = None
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._semaphore_loop: Optional[asyncio.AbstractEventLoop] = None
        self._tiktoken_encoder = None
        self._initialized = True

    @property
    def async_client(self) -> AsyncOpenAI:
        """Gibt den asynchronen OpenAI-Client zurück (Lazy Initialization)."""
        if self._async_client is None:
            if USE_OPENROUTER:
                self._async_client = AsyncOpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    timeout=REQUEST_TIMEOUT,
                )
                logger.info("Using OpenRouter for LLM calls")
            else:
                self._async_client = AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    timeout=REQUEST_TIMEOUT,
                )
        return self._async_client

    @property
    def sync_client(self) -> OpenAI:
        """Gibt den synchronen OpenAI-Client zurück (Lazy Initialization)."""
        if self._sync_client is None:
            if USE_OPENROUTER:
                self._sync_client = OpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    timeout=REQUEST_TIMEOUT,
                )
            else:
                self._sync_client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    timeout=REQUEST_TIMEOUT,
                )
        return self._sync_client

    @property
    def tiktoken_encoder(self):
        """Gibt den tiktoken-Encoder zurück (Lazy Initialization)."""
        if self._tiktoken_encoder is None:
            self._tiktoken_encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)
        return self._tiktoken_encoder

    def get_semaphore(self, context: str = "default") -> asyncio.Semaphore:
        """
        Gibt einen Semaphore für den angegebenen Verarbeitungskontext zurück.

        Args:
            context: Der Kontext für die Parallelitätssteuerung. Mögliche Werte
                sind "agentic", "evaluation", "ragas" oder "default".

        Returns:
            Ein asyncio.Semaphore mit der für den Kontext konfigurierten
            maximalen Anzahl gleichzeitiger Anfragen.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        
        if self._semaphore_loop is not current_loop:
            self._semaphores = {}
            self._semaphore_loop = current_loop
        
        if context not in self._semaphores:
            limits = {
                "agentic": MAX_CONCURRENT_AGENTIC,
                "evaluation": MAX_CONCURRENT_EVALUATION,
                "ragas": MAX_CONCURRENT_RAGAS,
                "default": MAX_CONCURRENT_REQUESTS,
            }
            limit = limits.get(context, MAX_CONCURRENT_REQUESTS)
            self._semaphores[context] = asyncio.Semaphore(limit)
        
        return self._semaphores[context]


_client_manager = ClientManager()


def get_async_client() -> AsyncOpenAI:
    """
    Gibt den globalen asynchronen OpenAI-Client zurück.

    Returns:
        Die Singleton-Instanz des AsyncOpenAI-Clients.
    """
    return _client_manager.async_client


def get_sync_client() -> OpenAI:
    """
    Gibt den globalen synchronen OpenAI-Client zurück.

    Returns:
        Die Singleton-Instanz des synchronen OpenAI-Clients.
    """
    return _client_manager.sync_client


def get_semaphore(context: str = "default") -> asyncio.Semaphore:
    """
    Gibt den Semaphore für einen Verarbeitungskontext zurück.

    Args:
        context: Der Kontext für die Parallelitätssteuerung.

    Returns:
        Der zugehörige asyncio.Semaphore.
    """
    return _client_manager.get_semaphore(context)


def get_tiktoken_encoder():
    """
    Gibt den tiktoken-Encoder für die Token-Zählung zurück.

    Returns:
        Die Singleton-Instanz des tiktoken-Encoders.
    """
    return _client_manager.tiktoken_encoder


def ensure_parent_directory(path: str) -> None:
    """
    Stellt sicher, dass das Elternverzeichnis eines Pfades existiert.

    Args:
        path: Der Dateipfad, dessen Elternverzeichnis erstellt werden soll.
    """
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def ensure_directory(path: str) -> None:
    """
    Stellt sicher, dass ein Verzeichnis existiert.

    Args:
        path: Der Verzeichnispfad, der erstellt werden soll.
    """
    os.makedirs(path, exist_ok=True)


def strip_json_markdown(text: str) -> str:
    """
    Entfernt Markdown-Code-Blöcke aus LLM-Ausgaben.

    Diese Funktion bereitet LLM-Antworten für das JSON-Parsing vor, indem
    sie typische Markdown-Formatierungen wie Code-Blöcke entfernt.

    Args:
        text: Der zu bereinigende Text mit möglicher Markdown-Formatierung.

    Returns:
        Der bereinigte Text ohne Markdown-Wrapper.
    """
    text = text.strip()
    
    match = _RE_JSON_MARKDOWN_BLOCK.match(text)
    if match:
        return match.group(1).strip()
    
    if text.startswith('```'):
        lines = text.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        return '\n'.join(lines).strip()
    
    return text


def safe_json_parse(text: str, fallback: Any = None) -> Any:
    """
    Parst JSON-Text sicher mit automatischer Markdown-Bereinigung.

    Args:
        text: Der zu parsende Text.
        fallback: Rückgabewert bei Parsing-Fehler.

    Returns:
        Das geparste JSON-Objekt oder der Fallback-Wert.
    """
    try:
        cleaned = strip_json_markdown(text)
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON parsing failed: {e}")
        return fallback


def count_tokens(text: str) -> int:
    """
    Zählt die Anzahl der Tokens in einem Text.

    Verwendet den tiktoken-Encoder cl100k_base, der für GPT-4 und
    GPT-3.5-turbo Modelle verwendet wird.

    Args:
        text: Der zu tokenisierende Text.

    Returns:
        Die Anzahl der Tokens im Text.
    """
    encoder = get_tiktoken_encoder()
    return len(encoder.encode(text))


def encode_text(text: str) -> List[int]:
    """
    Encodiert Text zu einer Liste von Token-IDs.

    Args:
        text: Der zu encodierende Text.

    Returns:
        Eine Liste von Integer-Token-IDs.
    """
    encoder = get_tiktoken_encoder()
    return encoder.encode(text)


def decode_tokens(tokens: List[int]) -> str:
    """
    Decodiert eine Liste von Token-IDs zurück zu Text.

    Args:
        tokens: Liste von Token-IDs.

    Returns:
        Der decodierte Textstring.
    """
    encoder = get_tiktoken_encoder()
    return encoder.decode(tokens)


def _log_rate_limit(sleep_time):
    """Protokolliert Wartezeiten bei Rate-Limit-Überschreitungen."""
    logger.warning(f"Rate limit or timeout reached. Retrying in {sleep_time:.2f} seconds...")


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    before_sleep=lambda retry_state: _log_rate_limit(retry_state.next_action.sleep),
)
async def async_llm_call(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = None,
    response_format: Optional[Dict] = None,
    context: str = "default",
) -> Dict[str, Any]:
    """
    Führt einen asynchronen LLM-Aufruf mit automatischer Retry-Logik durch.

    Diese Funktion kapselt die Kommunikation mit der OpenAI-API und
    implementiert Rate-Limiting durch Semaphore sowie automatische
    Wiederholungsversuche bei transienten Fehlern.

    Args:
        messages: Liste von Message-Dictionaries mit 'role' und 'content'.
        model: Das zu verwendende LLM-Modell.
        temperature: Sampling-Temperatur für die Generierung.
        response_format: Optionale Spezifikation des Antwortformats.
        context: Kontext für die Semaphore-Auswahl.

    Returns:
        Ein Dictionary mit den Schlüsseln 'content', 'input_tokens',
        'output_tokens' und 'total_tokens'.
    """
    model = model or LLM_MODEL
    temperature = temperature if temperature is not None else TEMPERATURE
    
    client = get_async_client()
    semaphore = get_semaphore(context)
    
    async with semaphore:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format
            
        response = await client.chat.completions.create(**kwargs)
        
        return {
            "content": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }


def sync_llm_call(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = None,
    response_format: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Führt einen synchronen LLM-Aufruf durch.

    Diese Funktion ist für Kontexte gedacht, in denen asynchrone Aufrufe
    nicht möglich oder nicht erwünscht sind.

    Args:
        messages: Liste von Message-Dictionaries mit 'role' und 'content'.
        model: Das zu verwendende LLM-Modell.
        temperature: Sampling-Temperatur für die Generierung.
        response_format: Optionale Spezifikation des Antwortformats.

    Returns:
        Ein Dictionary mit den Schlüsseln 'content', 'input_tokens',
        'output_tokens' und 'total_tokens'.
    """
    model = model or LLM_MODEL
    temperature = temperature if temperature is not None else TEMPERATURE
    
    client = get_sync_client()
    
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        kwargs["response_format"] = response_format
        
    response = client.chat.completions.create(**kwargs)
    
    return {
        "content": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
)
async def async_get_embeddings(
    texts: List[str],
    model: str = None,
) -> List[List[float]]:
    """
    Generiert Embeddings für eine Liste von Texten asynchron.

    Diese Funktion ruft die OpenAI Embeddings-API auf und garantiert,
    dass die Reihenfolge der Ergebnisvektoren der Eingabereihenfolge entspricht.

    Args:
        texts: Liste von Texten, für die Embeddings generiert werden sollen.
        model: Das zu verwendende Embedding-Modell.

    Returns:
        Eine Liste von Embedding-Vektoren (Float-Listen) in Eingabereihenfolge.
    """
    model = model or EMBEDDING_MODEL
    client = get_async_client()
    semaphore = get_semaphore()
    
    async with semaphore:
        response = await client.embeddings.create(
            model=model,
            input=texts,
        )
        
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [e.embedding for e in embeddings]


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    before_sleep=lambda retry_state: _log_rate_limit(retry_state.next_action.sleep),
)
def sync_get_embeddings(
    texts: List[str],
    model: str = None,
) -> List[List[float]]:
    """
    Generiert Embeddings für eine Liste von Texten synchron.

    Diese Funktion ruft die OpenAI Embeddings-API auf und garantiert,
    dass die Reihenfolge der Ergebnisvektoren der Eingabereihenfolge entspricht.

    Args:
        texts: Liste von Texten, für die Embeddings generiert werden sollen.
        model: Das zu verwendende Embedding-Modell.

    Returns:
        Eine Liste von Embedding-Vektoren (Float-Listen) in Eingabereihenfolge.
    """
    model = model or EMBEDDING_MODEL
    client = get_sync_client()
    
    response = client.embeddings.create(
        model=model,
        input=texts,
    )
    
    embeddings = sorted(response.data, key=lambda x: x.index)
    return [e.embedding for e in embeddings]


async def async_get_single_embedding(text: str, model: str = None) -> List[float]:
    """
    Generiert ein Embedding für einen einzelnen Text asynchron.

    Args:
        text: Der Text, für den ein Embedding generiert werden soll.
        model: Das zu verwendende Embedding-Modell.

    Returns:
        Der Embedding-Vektor als Liste von Floats.
    """
    embeddings = await async_get_embeddings([text], model)
    return embeddings[0]


async def run_sync_in_executor(func, *args, **kwargs):
    """
    Führt eine synchrone Funktion in einem Thread-Pool-Executor aus.

    Diese Funktion ermöglicht die nicht-blockierende Ausführung von
    synchronen Operationen (z.B. Dateisystem-I/O, Datenbankzugriffe)
    innerhalb eines asynchronen Kontexts.

    Args:
        func: Die auszuführende synchrone Funktion.
        *args: Positionale Argumente für die Funktion.
        **kwargs: Schlüsselwortargumente für die Funktion.

    Returns:
        Das Rückgabeergebnis der ausgeführten Funktion.
    """
    loop = asyncio.get_event_loop()
    if kwargs:
        func = partial(func, **kwargs)
    return await loop.run_in_executor(None, func, *args)


def normalize_whitespace(text: str) -> str:
    """
    Normalisiert Whitespace-Zeichen in einem Text.

    Ersetzt mehrfache aufeinanderfolgende Leerzeichen durch einzelne
    und reduziert mehr als zwei aufeinanderfolgende Zeilenumbrüche
    auf maximal zwei.

    Args:
        text: Der zu normalisierende Text.

    Returns:
        Der Text mit normalisiertem Whitespace.
    """
    text = _RE_MULTI_SPACES.sub(' ', text)
    text = _RE_MULTI_NEWLINES.sub('\n\n', text)
    return text.strip()


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenisiert einen Text für die BM25-Indexierung.

    Verwendet eine einfache Whitespace-Tokenisierung mit
    Konvertierung in Kleinbuchstaben.

    Args:
        text: Der zu tokenisierende Text.

    Returns:
        Eine Liste von Token in Kleinbuchstaben.
    """
    return text.lower().split()


def generate_chunk_id(strategy: str, source_id: str, chunk_index: int) -> str:
    """
    Generiert eine eindeutige Chunk-ID.

    Das ID-Format folgt dem Schema {strategy}:{source_id}:{chunk_index}
    und ermöglicht die eindeutige Identifikation eines Chunks über
    Strategie, Quelldokument und Position hinweg.

    Args:
        strategy: Die verwendete Chunking-Strategie.
        source_id: Die ID des Quelldokuments.
        chunk_index: Der fortlaufende Index des Chunks im Dokument.

    Returns:
        Eine eindeutige Chunk-ID als String.
    """
    return f"{strategy}:{source_id}:{chunk_index}"


from pydantic import BaseModel, Field
from typing import List


class Proposition(BaseModel):
    """
    Repräsentiert eine einzelne atomare Proposition.

    Attributes:
        statement: Der Textinhalt der Proposition.
    """
    statement: str


class PropositionOutput(BaseModel):
    """
    Strukturiertes Ausgabeformat für die Proposition-Extraktion.

    Attributes:
        propositions: Liste der extrahierten Propositionen.
    """
    propositions: List[Proposition]


class GroupItem(BaseModel):
    """
    Repräsentiert eine Gruppe von semantisch zusammenhängenden Propositionen.

    Attributes:
        group_indices: Indizes der Propositionen in dieser Gruppe.
        title: Beschreibender Titel der Gruppe.
    """
    group_indices: List[int]
    title: str


class GroupingOutput(BaseModel):
    """
    Strukturiertes Ausgabeformat für die Proposition-Gruppierung.

    Attributes:
        groups: Liste der gebildeten Gruppen.
    """
    groups: List[GroupItem]


def get_proposition_format_instructions() -> str:
    """
    Gibt die JSON-Formatierungsanweisungen für die Proposition-Extraktion zurück.

    Returns:
        Ein String mit den Formatierungsanweisungen für das LLM.
    """
    return '''Return your response as a JSON object with the following structure:
{
  "propositions": [
    {"statement": "First proposition..."},
    {"statement": "Second proposition..."}
  ]
}'''


def get_grouping_format_instructions() -> str:
    """
    Gibt die JSON-Formatierungsanweisungen für die Proposition-Gruppierung zurück.

    Returns:
        Ein String mit den Formatierungsanweisungen für das LLM.
    """
    return '''Return your response as a JSON object with the following structure:
{
  "groups": [
    {"group_indices": [0, 2, 3], "title": "Topic title 1"},
    {"group_indices": [1, 4], "title": "Topic title 2"}
  ]
}'''
