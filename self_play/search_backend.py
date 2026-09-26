from __future__ import annotations

import ast
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
import yaml


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

SEARCH_TAG_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
GET_ITEM_DETAILS_TAG_RE = re.compile(r"<get_item_details>(.*?)</get_item_details>", re.DOTALL)
GET_USER_PREFERENCE_TAG_RE = re.compile(r"<get_user_preference>(.*?)</get_user_preference>", re.DOTALL)
SENTENCE_SPLIT_RE = re.compile(r"[.!?;\n]+")
MULTI_SPACE_RE = re.compile(r"\s+")
DEFAULT_PROFILE_FILES: tuple[str, ...] = ()
COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "good",
    "great",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "like",
    "looking",
    "my",
    "need",
    "of",
    "on",
    "or",
    "product",
    "really",
    "so",
    "something",
    "that",
    "the",
    "their",
    "this",
    "to",
    "want",
    "with",
    "works",
}


def _clean_text(value: Any) -> str:
    text = str(value or "")
    text = MULTI_SPACE_RE.sub(" ", text.replace("\n", " "))
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    return [part.strip(" -,:") for part in SENTENCE_SPLIT_RE.split(_clean_text(text)) if len(part.strip()) >= 12]


def _compact_text(text: str, limit: int = 200) -> str:
    compact = _clean_text(text)
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def _normalized_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean_text(text).lower())


def _parse_tag_kwargs(action: str, pattern: re.Pattern[str]) -> tuple[dict[str, Any], bool]:
    try:
        match = pattern.search((action or "").strip())
        if not match:
            return {}, False
        inner = match.group(1).strip()
        if not inner:
            return {}, True
        kwargs = {}
        for kv in re.split(r",\s*(?=[\w_]+\s*=)", inner):
            if "=" not in kv:
                continue
            key, val = kv.split("=", 1)
            key, val = key.strip(), val.strip()
            try:
                kwargs[key] = ast.literal_eval(val)
            except Exception:
                kwargs[key] = val.strip("'\"")
        return kwargs, True
    except Exception:
        return {}, False


def _load_search_config() -> dict[str, Any]:
    cfg_path = Path(__file__).resolve().parent / "configs" / "search.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg = cfg or {}
    cfg.setdefault("search", {})
    search_cfg = cfg["search"]
    if os.environ.get("SEARCH_INDEX_DIR"):
        search_cfg["index_dir"] = os.environ["SEARCH_INDEX_DIR"]
    if os.environ.get("SEARCH_MODEL_PATH"):
        search_cfg["model_path"] = os.environ["SEARCH_MODEL_PATH"]
    if os.environ.get("SEARCH_DEVICE"):
        search_cfg["device"] = os.environ["SEARCH_DEVICE"]
    if os.environ.get("SEARCH_ENCODER_BACKEND"):
        search_cfg["encoder_backend"] = os.environ["SEARCH_ENCODER_BACKEND"]
    if os.environ.get("SEARCH_QUERY_INSTRUCTION"):
        search_cfg["query_instruction"] = os.environ["SEARCH_QUERY_INSTRUCTION"]
    if os.environ.get("SEARCH_TORCH_DTYPE"):
        search_cfg["torch_dtype"] = os.environ["SEARCH_TORCH_DTYPE"]
    if os.environ.get("SEARCH_LOW_CPU_MEM_USAGE"):
        search_cfg["low_cpu_mem_usage"] = os.environ["SEARCH_LOW_CPU_MEM_USAGE"]
    if os.environ.get("SEARCH_DEVICE_MAP"):
        search_cfg["device_map"] = os.environ["SEARCH_DEVICE_MAP"]
    if os.environ.get("SEARCH_MAX_LENGTH"):
        search_cfg["max_length"] = os.environ["SEARCH_MAX_LENGTH"]
    if os.environ.get("SEARCH_BM25_INDEX_DIR"):
        search_cfg["bm25_index_dir"] = os.environ["SEARCH_BM25_INDEX_DIR"]
    if os.environ.get("SEARCH_BM25_FIELD"):
        search_cfg["bm25_field"] = os.environ["SEARCH_BM25_FIELD"]
    if os.environ.get("SEARCH_BM25_K1"):
        search_cfg["bm25_k1"] = os.environ["SEARCH_BM25_K1"]
    if os.environ.get("SEARCH_BM25_B"):
        search_cfg["bm25_b"] = os.environ["SEARCH_BM25_B"]
    if os.environ.get("SEARCH_BM25_ESCAPE_LUCENE_SPECIALS"):
        search_cfg["bm25_escape_lucene_specials"] = os.environ["SEARCH_BM25_ESCAPE_LUCENE_SPECIALS"]
    return search_cfg


SEARCH_CONFIG = _load_search_config()


def _should_retry_on_cpu(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in ("no cuda", "cuda", "nvidia", "driver", "device"))


def _resolve_search_device(device: str | None) -> str:
    requested = _clean_text(device) or "cuda:0"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _torch_dtype(value: Any):
    text = _clean_text(value)
    if not text or text == "auto":
        return "auto"
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(text.lower(), text)


def _first_floating_param_dtype(model: torch.nn.Module) -> torch.dtype | None:
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return None


def _has_meta_tensor(model: torch.nn.Module) -> bool:
    return any(param.is_meta for param in model.parameters()) or any(buffer.is_meta for buffer in model.buffers())


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _configured_search_modes() -> set[str]:
    modes = {
        os.environ.get("SIMREC_SEARCH_MODE", "hybrid"),
        os.environ.get("SIMREC_RECALL_EXPANSION_MODE", "hybrid"),
        os.environ.get("SIMREC_FINAL_RETRIEVAL_MODE", "hybrid"),
    }
    return {str(mode or "").strip().lower() for mode in modes if str(mode or "").strip()}


def _format_lucene_query(query: str, field: str, *, escape_lucene_specials: bool = False) -> str:
    query = _clean_text(query)
    if escape_lucene_specials:
        query = re.sub(r"([+\-&|!(){}\[\]^\"~*?:\\/])", " ", query)
        query = _clean_text(query)
    return f"({field}:{query})" if field else query


def _safe_load_pyserini_doc(hit: Any) -> dict[str, Any]:
    raw = getattr(hit, "raw", None)
    if isinstance(raw, str):
        try:
            doc = json.loads(raw)
        except Exception:
            return {}
        return doc if isinstance(doc, dict) else {}
    if isinstance(raw, dict):
        return raw
    lucene_document = getattr(hit, "lucene_document", None)
    if lucene_document is not None:
        doc = {}
        for field in lucene_document.getFields():
            doc[field.name()] = field.stringValue()
        return doc
    return {}


def _doc_id(doc: dict[str, Any]) -> str:
    return _clean_text(
        doc.get("id")
        or doc.get("item_id")
        or doc.get("parent_asin")
        or doc.get("asin")
        or doc.get("product_id")
    ).upper()


def _doc_content(doc: dict[str, Any]) -> str:
    return _clean_text(doc.get("contents") or doc.get("content") or doc.get("metadata") or doc.get("title") or doc.get("name"))


class UniversalSearcher:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, index_dir: str, model_path: str, device: str = "cuda:0"):
        if self._initialized:
            return
        self.index_dir = index_dir
        self.encoder_backend = str(SEARCH_CONFIG.get("encoder_backend", "flag") or "flag").lower()
        self.query_instruction = str(
            SEARCH_CONFIG.get("query_instruction", "Represent this sentence for searching relevant products:") or ""
        )
        self.torch_dtype = str(SEARCH_CONFIG.get("torch_dtype", "auto") or "auto")
        self.device_map = SEARCH_CONFIG.get("device_map")
        self.low_cpu_mem_usage = str(SEARCH_CONFIG.get("low_cpu_mem_usage", "false")).lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.max_length = int(SEARCH_CONFIG.get("max_length", 512) or 512)
        self.tokenizer = None
        self.bm25_index_dir = str(SEARCH_CONFIG.get("bm25_index_dir", "") or "")
        self.bm25_field = str(SEARCH_CONFIG.get("bm25_field", "contents") or "contents")
        self.bm25_k1 = float(SEARCH_CONFIG.get("bm25_k1", 1.2) or 1.2)
        self.bm25_b = float(SEARCH_CONFIG.get("bm25_b", 0.75) or 0.75)
        self.bm25_escape_lucene_specials = _as_bool(SEARCH_CONFIG.get("bm25_escape_lucene_specials"), default=False)
        self._bm25_searcher = None
        requested_device = _resolve_search_device(device)
        self.device = requested_device
        self.bm25_only = _configured_search_modes() <= {"bm25"}
        self.model = None
        self.dense_index = None
        if not self.bm25_only:
            self.model = self._load_encoder(model_path)
            self.dense_index = faiss.read_index(os.path.join(index_dir, "vector.index"))
        if self.dense_index is not None and hasattr(self.dense_index, "nprobe"):
            self.dense_index.nprobe = 16

        with open(os.path.join(index_dir, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        self.metadata_by_id = {}
        for item in self.metadata:
            if isinstance(item, dict):
                item_id = _doc_id(item)
                if item_id and item_id not in self.metadata_by_id:
                    self.metadata_by_id[item_id] = item

        sparse_weights_path = os.path.join(index_dir, "sparse_weights.pkl")
        self.sparse_weights_list = None
        if os.path.exists(sparse_weights_path):
            with open(sparse_weights_path, "rb") as f:
                self.sparse_weights_list = pickle.load(f)

        inverted_path = os.path.join(index_dir, "inverted_index.pkl")
        if os.path.exists(inverted_path):
            with open(inverted_path, "rb") as f:
                self.inverted_index = pickle.load(f)
        elif self.sparse_weights_list is not None:
            self.inverted_index = defaultdict(list)
            for doc_id, weights in enumerate(self.sparse_weights_list):
                if not weights:
                    continue
                for token_id, score in weights.items():
                    self.inverted_index[int(token_id)].append((doc_id, float(score)))
        else:
            self.inverted_index = defaultdict(list)
        self._initialized = True

    def _load_encoder(self, model_path: str):
        if self.encoder_backend == "flag":
            from FlagEmbedding import FlagAutoModel

            try:
                return FlagAutoModel.from_finetuned(
                    model_path,
                    query_instruction_for_retrieval=self.query_instruction or None,
                    use_fp16=False,
                    devices=self.device,
                    local_files_only=True,
                )
            except Exception as exc:
                if self.device == "cpu" or not _should_retry_on_cpu(exc):
                    raise
                self.device = "cpu"
                return FlagAutoModel.from_finetuned(
                    model_path,
                    query_instruction_for_retrieval=self.query_instruction or None,
                    use_fp16=False,
                    devices="cpu",
                    local_files_only=True,
                )
        if self.encoder_backend == "transformers":
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            target_dtype = _torch_dtype(self.torch_dtype)
            model_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": True,
                "low_cpu_mem_usage": self.low_cpu_mem_usage,
            }
            model_kwargs["torch_dtype"] = target_dtype if target_dtype != "auto" else "auto"
            if self.device_map:
                model_kwargs["device_map"] = self.device_map
            model = AutoModel.from_pretrained(model_path, **model_kwargs)
            target_device = torch.device(self.device if self.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
            if not self.device_map:
                if _has_meta_tensor(model):
                    raise RuntimeError(
                        "Embedding model still contains meta tensors after from_pretrained. "
                        "Set SEARCH_LOW_CPU_MEM_USAGE=False and avoid SEARCH_DEVICE_MAP for reward search."
                    )
                if isinstance(target_dtype, torch.dtype):
                    model.to(device=target_device, dtype=target_dtype)
                else:
                    model.to(target_device)
            elif isinstance(target_dtype, torch.dtype):
                model.to(dtype=target_dtype)
            model.eval()
            return model
        raise ValueError(f"Unknown SEARCH_ENCODER_BACKEND={self.encoder_backend!r}. Use 'flag' or 'transformers'.")

    def _format_query(self, query: str) -> str:
        query = _clean_text(query)
        if self.encoder_backend == "flag" or not self.query_instruction:
            return query
        return self.query_instruction.format(query=query) if "{query}" in self.query_instruction else f"{self.query_instruction}{query}"

    @staticmethod
    def _last_token_pool(hidden_states, attention_mask):
        sequence_lengths = (attention_mask.sum(dim=1) - 1).to(hidden_states.device)
        batch_size = hidden_states.shape[0]
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

    def _encode_dense_query(self, query: str) -> np.ndarray:
        if self.model is None:
            raise ValueError("Dense search is unavailable because this run is configured for BM25-only search.")
        if self.encoder_backend == "flag":
            q_output = self.model.encode_queries([query], return_dense=True)
            return np.ascontiguousarray(q_output["dense_vecs"], dtype=np.float32)

        import torch.nn.functional as F

        assert self.tokenizer is not None
        encoded = self.tokenizer(
            [self._format_query(query)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_device = next(self.model.parameters()).device
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        model_dtype = _first_floating_param_dtype(self.model)
        with torch.no_grad():
            output = self.model(**encoded)
            embeddings = self._last_token_pool(output.last_hidden_state, encoded["attention_mask"])
            if model_dtype is not None and embeddings.dtype != model_dtype:
                embeddings = embeddings.to(model_dtype)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return np.ascontiguousarray(embeddings.detach().cpu().float().numpy(), dtype=np.float32)

    def get_item_record(self, item_id: str) -> dict[str, Any] | None:
        return self.metadata_by_id.get(_clean_text(item_id).upper())

    def search_dense(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if self.dense_index is None:
            raise ValueError("Dense search is unavailable because this run is configured for BM25-only search.")
        q_dense = self._encode_dense_query(query)
        faiss.normalize_L2(q_dense)
        scores, indices = self.dense_index.search(q_dense, top_k)
        results = []
        for j, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "id": self.metadata[idx]["id"],
                    "content": self.metadata[idx]["contents"],
                    "score": float(scores[0][j]),
                    "type": "dense",
                }
            )
        return results

    def _get_bm25_searcher(self):
        if self._bm25_searcher is None:
            if not self.bm25_index_dir:
                raise ValueError("BM25 index directory is not configured. Set SEARCH_BM25_INDEX_DIR.")
            from pyserini.search.lucene import LuceneSearcher

            searcher = LuceneSearcher(self.bm25_index_dir)
            searcher.set_bm25(self.bm25_k1, self.bm25_b)
            self._bm25_searcher = searcher
        return self._bm25_searcher

    def search_bm25(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        lucene_query = _format_lucene_query(
            query,
            self.bm25_field,
            escape_lucene_specials=self.bm25_escape_lucene_specials,
        )
        hits = self._get_bm25_searcher().search(lucene_query, k=top_k)
        results = []
        for hit in hits:
            doc = _safe_load_pyserini_doc(hit)
            item_id = _doc_id(doc)
            if not item_id:
                continue
            results.append(
                {
                    "id": item_id,
                    "content": _doc_content(doc),
                    "score": float(getattr(hit, "score", 0.0) or 0.0),
                    "type": "bm25",
                }
            )
        return results

    def search_sparse(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if self.encoder_backend != "flag" or self.sparse_weights_list is None:
            raise ValueError("Sparse search requires a FlagEmbedding index with sparse_weights.pkl.")
        q_output = self.model.encode_queries([query], return_sparse=True, return_dense=False)
        q_sparse = q_output["lexical_weights"][0]
        doc_scores = defaultdict(float)
        for token_id, q_score in q_sparse.items():
            token_id = int(token_id)
            for doc_id, d_score in self.inverted_index.get(token_id, []):
                doc_scores[doc_id] += q_score * d_score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "id": self.metadata[doc_id]["id"],
                "content": self.metadata[doc_id]["contents"],
                "score": float(score),
                "type": "sparse",
            }
            for doc_id, score in sorted_docs
        ]

    def search_hybrid(self, query: str, top_k: int = 10, alpha: float = 0.3) -> list[dict[str, Any]]:
        if self.encoder_backend != "flag" or self.sparse_weights_list is None:
            return self.search_dense(query, top_k)
        q_output = self.model.encode_queries([query], return_dense=True, return_sparse=True)
        q_dense = np.ascontiguousarray(q_output["dense_vecs"], dtype=np.float32)
        faiss.normalize_L2(q_dense)
        q_sparse = q_output["lexical_weights"][0]

        candidate_k = min(100, self.dense_index.ntotal)
        d_scores, indices = self.dense_index.search(q_dense, candidate_k)
        candidates = []
        for j, doc_idx in enumerate(indices[0]):
            if doc_idx == -1:
                continue
            d_score = float(d_scores[0][j])
            doc_sparse_vec = self.sparse_weights_list[doc_idx]
            s_score = 0.0
            for token_id, q_val in q_sparse.items():
                if token_id in doc_sparse_vec:
                    s_score += q_val * doc_sparse_vec[token_id]
            candidates.append(
                {
                    "id": self.metadata[doc_idx]["id"],
                    "content": self.metadata[doc_idx]["contents"],
                    "score": d_score + alpha * s_score,
                    "type": "hybrid",
                }
            )
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]


class SearchBackend:
    def __init__(self, index_dir: str | None = None, model_path: str | None = None, device: str | None = None):
        index_dir = index_dir or SEARCH_CONFIG.get("index_dir", "")
        model_path = model_path or SEARCH_CONFIG.get("model_path", "")
        device = _resolve_search_device(device or SEARCH_CONFIG.get("device", "cuda:0"))
        self.searcher = UniversalSearcher(index_dir=index_dir, model_path=model_path, device=device)

    @staticmethod
    def parse_search_action(action: str) -> tuple[dict[str, Any], bool]:
        kwargs, ok = _parse_tag_kwargs(action, SEARCH_TAG_RE)
        return kwargs, bool(ok and kwargs.get("query"))

    @staticmethod
    def parse_get_item_details_action(action: str) -> tuple[dict[str, Any], bool]:
        kwargs, ok = _parse_tag_kwargs(action, GET_ITEM_DETAILS_TAG_RE)
        item_id = _clean_text(kwargs.get("item_id")).upper()
        if not ok or not item_id:
            return {}, False
        kwargs["item_id"] = item_id
        return kwargs, True

    def search_records(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[dict[str, Any]]:
        mode = (mode or "hybrid").lower()
        if mode == "bm25":
            return self.searcher.search_bm25(query, top_k)
        if mode == "dense":
            return self.searcher.search_dense(query, top_k)
        if mode == "sparse":
            return self.searcher.search_sparse(query, top_k)
        if mode == "hybrid":
            return self.searcher.search_hybrid(query, top_k)
        raise ValueError(f"Unknown mode '{mode}'. Use 'bm25', 'dense', 'sparse', or 'hybrid'.")

    @staticmethod
    def render_search_results(results: list[dict[str, Any]]) -> str:
        if not results:
            return "No relevant products found."

        lines = []
        for idx, item in enumerate(results):
            content = _compact_text(item.get("content", ""), limit=300)
            lines.append(
                f"[{idx + 1}] ID: {item['id']} | Mode: {item['type']} | Score: {item['score']:.4f} | Content: {content}"
            )
        return "\n".join(lines)

    @staticmethod
    def _summarize_item_record(item_id: str, record: dict[str, Any]) -> str:
        title = _clean_text(record.get("title") or record.get("name") or record.get("product_title"))
        brand = _clean_text(record.get("brand") or record.get("manufacturer"))
        category = _clean_text(record.get("category") or record.get("categories"))
        content = _clean_text(record.get("contents") or record.get("content") or "")
        sentences = _split_sentences(content)
        summary = _compact_text(sentences[0] if sentences else content, limit=220) or "No summary available."
        details = sentences[1:4] if len(sentences) > 1 else []
        if not title:
            title = summary

        title_key = _normalized_text_key(title)
        summary_key = _normalized_text_key(summary)
        compact_content = _compact_text(content, limit=320)
        content_key = _normalized_text_key(compact_content)

        unique_details = []
        seen_keys = {key for key in (title_key, summary_key) if key}
        for sentence in details:
            compact_sentence = _compact_text(sentence, limit=180)
            sentence_key = _normalized_text_key(compact_sentence)
            if not sentence_key or sentence_key in seen_keys:
                continue
            seen_keys.add(sentence_key)
            unique_details.append(compact_sentence)

        lines = [
            "Item details:",
            f"ID: {item_id}",
            f"Title: {title}",
        ]
        if brand:
            lines.append(f"Brand: {brand}")
        if category:
            lines.append(f"Category: {category}")
        if summary_key and summary_key != title_key:
            lines.append(f"Summary: {summary}")
        if unique_details:
            lines.append("Key attributes:")
            lines.extend(f"- {sentence}" for sentence in unique_details[:3])
        elif content_key and content_key not in {title_key, summary_key}:
            lines.append(f"Content: {compact_content}")
        return "\n".join(lines)

    def get_item_details(self, item_id: str) -> tuple[str, bool]:
        record = self.searcher.get_item_record(item_id)
        if record is None:
            return (
                "Item details:\n"
                f"ID: {item_id}\n"
                "Status: not found in the local catalog.\n"
                "Try another item ID from the search results.",
                False,
            )
        return self._summarize_item_record(item_id, record), True

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> str:
        return self.render_search_results(self.search_records(query=query, top_k=top_k, mode=mode))


class UserPreferenceBackend:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, profile_path: str | None = None, data_files: list[str] | None = None):
        if self._initialized:
            return
        env_profile_path = os.environ.get("SIMREC_USER_PROFILE_PATH")
        self.profile_path = profile_path or env_profile_path
        self.data_files = [path for path in (data_files or self._discover_default_files()) if path]
        self.user_profiles = self._load_profiles()
        self._initialized = True

    @staticmethod
    def parse_get_user_preference_action(action: str) -> tuple[dict[str, Any], bool]:
        kwargs, ok = _parse_tag_kwargs(action, GET_USER_PREFERENCE_TAG_RE)
        if not ok:
            return {}, False
        history_k = kwargs.get("history_k", 5)
        try:
            history_k = max(1, int(history_k))
        except Exception:
            return {}, False
        return {"history_k": history_k}, True

    @staticmethod
    def _discover_default_files() -> list[str]:
        files = []
        for env_key in ("TRAIN_DATA", "VAL_DATA"):
            if value := os.environ.get(env_key):
                files.append(value)
        if files:
            return files
        return list(DEFAULT_PROFILE_FILES)

    @staticmethod
    def _normalize_history_entry(row: dict[str, Any]) -> dict[str, Any] | None:
        user_id = _clean_text(row.get("user_id"))
        if not user_id:
            return None

        if any(key in row for key in ("qid", "reference_query", "reference_review", "target_item_id")):
            return {
                "user_id": user_id,
                "qid": _clean_text(row.get("qid")),
                "item_id": _clean_text(row.get("target_item_id")).upper(),
                "query": _clean_text(row.get("reference_query")),
                "review_text": _clean_text(row.get("reference_review")),
                "title": "",
                "timestamp": row.get("timestamp"),
                "verified_purchase": row.get("verified_purchase"),
                "source_type": "SimRec.self_play",
            }

        if any(key in row for key in ("asin", "parent_asin", "text", "title")):
            item_id = _clean_text(row.get("parent_asin") or row.get("asin")).upper()
            return {
                "user_id": user_id,
                "qid": "",
                "item_id": item_id,
                "query": "",
                "review_text": _clean_text(row.get("text")),
                "title": _clean_text(row.get("title")),
                "timestamp": row.get("timestamp"),
                "verified_purchase": row.get("verified_purchase"),
                "source_type": "amazon_review",
            }

        if any(key in row for key in ("parent_asin", "review_text", "item_title")):
            item_id = _clean_text(row.get("parent_asin") or row.get("asin")).upper()
            return {
                "user_id": user_id,
                "qid": _clean_text(row.get("qid")),
                "item_id": item_id,
                "query": "",
                "review_text": _clean_text(row.get("review_text")),
                "title": _clean_text(row.get("item_title")),
                "timestamp": row.get("timestamp"),
                "verified_purchase": row.get("verified_purchase"),
                "source_type": "profile_history",
            }

        return None

    @staticmethod
    def _history_sort_key(item: dict[str, Any]) -> tuple[int, str, str]:
        timestamp = item.get("timestamp")
        try:
            timestamp_key = int(timestamp)
        except Exception:
            timestamp_key = -1
        return timestamp_key, _clean_text(item.get("qid")), _clean_text(item.get("item_id"))

    @staticmethod
    def _history_text(item: dict[str, Any]) -> str:
        return " ".join(
            part
            for part in (
                item.get("query", ""),
                item.get("review_text", ""),
                item.get("title", ""),
            )
            if part
        ).strip()

    def _load_profiles(self) -> dict[str, list[dict[str, Any]]]:
        if self.profile_path and Path(self.profile_path).exists():
            path = Path(self.profile_path)
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                profiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for user_id, value in data.items():
                    entries = value.get("purchase_history") if isinstance(value, dict) else value
                    if not isinstance(entries, list):
                        continue
                    for row in entries:
                        if not isinstance(row, dict):
                            continue
                        normalized = self._normalize_history_entry({"user_id": user_id, **row})
                        if normalized is not None:
                            profiles[user_id].append(normalized)
                return {uid: sorted(items, key=self._history_sort_key) for uid, items in profiles.items()}
            profiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    normalized = self._normalize_history_entry(row)
                    if normalized is not None:
                        profiles[normalized["user_id"]].append(normalized)
            return {uid: sorted(items, key=self._history_sort_key) for uid, items in profiles.items()}

        profiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for data_file in self.data_files:
            path = Path(data_file)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    normalized = self._normalize_history_entry(row)
                    if normalized is not None:
                        profiles[normalized["user_id"]].append(normalized)
        return {uid: sorted(items, key=self._history_sort_key) for uid, items in profiles.items()}

    def _derive_price_tendency(self, history: list[dict[str, Any]]) -> str:
        joined = " ".join(self._history_text(item) for item in history).lower()
        if any(token in joined for token in ("cheap", "cheapest", "budget", "affordable", "low price", "value")):
            return "Budget conscious"
        if any(token in joined for token in ("premium", "high quality", "best", "durable", "worth")):
            return "Quality focused"
        return "Flexible"

    def _derive_common_needs(self, history: list[dict[str, Any]], limit: int = 3) -> list[str]:
        candidates = []
        for item in history:
            for source in (item.get("query", ""), item.get("review_text", ""), item.get("title", "")):
                for sentence in _split_sentences(source):
                    if sentence not in candidates:
                        candidates.append(sentence)
                    if len(candidates) >= limit:
                        return candidates
        if not candidates:
            return ["No recurring needs extracted from history."]
        return candidates[:limit]

    def _derive_preferred_terms(self, history: list[dict[str, Any]], limit: int = 5) -> str:
        counts: defaultdict[str, int] = defaultdict(int)
        for item in history:
            text = self._history_text(item).lower()
            for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text):
                if token in COMMON_STOPWORDS:
                    continue
                counts[token] += 1
        ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        if not ranked:
            return "No strong recurring keywords."
        return ", ".join(token for token, _ in ranked[:limit])

    def get_user_preference(self, user_id: str, history_k: int = 5, exclude_qid: str | None = None) -> tuple[str, bool]:
        user_id = _clean_text(user_id)
        if not user_id:
            return "User preference:\nStatus: missing user ID for this episode.", False

        raw_history = self.user_profiles.get(user_id, [])
        history = [item for item in raw_history if _clean_text(item.get("qid")) != _clean_text(exclude_qid)]
        if not history:
            return (
                "User preference:\n"
                f"User ID: {user_id}\n"
                "Recent purchase history: unavailable.\n"
                "Preference summary: no prior history found for this user in local data.",
                False,
            )

        history = sorted(history, key=self._history_sort_key)[-history_k:]
        common_needs = self._derive_common_needs(history)
        preferred_terms = self._derive_preferred_terms(history)
        price_tendency = self._derive_price_tendency(history)

        lines = [
            "User preference:",
            f"User ID: {user_id}",
            f"Price tendency: {price_tendency}",
            f"Preference keywords: {preferred_terms}",
            "Common needs:",
        ]
        lines.extend(f"- {_compact_text(need, limit=150)}" for need in common_needs)
        lines.append("Recent purchase history:")
        for idx, item in enumerate(reversed(history), start=1):
            item_id = _clean_text(item.get("item_id")).upper() or "UNKNOWN"
            title = _compact_text(item.get("title", ""), limit=80)
            query = _compact_text(item.get("query", ""), limit=120)
            review = _compact_text(item.get("review_text", ""), limit=120)
            verified = item.get("verified_purchase")
            verified_text = ""
            if isinstance(verified, bool):
                verified_text = f" | Verified: {'yes' if verified else 'no'}"

            if title:
                detail = f"Title: {title}"
            elif query:
                detail = f"Query: {query}"
            elif review:
                detail = f"Review: {review}"
            else:
                detail = "No summary available."
            lines.append(f"[{idx}] ID: {item_id} | {detail}{verified_text}")
        return "\n".join(lines), True
