"""Live HuggingFace metadata fetcher.

Mirrors obsidian-brain's ``src/embeddings/hf-metadata/`` subdir, in
particular the prompt-resolution chain in ``readme.ts`` /
``prompts.ts``. Layer 1 of the resolver chain — pure HF API client with
no DB / no project deps beyond ``huggingface_hub`` (already a
``[semantic]`` extra) and ``httpx``.

The fetcher reads a handful of config files from a model's HF repo to
extract:

    * embedding dim (from ``config.json``'s ``hidden_size`` / ``d_model``
      / ``n_embd`` / ``n_embed`` — different families use different
      field names; Dense layer override from ``modules.json`` when
      present)
    * max input tokens (priority:
      ``sentence_bert_config.max_seq_length``
      > ``tokenizer_config.model_max_length``
      > ``config.max_position_embeddings``)
    * query / document prefixes (priority: explicit
      ``config_sentence_transformers.json prompts``
      → upstream ``base_model``'s same JSON
      → README-fingerprinting fallback with language-aware script
      filtering — catches BGE-family models whose prefix is documented
      in prose only)
    * base model + ONNX size (best-effort, display-only)

All HF fetches are best-effort: ``config.json`` 404 is a hard fail
(``None``); every other file 404 just falls through to defaults.
Network failures (connection refused, timeout, DNS) are caught and
logged; the fetcher returns ``None`` so the resolver chain falls
through to the embedder probe.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ._logging import debug_log

_log = logging.getLogger("apple-notes-brain")


# 5s per-request timeout — HF API is usually <100ms; this is generous
# enough that flaky CI runners pass and tight enough that we don't hang
# the resolver waiting on a dead network.
DEFAULT_HF_TIMEOUT_S: float = 5.0

# Sanity cap for max-token fields — some configs set INT32_MAX as a
# sentinel; we treat anything above this as "no useful limit declared"
# and fall through to the next priority.
_SANE_MAX_TOKENS = 1_000_000


@dataclass(frozen=True)
class HfSources:
    """Diagnostic: which config files contributed to a resolved HfMetadata."""

    had_modules_json: bool = False
    had_sentence_bert_config: bool = False
    had_sentence_transformers_config: bool = False
    had_readme: bool = False
    max_tokens_from: str = "default"  # 'sentence_bert_config' | 'tokenizer_config' | 'config' | 'default'


@dataclass(frozen=True)
class HfMetadata:
    """Resolved per-model metadata from HF.

    Matches obsidian-brain's ``HfMetadata`` shape. Fields that the live
    HF API cannot supply are ``None`` so the resolver layer can layer
    overrides + defaults on top without losing the distinction between
    "unknown" and "explicitly empty".
    """

    model_id: str
    dim: int | None
    max_tokens: int
    query_prefix: str | None
    document_prefix: str | None
    prefix_source: str  # 'metadata' | 'metadata-base' | 'readme' | 'none'
    base_model: str | None
    size_bytes: int | None
    sources: HfSources = field(default_factory=HfSources)


# ---------------------------------------------------------------------------
# File fetch helpers — small, isolated, mockable
# ---------------------------------------------------------------------------


def _fetch_json_via_hub(
    repo_id: str, filename: str, *, cache_dir: str | None = None
) -> dict | list | None:
    """Best-effort fetch + JSON-parse of a single file from a HF repo.

    Returns ``None`` on any failure (404, JSON error, network, missing
    dep). Callers use ``None`` as "this file was unavailable" and
    fall through.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        _log.warning(
            "hf_metadata: huggingface_hub not installed; install [semantic] extra"
        )
        return None
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            etag_timeout=DEFAULT_HF_TIMEOUT_S,
        )
    except Exception as exc:  # 404, connection error, timeout, etc.
        debug_log(
            "hf_metadata: file fetch failed",
            repo=repo_id,
            file=filename,
            err=exc.__class__.__name__,
        )
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning(
            "hf_metadata: failed to parse %s for %s: %s",
            filename,
            repo_id,
            exc,
        )
        return None


def _fetch_model_card(repo_id: str) -> object | None:
    """Best-effort ``ModelCard.load`` for the repo's README frontmatter.

    Returns the raw ``ModelCard`` object (caller pokes at ``.data`` /
    ``.text``) or ``None`` when the README is absent / network failed.
    """
    try:
        from huggingface_hub import ModelCard  # type: ignore
    except ImportError:
        return None
    try:
        return ModelCard.load(repo_id)
    except Exception as exc:
        debug_log(
            "hf_metadata: ModelCard.load failed",
            repo=repo_id,
            err=exc.__class__.__name__,
        )
        return None


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------


def _extract_dim(config: dict, modules: list | None) -> int | None:
    """Embedding output dim — Dense layer override wins, else hidden_size family."""
    # Different families use different keys.
    hidden = (
        config.get("hidden_size")
        or config.get("d_model")
        or config.get("n_embd")
        or config.get("n_embed")
    )
    if isinstance(hidden, bool) or not isinstance(hidden, int):
        return None

    # modules.json may declare a Dense layer that projects to a different
    # output dim than the transformer's hidden_size (BGE-large variants,
    # mxbai-embed, mdbr-leaf, etc.). When present, its `out_features`
    # wins — but we can only see that here if the caller pre-fetched the
    # dense subconfig and attached it. The simple version of this
    # fetcher records hidden_size; the resolver can probe the live
    # embedder for the authoritative output dim.
    if isinstance(modules, list):
        for mod in modules:
            if not isinstance(mod, dict):
                continue
            t = mod.get("type", "")
            if isinstance(t, str) and t.endswith(".Dense"):
                # If the caller attached an `out_features` hint we use
                # it. Otherwise we leave dim at hidden_size — better
                # under-report than over-report.
                of = mod.get("out_features")
                if isinstance(of, int) and not isinstance(of, bool):
                    return of
    return int(hidden)


def _extract_max_tokens(
    config: dict | None,
    tokenizer_config: dict | None,
    sbert_config: dict | None,
) -> tuple[int, str]:
    """Resolve max tokens with documented priority. Returns (value, source)."""

    def _sane(v: object) -> int | None:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v <= 0 or v >= _SANE_MAX_TOKENS:
            return None
        return v

    if sbert_config:
        v = _sane(sbert_config.get("max_seq_length"))
        if v is not None:
            return v, "sentence_bert_config"
    if tokenizer_config:
        v = _sane(tokenizer_config.get("model_max_length"))
        if v is not None:
            return v, "tokenizer_config"
    if config:
        v = (
            _sane(config.get("max_position_embeddings"))
            or _sane(config.get("n_positions"))
            or _sane(config.get("max_trained_positions"))
        )
        if v is not None:
            # xlm-roberta reserves two positions for special tokens.
            if config.get("model_type") == "xlm-roberta":
                v = v - 2
            return v, "config"
    return 512, "default"


def _extract_prompts_from_st_config(
    st_config: dict | None,
) -> tuple[str | None, str | None]:
    """Pull (query, document) from ``config_sentence_transformers.json``."""
    if not isinstance(st_config, dict):
        return None, None
    prompts = st_config.get("prompts")
    if not isinstance(prompts, dict):
        return None, None
    q = prompts.get("query") if isinstance(prompts.get("query"), str) else None
    # `passage` is the older Sentence-Transformers key; treat as document.
    d = prompts.get("document")
    if not isinstance(d, str):
        d = prompts.get("passage") if isinstance(prompts.get("passage"), str) else None
    return q, d


# ---------------------------------------------------------------------------
# README frontmatter helpers (port of obsidian-brain readme.ts)
# ---------------------------------------------------------------------------


_LANG_CODES_ALLOWED = frozenset(
    {
        "en", "zh", "ja", "ko", "ar", "fa", "ru", "de", "fr", "es", "pt",
        "it", "nl", "vi", "tr", "pl", "hi",
    }
)


def extract_base_model(readme_text: str) -> str | None:
    """Pull ``base_model:`` from a README's YAML frontmatter.

    Handles both single-string and list forms (returns the first entry
    in the list case). Returns ``None`` when the README has no
    frontmatter or no ``base_model`` field.
    """
    if not isinstance(readme_text, str):
        return None
    fm_match = re.match(r"^---\n([\s\S]*?)\n---", readme_text)
    if not fm_match:
        return None
    fm = fm_match.group(1)
    # Single-string form. `[ \t]*` (NOT `\s*`) — the latter would
    # swallow the newline and fire on the list form.
    single = re.search(r"^base_model:[ \t]*(.+)$", fm, re.MULTILINE)
    if single:
        return single.group(1).strip().strip("\"'")
    # List form.
    lst = re.search(
        r"^base_model:[ \t]*\n[ \t]*-[ \t]*(.+)$", fm, re.MULTILINE
    )
    if lst:
        return lst.group(1).strip().strip("\"'")
    return None


def detect_model_language(readme_text: str | None, model_id: str) -> str | None:
    """Detect the model's primary language from README frontmatter, then
    the model id's suffix conventions. Returns an ISO 639 code or None
    for multilingual / unknown.
    """
    if isinstance(readme_text, str):
        fm_match = re.match(r"^---\n([\s\S]*?)\n---", readme_text)
        if fm_match:
            fm = fm_match.group(1) + "\n"
            single = re.search(
                r"^language:[ \t]*([a-z]{2,3})[ \t]*$", fm, re.MULTILINE
            )
            if single:
                return single.group(1).lower()
            list_match = re.search(
                r"^language:[ \t]*\n((?:[ \t]*-[ \t]*[a-z]{2,3}[ \t]*\n)+)",
                fm,
                re.MULTILINE,
            )
            if list_match:
                langs = re.findall(r"-[ \t]*([a-z]{2,3})", list_match.group(1))
                if len(langs) == 1:
                    return langs[0].lower()
                return None  # multilingual list — don't claim a single language
    id_match = re.search(r"[-_/]([a-z]{2})(?:[-_.]|$)", model_id, re.IGNORECASE)
    if id_match:
        code = id_match.group(1).lower()
        if code in _LANG_CODES_ALLOWED:
            return code
    return None


_LANG_TO_SCRIPT = {
    "en": "latin", "de": "latin", "fr": "latin", "es": "latin", "pt": "latin",
    "it": "latin", "nl": "latin", "vi": "latin", "tr": "latin", "pl": "latin",
    "id": "latin",
    "zh": "cjk", "ja": "cjk", "ko": "cjk",
    "ar": "arabic", "fa": "arabic", "ur": "arabic",
    "ru": "cyrillic", "uk": "cyrillic", "bg": "cyrillic",
    "hi": "devanagari",
}


def language_to_script(lang: str) -> str | None:
    """Map an ISO 639 language code to its dominant script class."""
    return _LANG_TO_SCRIPT.get(lang)


def detect_prefix_script(prefix: str) -> str:
    """Classify a candidate prefix string by dominant script.

    Strips punctuation/digits/whitespace then counts code-point ranges.
    Used to filter README-fingerprinted prefixes against the model's
    declared language so BGE-en doesn't pick the Chinese prefix that
    appears more frequently in a side-by-side EN+ZH README.
    """
    text = re.sub(r"[\s\d:：_/.\-,'\"!?]", "", prefix)
    if not text:
        return "latin"
    cjk = arabic = cyrillic = latin = 0
    for ch in text:
        cp = ord(ch)
        if (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3000 <= cp <= 0x30FF)
            or (0xAC00 <= cp <= 0xD7AF)
        ):
            cjk += 1
        elif 0x0600 <= cp <= 0x06FF:
            arabic += 1
        elif 0x0400 <= cp <= 0x04FF:
            cyrillic += 1
        elif 0x41 <= cp <= 0x7A:
            latin += 1
    biggest = max(cjk, arabic, cyrillic, latin)
    if biggest == 0:
        return "latin"
    if cjk == biggest:
        return "cjk"
    if arabic == biggest:
        return "arabic"
    if cyrillic == biggest:
        return "cyrillic"
    return "latin"


def _is_plausible_prefix(s: str) -> bool:
    """A real model prefix ends in ``": "`` (Latin) or ``"："`` (CJK
    fullwidth) — bare ``":"`` would let Python print labels like
    ``"Sentence embeddings:"`` slip through.
    """
    if len(s) < 5 or len(s) > 80:
        return False
    if re.search(r"[\n\r{}\[\]()=<>|;]", s):
        return False
    if not re.search(r"(: |：)$", s):
        return False
    trimmed = s.rstrip()
    if re.match(r"^[#/]", trimmed):
        return False
    if re.search(r"Score\s*:|Options\s*:", trimmed, re.IGNORECASE):
        return False
    return True


_QUERY_KEYWORDS = re.compile(
    r"(query|search|represent|instruction|为这个句子|سوال|质问)",
    re.IGNORECASE,
)

_DOC_PREFIX_RE = re.compile(
    r"^([a-z_]+_)?(passage|document)s?\s*[:：]\s*$", re.IGNORECASE
)


def resolve_prompts_from_readme(
    readme_text: str, expected_script: str | None = None
) -> tuple[str | None, str | None]:
    """Fingerprint a README for query/document prefixes.

    Generic — counts quoted candidate strings and ranks by frequency +
    presence of query/instruction keywords. When ``expected_script`` is
    set (i.e. the model declares a single language), candidates with a
    non-matching script are dropped first. Returns ``(query, document)``;
    either may be ``None``.
    """
    if not isinstance(readme_text, str):
        return None, None
    # Strip the YAML frontmatter so `description: "query: ..."` from the
    # metadata block doesn't get picked up.
    body = re.sub(r"^---\n[\s\S]*?\n---\n?", "", readme_text)

    strings: list[str] = []
    for pat in (
        r'"([^"\n]{1,200})"',
        r"'([^'\n]{1,200})'",
        r"`([^`\n]{1,200})`",
    ):
        for m in re.finditer(pat, body):
            strings.append(m.group(1))

    counts: dict[str, int] = {}

    def _bump(s: str) -> None:
        counts[s] = counts.get(s, 0) + 1

    for s in strings:
        # Pattern A — the whole string IS a prefix.
        if re.search(r"(: |：)$", s) and _is_plausible_prefix(s):
            _bump(s)
        # Pattern B — string starts with a `prefix: <body>` shape.
        pmatch = re.match(r"^([A-Za-z][A-Za-z0-9 _]{2,40}: )", s)
        if pmatch and _is_plausible_prefix(pmatch.group(1)):
            _bump(pmatch.group(1))

    if not counts:
        return None, None

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])

    # Language-aware filter — drop candidates from other scripts when at
    # least one candidate in the expected script exists.
    if expected_script:
        matching = [(p, c) for (p, c) in ranked if detect_prefix_script(p) == expected_script]
        if matching:
            ranked = matching

    credible = [(p, c) for (p, c) in ranked if c >= 2 or _QUERY_KEYWORDS.search(p)]
    if not credible:
        return None, None

    doc_prefix: str | None = None
    query_prefix: str | None = None
    for p, _c in credible:
        if doc_prefix is None and _DOC_PREFIX_RE.match(p):
            doc_prefix = p
        elif query_prefix is None and not _DOC_PREFIX_RE.match(p):
            query_prefix = p
        if query_prefix and doc_prefix:
            break

    return query_prefix, doc_prefix


def _readme_text_from_card(card: object | None) -> str | None:
    """Pull the plain-text README content out of a ModelCard object."""
    if card is None:
        return None
    text = getattr(card, "text", None) or getattr(card, "content", None)
    if isinstance(text, str) and text:
        return text
    # ModelCard.__str__ returns the full markdown source — fall back.
    try:
        rendered = str(card)
    except Exception:
        return None
    return rendered if rendered else None


def _card_data_get(card: object | None, key: str) -> object:
    """Read a field from ``ModelCard.data`` defensively."""
    if card is None:
        return None
    data = getattr(card, "data", None)
    if data is None:
        return None
    # ModelCardData behaves like a mapping; ``.get`` may exist or may not.
    getter = getattr(data, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except TypeError:
            return None
    return getattr(data, key, None)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def get_embedding_metadata(
    repo_id: str, *, cache_dir: str | None = None
) -> HfMetadata | None:
    """Fetch HF metadata for ``repo_id``. Returns ``None`` on hard failure.

    Hard failure means ``config.json`` is unreachable / 404 / not
    parseable — the model probably doesn't exist or HF is unreachable.
    Every other config file is best-effort; a missing
    ``modules.json`` / ``tokenizer_config.json`` / etc. falls through to
    defaults.

    Network/connection failures are caught and logged at WARN level so
    the resolver chain falls through to the embedder probe instead of
    crashing the boot.
    """
    debug_log("hf_metadata: fetching", repo=repo_id)
    config = _fetch_json_via_hub(repo_id, "config.json", cache_dir=cache_dir)
    if not isinstance(config, dict):
        _log.warning(
            "hf_metadata: config.json unreachable / unparseable for %s — "
            "skipping HF metadata fetch (resolver will fall through)",
            repo_id,
        )
        return None

    tokenizer_config = _fetch_json_via_hub(
        repo_id, "tokenizer_config.json", cache_dir=cache_dir
    )
    sbert_config = _fetch_json_via_hub(
        repo_id, "sentence_bert_config.json", cache_dir=cache_dir
    )
    st_config = _fetch_json_via_hub(
        repo_id, "config_sentence_transformers.json", cache_dir=cache_dir
    )
    modules = _fetch_json_via_hub(repo_id, "modules.json", cache_dir=cache_dir)

    dim = _extract_dim(config, modules if isinstance(modules, list) else None)
    max_tokens, mt_from = _extract_max_tokens(
        config,
        tokenizer_config if isinstance(tokenizer_config, dict) else None,
        sbert_config if isinstance(sbert_config, dict) else None,
    )

    # Prompts — tier 1: direct config_sentence_transformers.json.
    query, document = _extract_prompts_from_st_config(
        st_config if isinstance(st_config, dict) else None
    )
    prefix_source = "metadata" if (query or document) else "none"
    base_model: str | None = None
    readme_text: str | None = None
    card = _fetch_model_card(repo_id)
    if card is not None:
        readme_text = _readme_text_from_card(card)
        # ModelCardData parses the YAML frontmatter for us.
        bm = _card_data_get(card, "base_model")
        if isinstance(bm, list) and bm:
            bm = bm[0]
        if isinstance(bm, str) and bm.strip():
            base_model = bm.strip()
        elif readme_text:
            # Fallback regex when the parser didn't pick it up.
            base_model = extract_base_model(readme_text)

    # Tier 2: upstream base_model's same JSON, when we still don't have
    # prompts and the README declared a base.
    if (query is None and document is None) and base_model and base_model != repo_id:
        upstream_st = _fetch_json_via_hub(
            base_model, "config_sentence_transformers.json", cache_dir=cache_dir
        )
        if isinstance(upstream_st, dict):
            uq, ud = _extract_prompts_from_st_config(upstream_st)
            if uq or ud:
                query, document = uq, ud
                prefix_source = "metadata-base"

    # Tier 3: README fingerprinting (this repo, then upstream's). Language-
    # aware script filter prevents the well-known BGE-en-picks-Chinese bug.
    if query is None and document is None:
        candidates: list[tuple[str, str]] = []  # (id, readme text)
        if readme_text:
            candidates.append((repo_id, readme_text))
        if base_model and base_model != repo_id:
            upstream_card = _fetch_model_card(base_model)
            upstream_readme = _readme_text_from_card(upstream_card)
            if upstream_readme:
                candidates.append((base_model, upstream_readme))
        for cid, ctext in candidates:
            lang = detect_model_language(ctext, cid)
            expected = language_to_script(lang) if lang else None
            fq, fd = resolve_prompts_from_readme(ctext, expected)
            if fq or fd:
                query, document = fq, fd
                prefix_source = "readme"
                break

    sources = HfSources(
        had_modules_json=isinstance(modules, list),
        had_sentence_bert_config=isinstance(sbert_config, dict),
        had_sentence_transformers_config=isinstance(st_config, dict),
        had_readme=bool(readme_text),
        max_tokens_from=mt_from,
    )

    debug_log(
        "hf_metadata: resolved",
        repo=repo_id,
        dim=dim,
        max_tokens=max_tokens,
        prefix_source=prefix_source,
        base_model=base_model,
    )
    return HfMetadata(
        model_id=repo_id,
        dim=dim,
        max_tokens=max_tokens,
        query_prefix=query,
        document_prefix=document,
        prefix_source=prefix_source,
        base_model=base_model,
        size_bytes=None,
        sources=sources,
    )
