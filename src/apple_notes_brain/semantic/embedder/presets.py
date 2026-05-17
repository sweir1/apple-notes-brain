"""Named embedding presets + atomic env-var resolver.

Mirrors obsidian-brain's `src/embeddings/presets.ts` 1:1 — same six
preset names, same default, same precedence ladder in
`resolve_preset_config()`. When obsidian-brain adds a preset, mirror it
here.

The point of this module is to keep `(provider, model)` paired
*atomically*: every consumer that needs to translate user-facing env
vars into the concrete (provider, model) the embedder should construct
calls `resolve_preset_config(os.environ)` — never re-implements the
precedence locally. (Pre-uplift the resolution was split across
`config.py` + the embedder factory and we had drift bugs.)

Precedence (highest first):
  1. `EMBEDDING_MODEL` set → use it raw; provider = `EMBEDDING_PROVIDER`
     or default `onnx`. `preset_short_name = None`.
  2. `EMBEDDING_PROVIDER` + `EMBEDDING_PRESET` mismatch → provider wins,
     preset's model carried over, mismatch warning emitted (one-shot
     stderr).
  3. `EMBEDDING_PRESET` set → preset's declared `(provider, model)` pair
     used atomically. Deprecated aliases resolve to canonical names with
     a one-shot warning.
  4. `EMBEDDING_PROVIDER` set alone → provider-default model
     (`DEFAULT_OLLAMA_MODEL` for ollama, `DEFAULT_PRESET.onnx_repo` for
     onnx).
  5. Nothing set → `DEFAULT_PRESET`.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Literal, Mapping

_log = logging.getLogger("apple-notes-brain")


@dataclass(frozen=True)
class EmbeddingPreset:
    """A named (provider-bound) embedding model spec.

    Each preset declares its *primary* provider (`onnx` or `ollama`) and
    the concrete model identifier on that provider. `onnx_repo` /
    `ollama_model` differ because providers use different identifier
    conventions: HuggingFace uses `<org>/<model>`, Ollama uses
    `<model>:<tag>`.
    """

    short_name: str
    provider: Literal["onnx", "ollama"]
    # On-provider model identifier (HF repo for onnx, Ollama tag for ollama).
    model: str
    # ONNX-side download paths inside the HF repo. Set for onnx presets;
    # may also be useful for ollama presets that have an HF anchor.
    onnx_repo: str
    onnx_file: str
    onnx_tokenizer: str
    # Ollama-side identifier (used when the preset is provider=ollama, OR
    # when the user pairs the preset with `EMBEDDING_PROVIDER=ollama`).
    ollama_model: str
    dim: int
    description: str


# ─── Defaults — change in ONE place, every consumer follows ──────────────
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"


# ─── Preset registry — mirrors obsidian-brain/src/embeddings/presets.ts ──
# Dim values:
#   - bge-small/bge-base/multilingual-e5-small: well-known fixed dims (384/768/384).
#   - mdbr-leaf-ir: 384-dim base, runtime probe will confirm at boot.
#   - multilingual-e5-base: 768.
#   - qwen3-embedding:0.6b: 1024 per official model card.
EMBEDDING_PRESETS: dict[str, EmbeddingPreset] = {
    "english": EmbeddingPreset(
        short_name="english",
        provider="onnx",
        model="Xenova/bge-small-en-v1.5",
        onnx_repo="Xenova/bge-small-en-v1.5",
        onnx_file="onnx/model_quantized.onnx",
        onnx_tokenizer="tokenizer.json",
        ollama_model="bge-small-en-v1.5",
        dim=384,
        description=(
            "Default. English-only retrieval (BAAI/bge-small-en-v1.5 via "
            "Xenova ONNX quant). 384-dim, ~30MB."
        ),
    ),
    "english-fast": EmbeddingPreset(
        short_name="english-fast",
        provider="onnx",
        model="MongoDB/mdbr-leaf-ir",
        onnx_repo="MongoDB/mdbr-leaf-ir",
        onnx_file="onnx/model_quantized.onnx",
        onnx_tokenizer="tokenizer.json",
        ollama_model="mdbr-leaf-ir",
        dim=384,
        description=(
            "Fastest English retrieval. MongoDB/mdbr-leaf-ir, Apache-2.0, "
            "retrieval-tuned 23M-param distillation of mxbai-embed-large-v1. "
            "Asymmetric (uses query prefix); 384-dim."
        ),
    ),
    "english-quality": EmbeddingPreset(
        short_name="english-quality",
        provider="onnx",
        model="Xenova/bge-base-en-v1.5",
        onnx_repo="Xenova/bge-base-en-v1.5",
        onnx_file="onnx/model_quantized.onnx",
        onnx_tokenizer="tokenizer.json",
        ollama_model="bge-base-en-v1.5",
        dim=768,
        description=(
            "Higher-quality English retrieval (BAAI/bge-base-en-v1.5 via "
            "Xenova ONNX quant). 768-dim, ~100MB."
        ),
    ),
    "multilingual": EmbeddingPreset(
        short_name="multilingual",
        provider="onnx",
        model="Xenova/multilingual-e5-small",
        onnx_repo="Xenova/multilingual-e5-small",
        onnx_file="onnx/model_quantized.onnx",
        onnx_tokenizer="tokenizer.json",
        ollama_model="multilingual-e5-small",
        dim=384,
        description=(
            "Multilingual retrieval (intfloat/multilingual-e5-small via "
            "Xenova ONNX quant). 384-dim, ~120MB. Asymmetric "
            "(query:/passage: prefixes)."
        ),
    ),
    "multilingual-quality": EmbeddingPreset(
        short_name="multilingual-quality",
        provider="onnx",
        model="Xenova/multilingual-e5-base",
        onnx_repo="Xenova/multilingual-e5-base",
        onnx_file="onnx/model_quantized.onnx",
        onnx_tokenizer="tokenizer.json",
        ollama_model="multilingual-e5-base",
        dim=768,
        description=(
            "Higher-quality multilingual retrieval (intfloat/multilingual-e5-base "
            "via Xenova ONNX quant). 768-dim. Known transformers.js "
            "token_type_ids bug for long inputs; prefer multilingual-ollama "
            "for lossless multilingual quality."
        ),
    ),
    "multilingual-ollama": EmbeddingPreset(
        short_name="multilingual-ollama",
        provider="ollama",
        model="qwen3-embedding:0.6b",
        # Even though this preset is provider=ollama, we record the HF
        # repo for the same model so users who want to switch to ONNX
        # have a sane anchor (Qwen/Qwen3-Embedding-0.6B on HF).
        onnx_repo="Qwen/Qwen3-Embedding-0.6B",
        onnx_file="onnx/model_quantized.onnx",
        onnx_tokenizer="tokenizer.json",
        ollama_model="qwen3-embedding:0.6b",
        dim=1024,
        description=(
            "Multilingual via Ollama (qwen3-embedding:0.6b). 1024-dim, 32k ctx. "
            "Requires a local Ollama server."
        ),
    ),
}


DEFAULT_PRESET = EMBEDDING_PRESETS["english"]


# ─── Deprecation aliases ─────────────────────────────────────────────────
# Pre-uplift our presets were the bare model names. Map them to the new
# canonical preset names with a one-shot stderr warning.
DEPRECATED_PRESET_ALIASES: dict[str, str] = {
    "bge-small-en-v1.5": "english",
    "bge-base-en-v1.5": "english-quality",
    "all-MiniLM-L6-v2": "english",
    # Obsidian-brain back-compat (transferable muscle memory):
    "fastest": "english-fast",
    "balanced": "english",
}


# ─── One-shot warning trackers (process-lifetime) ────────────────────────
_warned_aliases: set[str] = set()
_warned_provider_mismatch: bool = False


def _warn_stderr(msg: str) -> None:
    """Sync stderr write — race-safe on crash; mirrors obsidian-brain `logger.warn`."""
    sys.stderr.write(f"apple-notes-brain WARN: {msg}\n")
    sys.stderr.flush()
    _log.warning(msg)


def _emit_alias_warning(raw: str, canonical: str) -> None:
    if raw in _warned_aliases:
        return
    _warned_aliases.add(raw)
    if raw == "all-MiniLM-L6-v2":
        _warn_stderr(
            f"EMBEDDING_PRESET/EMBEDDING_MODEL={raw!r} is deprecated. "
            f"It now resolves to {canonical!r} (Xenova/bge-small-en-v1.5) — "
            "a different model than the old Xenova/all-MiniLM-L6-v2. Your "
            "index will re-embed once on next boot. To keep the old model "
            "explicitly, set EMBEDDING_MODEL=Xenova/all-MiniLM-L6-v2."
        )
    else:
        _warn_stderr(
            f"EMBEDDING_PRESET={raw!r} is deprecated and has been renamed "
            f"to {canonical!r}. Please update your configuration."
        )


def _emit_provider_mismatch_warning(
    override: str, preset_name: str, preset_provider: str, preset_model: str
) -> None:
    global _warned_provider_mismatch
    if _warned_provider_mismatch:
        return
    _warned_provider_mismatch = True
    _warn_stderr(
        f"EMBEDDING_PROVIDER={override!r} overrides EMBEDDING_PRESET={preset_name!r} "
        f"which declares provider={preset_provider!r}. Will attempt "
        f"model={preset_model!r} on provider={override!r} — likely fails unless "
        "that model exists on the chosen provider. Either remove "
        "EMBEDDING_PROVIDER, switch to a preset that declares the desired "
        "provider, or set EMBEDDING_MODEL explicitly."
    )


# ─── Resolver primitives ─────────────────────────────────────────────────

def _parse_explicit_provider(env: Mapping[str, str]) -> Literal["onnx", "ollama"] | None:
    raw = env.get("EMBEDDING_PROVIDER")
    if raw is None or not raw.strip():
        return None
    v = raw.strip().lower()
    if v not in {"onnx", "ollama"}:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER={raw!r}. Valid providers: onnx, ollama."
        )
    return v  # type: ignore[return-value]


def _resolve_canonical_preset(raw_name: str) -> str:
    """Map a raw preset/alias name to a canonical preset short name.

    Raises ValueError when the name is neither a known preset nor an alias.
    """
    lowered = raw_name.strip()
    if lowered in DEPRECATED_PRESET_ALIASES:
        canonical = DEPRECATED_PRESET_ALIASES[lowered]
        _emit_alias_warning(lowered, canonical)
        return canonical
    if lowered in EMBEDDING_PRESETS:
        return lowered
    valid = ", ".join(EMBEDDING_PRESETS.keys())
    raise ValueError(
        f"Unknown EMBEDDING_PRESET={raw_name!r}. Valid presets: {valid}. "
        "Or set EMBEDDING_MODEL to a specific HuggingFace repo / Ollama model id."
    )


def resolve_preset(
    model: str, provider: Literal["onnx", "ollama"]
) -> tuple[EmbeddingPreset | None, str]:
    """Resolve a model identifier to (preset, on-provider identifier).

    Lookup tries (in order):
      1. Exact short-name match in `EMBEDDING_PRESETS`.
      2. Deprecation alias map (`bge-small-en-v1.5` → `english`, etc.) —
         emits a one-shot stderr warning.
      3. Exact match against any preset's `onnx_repo` or `ollama_model`
         field (so a resolved HF id from `resolve_preset_config` lands
         back on the right preset).
      4. Unknown → `(None, model)` — caller treats `model` as a literal.
    """
    if not isinstance(model, str):
        return None, model  # type: ignore[unreachable]

    # 1. Exact short-name match
    preset = EMBEDDING_PRESETS.get(model)
    if preset is not None:
        identifier = preset.onnx_repo if provider == "onnx" else preset.ollama_model
        return preset, identifier

    # 2. Deprecation aliases
    if model in DEPRECATED_PRESET_ALIASES:
        canonical = DEPRECATED_PRESET_ALIASES[model]
        _emit_alias_warning(model, canonical)
        preset = EMBEDDING_PRESETS[canonical]
        identifier = preset.onnx_repo if provider == "onnx" else preset.ollama_model
        return preset, identifier

    # 3. Reverse lookup by on-provider model id (so a resolved HF id like
    #    "Xenova/bge-small-en-v1.5" still maps back to the english preset).
    for p in EMBEDDING_PRESETS.values():
        if provider == "onnx" and model == p.onnx_repo:
            return p, p.onnx_repo
        if provider == "ollama" and model == p.ollama_model:
            return p, p.ollama_model

    # 4. Unknown — treat as literal
    return None, model


# ─── Atomic env resolver — every consumer calls this ─────────────────────

@dataclass(frozen=True)
class ResolvedPresetConfig:
    """The atomic resolution returned by `resolve_preset_config`.

    `preset_short_name` is `None` only when the user set `EMBEDDING_MODEL`
    explicitly (the "power-user" path) or `EMBEDDING_PROVIDER=ollama`
    alone (no preset implied); every other branch returns a canonical
    preset name.
    """

    provider: Literal["onnx", "ollama"]
    model: str
    preset_short_name: str | None
    source: Literal["env-model", "env-preset", "env-provider", "default"]


def resolve_preset_config(env: Mapping[str, str]) -> ResolvedPresetConfig:
    """The single env-var → (provider, model, preset_short_name) resolver.

    See module docstring for the precedence ladder. Every consumer should
    call this — never re-implement env-var precedence locally — so
    provider and model cannot desync.
    """
    explicit_provider = _parse_explicit_provider(env)

    # (1) Power-user path: EMBEDDING_MODEL set → use it raw.
    raw_model = env.get("EMBEDDING_MODEL")
    if raw_model and raw_model.strip():
        # Honour deprecation aliases for back-compat with users who still
        # set EMBEDDING_MODEL=bge-small-en-v1.5: route through the alias
        # map AND surface a warning, but expand to the canonical model id
        # so the embedder constructs against the actual HF repo.
        cleaned = raw_model.strip()
        if cleaned in DEPRECATED_PRESET_ALIASES:
            canonical = DEPRECATED_PRESET_ALIASES[cleaned]
            _emit_alias_warning(cleaned, canonical)
            preset = EMBEDDING_PRESETS[canonical]
            return ResolvedPresetConfig(
                provider=explicit_provider or preset.provider,
                model=preset.model,
                preset_short_name=canonical,
                source="env-model",
            )
        return ResolvedPresetConfig(
            provider=explicit_provider or "onnx",
            model=cleaned,
            preset_short_name=None,
            source="env-model",
        )

    # (2/3) Preset path: EMBEDDING_PRESET set.
    raw_preset = env.get("EMBEDDING_PRESET")
    if raw_preset and raw_preset.strip():
        canonical = _resolve_canonical_preset(raw_preset.strip().lower())
        preset = EMBEDDING_PRESETS[canonical]

        if explicit_provider is None:
            # No provider override → preset's declared pair.
            return ResolvedPresetConfig(
                provider=preset.provider,
                model=preset.model,
                preset_short_name=canonical,
                source="env-preset",
            )
        if explicit_provider == preset.provider:
            return ResolvedPresetConfig(
                provider=explicit_provider,
                model=preset.model,
                preset_short_name=canonical,
                source="env-preset",
            )
        # Provider override CONFLICTS with preset → warn, honour override
        # on provider, carry the preset's model anyway (likely fails at
        # runtime, but the warning explains why).
        _emit_provider_mismatch_warning(
            explicit_provider, canonical, preset.provider, preset.model
        )
        return ResolvedPresetConfig(
            provider=explicit_provider,
            model=preset.model,
            preset_short_name=canonical,
            source="env-preset",
        )

    # (4) Provider override alone → provider-default model.
    if explicit_provider == "ollama":
        return ResolvedPresetConfig(
            provider="ollama",
            model=DEFAULT_OLLAMA_MODEL,
            preset_short_name=None,
            source="env-provider",
        )
    if explicit_provider == "onnx":
        return ResolvedPresetConfig(
            provider="onnx",
            model=DEFAULT_PRESET.model,
            preset_short_name=DEFAULT_PRESET.short_name,
            source="env-provider",
        )

    # (5) Nothing set → DEFAULT_PRESET.
    return ResolvedPresetConfig(
        provider=DEFAULT_PRESET.provider,
        model=DEFAULT_PRESET.model,
        preset_short_name=DEFAULT_PRESET.short_name,
        source="default",
    )


# ─── Test-only helpers (do NOT call in production) ───────────────────────

def _reset_warning_state() -> None:
    """Clear the one-shot warning trackers. Tests call this between cases."""
    global _warned_provider_mismatch
    _warned_aliases.clear()
    _warned_provider_mismatch = False
