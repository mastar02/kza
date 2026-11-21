"""Pure pre-field inventory validation and scoring for the P6 protocol."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from itertools import combinations
import json
import math
import re
from statistics import mean, median

from src.ambient.wer import _edit_ops, is_excluded, normalize_words, score
from src.core.task_outcome import TaskOutcome
from src.core.voice_delivery import ResponseResult


class Partition(StrEnum):
    """Independently registered sources of evaluation material."""

    TRAINING = "training"
    CALIBRATION = "calibration"
    RESERVED = "reserved"


@dataclass(frozen=True)
class EvaluationClip:
    """Opaque clip provenance as verified by the corpus custodian."""

    clip_id: str
    audio_sha256: str
    source_sha256: str
    session_id: str
    split: Partition
    start: datetime
    end: datetime
    speaker_refs: frozenset[str]
    consent_ref: str


@dataclass(frozen=True)
class ConsentReceipt:
    """Current private register entry for a complete capture session."""

    session_id: str
    purpose: str
    granted: bool
    speaker_refs: frozenset[str]
    granted_at: datetime
    expires_at: datetime
    revoked_at: datetime | None = None


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("capture_clock_requires_timezone")
    return value.astimezone(timezone.utc)


def _opaque(value: str) -> bool:
    return type(value) is str and bool(value) and value.strip() == value


def validate_inventory(
    clips: Iterable[EvaluationClip],
    receipts: Mapping[str, ConsentReceipt],
    *,
    now: datetime,
    shared_panel: frozenset[str] = frozenset(),
) -> None:
    """Require consent and separation before admitting a private batch.

    Args:
        clips: Full split inventory with original lineage and canonical PCM hashes.
        receipts: Current independent consent records keyed by opaque receipt reference.
        now: Aware admission time.
        shared_panel: Preregistered voices allowed in calibration and reserve.

    Raises:
        PermissionError: Consent is missing, invalid, revoked or expired.
        ValueError: The inventory is incomplete, invalid or contaminated.
    """
    if type(shared_panel) is not frozenset or not all(map(_opaque, shared_panel)):
        raise ValueError("invalid_shared_panel")
    now = _utc(now)
    partitions: set[Partition] = set()
    clip_ids: set[str] = set()
    identifiers: dict[tuple[str, str], Partition] = {}
    speakers: dict[str, set[Partition]] = {}
    windows: list[tuple[datetime, datetime, bool]] = []
    for clip in clips:
        if (
            type(clip.split) is not Partition
            or not all(map(_opaque, (clip.clip_id, clip.session_id, clip.consent_ref)))
            or any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in (clip.audio_sha256, clip.source_sha256))
            or type(clip.speaker_refs) is not frozenset
            or not all(map(_opaque, clip.speaker_refs))
        ):
            raise ValueError("invalid_clip_provenance")
        start, end = _utc(clip.start), _utc(clip.end)
        if not start < end <= now:
            raise ValueError("invalid_capture_window")
        receipt = receipts.get(clip.consent_ref)
        if (
            type(receipt) is not ConsentReceipt
            or receipt.granted is not True
            or receipt.purpose != "consented_calibration_and_evaluation"
            or receipt.session_id != clip.session_id
            or type(receipt.speaker_refs) is not frozenset
            or not clip.speaker_refs <= receipt.speaker_refs
            or not _utc(receipt.granted_at) <= start < end <= now < _utc(receipt.expires_at)
            or (receipt.revoked_at is not None and _utc(receipt.revoked_at) <= now)
        ):
            raise PermissionError("explicit_current_consent_required")

        partitions.add(clip.split)
        for key in (("clip", clip.clip_id), ("session", clip.session_id),
                    ("audio", clip.audio_sha256), ("audio", clip.source_sha256)):
            if identifiers.setdefault(key, clip.split) is not clip.split:
                raise ValueError(f"cross_partition_{key[0]}")
        if clip.clip_id in clip_ids:
            raise ValueError("duplicate_clip")
        clip_ids.add(clip.clip_id)
        for speaker in clip.speaker_refs:
            speakers.setdefault(speaker, set()).add(clip.split)
        windows.append((start, end, clip.split is Partition.RESERVED))

    if not {Partition.CALIBRATION, Partition.RESERVED} <= partitions:
        raise ValueError("calibration_and_reserve_required")
    for speaker, splits in speakers.items():
        if Partition.RESERVED in splits and len(splits) > 1 and (
            speaker not in shared_panel or Partition.TRAINING in splits
        ):
            raise ValueError("cross_partition_speaker")

    latest_end: dict[bool, datetime] = {}
    for start, end, reserved in sorted(windows):
        other_end = latest_end.get(not reserved)
        if other_end is not None and start < other_end:
            raise ValueError("cross_partition_window")
        if reserved not in latest_end or end > latest_end[reserved]:
            latest_end[reserved] = end


def canonical_sha256(value: object) -> str:
    """Hash a deterministic UTF-8 JSON serialization of manifest metadata."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def seeded_order(ids: Iterable[str], *, seed: int, purpose: str) -> tuple[str, ...]:
    """Order unique opaque IDs reproducibly, independent of enumeration order."""
    ids = tuple(ids)
    if type(seed) is not int or not 0 <= seed <= 9 or purpose not in {"h3", "listening"}:
        raise ValueError("unknown_order_protocol")
    if not all(map(_opaque, ids)) or len(set(ids)) != len(ids):
        raise ValueError("order_requires_unique_ids")
    return tuple(sorted(ids, key=lambda item: (canonical_sha256(["p6-v1", purpose, seed, item]), item)))


@dataclass(frozen=True)
class Recognition:
    """Counts and rates for eligible speech in one stratum."""

    eligible: int
    measured: int
    reference_words: int
    reference_characters: int
    substitutions: int
    insertions: int
    deletions: int
    empty_hypotheses: int
    wer_micro: float
    wer_macro: float
    cer_micro: float
    coverage: float


def recognition(pairs: Iterable[tuple[str | None, str | None]]) -> Recognition:
    """Score aligned speech; None marks missing evidence and '' a confirmed empty STT."""
    eligible = measured = words = characters = subs = ins = dels = empty = char_errors = 0
    wers = []
    for reference, hypothesis in pairs:
        eligible += 1
        if reference is None:
            continue
        ref_words = normalize_words(reference)
        if not ref_words or is_excluded(reference):
            raise ValueError("speech_reference_requires_words")
        if hypothesis is None:
            continue
        hyp_words = normalize_words(hypothesis)
        result = score(reference, hypothesis)
        ref_chars, hyp_chars = list("".join(ref_words)), list("".join(hyp_words))
        measured += 1
        words += result.ref_words
        characters += len(ref_chars)
        subs += result.subs
        ins += result.ins
        dels += result.dels
        empty += not hyp_words
        char_errors += sum(_edit_ops(ref_chars, hyp_chars))
        wers.append(result.wer)
    if not measured:
        raise ValueError("no_aligned_speech_to_score")
    return Recognition(eligible, measured, words, characters, subs, ins, dels, empty,
                       (subs + ins + dels) / words, mean(wers), char_errors / characters,
                       measured / eligible)


def complete_task(
    outcome: TaskOutcome,
    *,
    intent_correct: bool,
    targets_correct: bool,
    effect_observed: bool,
) -> bool:
    """Require runtime completion and independent adjudication of the expected effect."""
    return outcome.task_completed and all(
        flag is True for flag in (intent_correct, targets_correct, effect_observed)
    )


@dataclass(frozen=True)
class WakeCounts:
    """Reviewed occurrence counts after corroborated physical deduplication."""

    accepted_intentional: int
    rejected_intentional: int
    uncertain_intentional: int
    confirmed_false: int
    uncertain_emitted: int
    reviewed_emitted: int
    useful_house_seconds: int

    def __post_init__(self) -> None:
        counts = (self.accepted_intentional, self.rejected_intentional, self.uncertain_intentional,
                  self.confirmed_false, self.uncertain_emitted, self.reviewed_emitted,
                  self.useful_house_seconds)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("invalid_wake_counts")
        if self.confirmed_false > self.reviewed_emitted:
            raise ValueError("false_wake_requires_review")

    @property
    def rejection_upper(self) -> float:
        """Count unverified intentional trials as potential rejections."""
        total = self.accepted_intentional + self.rejected_intentional + self.uncertain_intentional
        return (self.rejected_intentional + self.uncertain_intentional) / total

    @property
    def false_per_hour_upper(self) -> float:
        """Count every unverified emission as a potential false wake."""
        return (self.confirmed_false + self.uncertain_emitted) * 3600 / self.useful_house_seconds

    @property
    def review_coverage(self) -> float:
        """Return reviewed emissions divided by all registered emissions."""
        return self.reviewed_emitted / (self.reviewed_emitted + self.uncertain_emitted)


def complete_delivery(result: ResponseResult, *, audible_end_observed: bool) -> bool:
    """Require both a successful runtime result and full audible output."""
    return result.success and result.output_started is True and audible_end_observed is True


def nearest_rank(values: Sequence[float | None], percentile: int) -> float:
    """Score scheduled attempts; missing endpoints rank after all finite durations."""
    if not values or type(percentile) is not int or percentile not in {50, 95}:
        raise ValueError("unsupported_latency_sample")
    if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                                  or value < 0) for value in values):
        raise ValueError("invalid_latency")
    ordered = sorted(math.inf if value is None else value for value in values)
    return ordered[(percentile * len(ordered) + 99) // 100 - 1]


def clarity_median(ratings: Sequence[int]) -> float:
    """Aggregate one participant's scores on the frozen five-point rubric."""
    if not ratings or any(type(value) is not int or not 1 <= value <= 5 for value in ratings):
        raise ValueError("invalid_clarity_ratings")
    return median(ratings)


@dataclass(frozen=True)
class VoiceAssignment:
    """Human reference and one candidate assignment for a reserve segment."""

    clip_id: str
    speaker_ref: str | None
    group_ref: str | None
    clear: bool
    must_abstain: bool = False


@dataclass(frozen=True)
class PairCounts:
    """Pair counts for clear segments and safety violations across conditions."""

    tp: int
    fp: int
    fn: int
    clear_segments: int
    assigned_clear: int
    mixed_pairs: int
    abstention_violations: int

    @property
    def precision(self) -> float:
        """Return precision over predicted clear pairs; undefined if none."""
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        """Return recall over reference clear pairs; undefined if none."""
        return self.tp / (self.tp + self.fn)

    @property
    def coverage(self) -> float:
        """Return the fraction of eligible clear segments with a group."""
        return self.assigned_clear / self.clear_segments


def voice_pair_counts(assignments: Iterable[VoiceAssignment]) -> PairCounts:
    """Count unordered pairs, with each abstention outside every predicted group."""
    rows = tuple(assignments)
    if any(not _opaque(row.clip_id) or
           (row.speaker_ref is None and not row.must_abstain) or
           (row.speaker_ref is not None and not _opaque(row.speaker_ref)) or
           (row.group_ref is not None and not _opaque(row.group_ref)) or
           type(row.clear) is not bool or type(row.must_abstain) is not bool or
           (row.clear and row.must_abstain) for row in rows) or len({row.clip_id for row in rows}) != len(rows):
        raise ValueError("invalid_voice_assignment")
    tp = fp = fn = mixed = 0
    for a, b in combinations(rows, 2):
        same_speaker = a.speaker_ref == b.speaker_ref
        same_group = a.group_ref is not None and a.group_ref == b.group_ref
        mixed += same_group and a.speaker_ref is not None and b.speaker_ref is not None and not same_speaker
        if a.clear and b.clear:
            tp += same_group and same_speaker
            fp += same_group and not same_speaker
            fn += not same_group and same_speaker
    return PairCounts(tp, fp, fn, sum(row.clear for row in rows),
                      sum(row.clear and row.group_ref is not None for row in rows),
                      mixed, sum(row.must_abstain and row.group_ref is not None for row in rows))
