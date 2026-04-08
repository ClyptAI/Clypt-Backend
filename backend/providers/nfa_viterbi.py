from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

BLANK_TOKEN = "<b>"
V_NEGATIVE_NUM = -3.4e38


def _is_sub_or_superscript_pair(ref_text: str, text: str) -> bool:
    sub_or_superscript_to_num = {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
    return text in sub_or_superscript_to_num and sub_or_superscript_to_num[text] == ref_text


def _restore_token_case(word: str, word_tokens: list[str]) -> list[str]:
    # tokenizer normalization in NeMo can collapse repeated ▁ / _
    while "▁▁" in word:
        word = word.replace("▁▁", "▁")
    while "__" in word:
        word = word.replace("__", "_")

    word_tokens_cased: list[str] = []
    word_char_pointer = 0

    for token in word_tokens:
        token_cased = ""
        for token_char in token:
            if word_char_pointer >= len(word):
                if token_char in {"▁", "_"}:
                    token_cased += token_char
                    continue
                raise RuntimeError(f"Failed to recover token case for word={word!r} token={token!r}")

            if token_char == word[word_char_pointer]:
                token_cased += token_char
                word_char_pointer += 1
                continue

            if token_char.upper() == word[word_char_pointer] or _is_sub_or_superscript_pair(
                token_char, word[word_char_pointer]
            ):
                token_cased += token_char.upper()
                word_char_pointer += 1
                continue

            if token_char in {"▁", "_"}:
                if word[word_char_pointer] in {"▁", "_"}:
                    token_cased += token_char
                    word_char_pointer += 1
                elif word_char_pointer == 0:
                    token_cased += token_char
                continue

            raise RuntimeError(f"Failed to recover token case for word={word!r} token={token!r}")

        word_tokens_cased.append(token_cased)

    return word_tokens_cased


@dataclass
class Token:
    text: str | None = None
    text_cased: str | None = None
    s_start: int | None = None
    s_end: int | None = None
    t_start: float | None = None
    t_end: float | None = None


@dataclass
class Word:
    text: str | None = None
    s_start: int | None = None
    s_end: int | None = None
    t_start: float | None = None
    t_end: float | None = None
    tokens: list[Token] = field(default_factory=list)


@dataclass
class Segment:
    text: str | None = None
    s_start: int | None = None
    s_end: int | None = None
    t_start: float | None = None
    t_end: float | None = None
    words_and_tokens: list[Word | Token] = field(default_factory=list)


@dataclass
class Utterance:
    token_ids_with_blanks: list[int] = field(default_factory=list)
    segments_and_tokens: list[Segment | Token] = field(default_factory=list)
    text: str | None = None
    pred_text: str | None = None
    audio_filepath: str | None = None
    utt_id: str | None = None


def _get_utt_obj(*, text: str, model: Any, T: int, audio_filepath: str) -> Utterance:
    if not hasattr(model, "tokenizer"):
        raise RuntimeError("NFA compatibility path currently requires tokenizer-based NeMo models.")

    blank_id = getattr(model, "blank_id", len(model.tokenizer.vocab))
    utt = Utterance(text=text, audio_filepath=audio_filepath, utt_id=Path(audio_filepath).stem)
    utt.token_ids_with_blanks = [blank_id]

    if len(text) == 0:
        return utt

    all_tokens = model.tokenizer.text_to_ids(text)
    n_token_repetitions = sum(
        1 for i_tok in range(1, len(all_tokens)) if all_tokens[i_tok] == all_tokens[i_tok - 1]
    )
    if len(all_tokens) + n_token_repetitions > T:
        # Not enough acoustic frames for this text.
        return utt

    # Treat full text as a single segment; words are still space-separated.
    utt.segments_and_tokens.append(Token(text=BLANK_TOKEN, text_cased=BLANK_TOKEN, s_start=0, s_end=0))

    segment_tokens = model.tokenizer.text_to_tokens(text)
    segment = Segment(
        text=text,
        s_start=1,
        s_end=1 + len(segment_tokens) * 2 - 2,
    )
    utt.segments_and_tokens.append(segment)

    words = text.split(" ")
    word_s_pointer = 1
    for word_i, word in enumerate(words):
        word_tokens = model.tokenizer.text_to_tokens(word)
        word_token_ids = model.tokenizer.text_to_ids(word)
        word_tokens_cased = _restore_token_case(word, word_tokens)

        word_obj = Word(
            text=word,
            s_start=word_s_pointer,
            s_end=word_s_pointer + len(word_tokens) * 2 - 2,
        )
        word_s_pointer += len(word_tokens) * 2

        for token_i, (token, token_id, token_cased) in enumerate(
            zip(word_tokens, word_token_ids, word_tokens_cased)
        ):
            utt.token_ids_with_blanks.extend([token_id, blank_id])
            word_obj.tokens.append(
                Token(
                    text=token,
                    text_cased=token_cased,
                    s_start=len(utt.token_ids_with_blanks) - 2,
                    s_end=len(utt.token_ids_with_blanks) - 2,
                )
            )
            if token_i < len(word_tokens) - 1:
                word_obj.tokens.append(
                    Token(
                        text=BLANK_TOKEN,
                        text_cased=BLANK_TOKEN,
                        s_start=len(utt.token_ids_with_blanks) - 1,
                        s_end=len(utt.token_ids_with_blanks) - 1,
                    )
                )

        segment.words_and_tokens.append(word_obj)
        if word_i < len(words) - 1:
            segment.words_and_tokens.append(
                Token(
                    text=BLANK_TOKEN,
                    text_cased=BLANK_TOKEN,
                    s_start=len(utt.token_ids_with_blanks) - 1,
                    s_end=len(utt.token_ids_with_blanks) - 1,
                )
            )

    utt.segments_and_tokens.append(
        Token(
            text=BLANK_TOKEN,
            text_cased=BLANK_TOKEN,
            s_start=len(utt.token_ids_with_blanks) - 1,
            s_end=len(utt.token_ids_with_blanks) - 1,
        )
    )
    return utt


def get_single_sample_batch_variables(
    *,
    audio_filepath: str,
    text: str,
    model: Any,
    output_timestep_duration: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[Utterance], float]:
    with torch.no_grad():
        hypotheses = model.transcribe([audio_filepath], return_hypotheses=True, batch_size=1)

    # Hybrid model can return (best_hypotheses, all_hypotheses)
    if isinstance(hypotheses, tuple) and len(hypotheses) == 2:
        hypotheses = hypotheses[0]
    hypothesis = hypotheses[0]

    log_probs_utt = hypothesis.y_sequence
    if not torch.is_tensor(log_probs_utt):
        log_probs_utt = torch.tensor(log_probs_utt)
    log_probs_utt = log_probs_utt.to(dtype=torch.float32)

    t_len = int(log_probs_utt.shape[0])
    utt_obj = _get_utt_obj(text=text, model=model, T=t_len, audio_filepath=audio_filepath)
    y_list = utt_obj.token_ids_with_blanks

    if hasattr(model, "tokenizer"):
        vocab_size = len(model.tokenizer.vocab) + 1
    else:
        vocab_size = len(model.decoder.vocabulary) + 1

    u_len = len(y_list)
    log_probs_batch = V_NEGATIVE_NUM * torch.ones((1, t_len, vocab_size), dtype=torch.float32)
    log_probs_batch[0, :t_len, :] = log_probs_utt

    y_batch = vocab_size * torch.ones((1, u_len), dtype=torch.int64)
    y_batch[0, :u_len] = torch.tensor(y_list, dtype=torch.int64)

    t_batch = torch.tensor([t_len], dtype=torch.int64)
    u_batch = torch.tensor([u_len], dtype=torch.int64)

    if output_timestep_duration is None:
        if "window_stride" not in model.cfg.preprocessor:
            raise ValueError("model.cfg.preprocessor.window_stride is required for forced alignment")
        if "sample_rate" not in model.cfg.preprocessor:
            raise ValueError("model.cfg.preprocessor.sample_rate is required for forced alignment")

        with sf.SoundFile(audio_filepath) as f:
            audio_dur = f.frames / f.samplerate
        n_input_frames = audio_dur / model.cfg.preprocessor.window_stride
        model_downsample_factor = round(n_input_frames / max(1, t_len))
        output_timestep_duration = (
            model.preprocessor.featurizer.hop_length
            * model_downsample_factor
            / model.cfg.preprocessor.sample_rate
        )

    return (
        log_probs_batch,
        y_batch,
        t_batch,
        u_batch,
        [utt_obj],
        float(output_timestep_duration),
    )


def add_t_start_end_to_utt_obj(
    utt_obj: Utterance, alignment_utt: list[int], output_timestep_duration: float
) -> Utterance:
    num_to_first_alignment_appearance: dict[int, int] = {}
    num_to_last_alignment_appearance: dict[int, int] = {}

    prev_s = -1
    for t, s in enumerate(alignment_utt):
        if s > prev_s:
            num_to_first_alignment_appearance[s] = t
            if prev_s >= 0:
                num_to_last_alignment_appearance[prev_s] = t - 1
        prev_s = s
    num_to_last_alignment_appearance[prev_s] = len(alignment_utt) - 1

    for segment_or_token in utt_obj.segments_and_tokens:
        if isinstance(segment_or_token, Segment):
            segment = segment_or_token
            segment.t_start = num_to_first_alignment_appearance[segment.s_start] * output_timestep_duration
            segment.t_end = (num_to_last_alignment_appearance[segment.s_end] + 1) * output_timestep_duration

            for word_or_token in segment.words_and_tokens:
                if isinstance(word_or_token, Word):
                    word = word_or_token
                    word.t_start = num_to_first_alignment_appearance[word.s_start] * output_timestep_duration
                    word.t_end = (num_to_last_alignment_appearance[word.s_end] + 1) * output_timestep_duration

                    for token in word.tokens:
                        token.t_start = (
                            num_to_first_alignment_appearance[token.s_start] * output_timestep_duration
                            if token.s_start in num_to_first_alignment_appearance
                            else -1
                        )
                        token.t_end = (
                            (num_to_last_alignment_appearance[token.s_end] + 1) * output_timestep_duration
                            if token.s_end in num_to_last_alignment_appearance
                            else -1
                        )
                else:
                    token = word_or_token
                    token.t_start = (
                        num_to_first_alignment_appearance[token.s_start] * output_timestep_duration
                        if token.s_start in num_to_first_alignment_appearance
                        else -1
                    )
                    token.t_end = (
                        (num_to_last_alignment_appearance[token.s_end] + 1) * output_timestep_duration
                        if token.s_end in num_to_last_alignment_appearance
                        else -1
                    )
        else:
            token = segment_or_token
            token.t_start = (
                num_to_first_alignment_appearance[token.s_start] * output_timestep_duration
                if token.s_start in num_to_first_alignment_appearance
                else -1
            )
            token.t_end = (
                (num_to_last_alignment_appearance[token.s_end] + 1) * output_timestep_duration
                if token.s_end in num_to_last_alignment_appearance
                else -1
            )

    return utt_obj


def viterbi_decoding(
    log_probs_batch: torch.Tensor,
    y_batch: torch.Tensor,
    t_batch: torch.Tensor,
    u_batch: torch.Tensor,
    viterbi_device: torch.device,
) -> list[list[int]]:
    bsz, t_max, _ = log_probs_batch.shape
    u_max = y_batch.shape[1]

    log_probs_batch = log_probs_batch.to(viterbi_device)
    y_batch = y_batch.to(viterbi_device)
    t_batch = t_batch.to(viterbi_device)
    u_batch = u_batch.to(viterbi_device)

    padding_for_log_probs = V_NEGATIVE_NUM * torch.ones((bsz, t_max, 1), device=viterbi_device)
    log_probs_padded = torch.cat((log_probs_batch, padding_for_log_probs), dim=2)

    v_prev = V_NEGATIVE_NUM * torch.ones((bsz, u_max), device=viterbi_device)
    v_prev[:, :2] = torch.gather(input=log_probs_padded[:, 0, :], dim=1, index=y_batch[:, :2])

    backpointers_rel = -99 * torch.ones((bsz, t_max, u_max), dtype=torch.int8, device=viterbi_device)

    y_shifted_left = torch.roll(y_batch, shifts=2, dims=1)
    letter_repetition_mask = (y_batch - y_shifted_left) == 0
    letter_repetition_mask[:, :2] = False

    for t in range(1, t_max):
        e_current = torch.gather(input=log_probs_padded[:, t, :], dim=1, index=y_batch)

        t_exceeded_t_batch = t >= t_batch
        u_can_be_final = torch.logical_or(
            torch.arange(0, u_max, device=viterbi_device).unsqueeze(0) == (u_batch.unsqueeze(1) - 0),
            torch.arange(0, u_max, device=viterbi_device).unsqueeze(0) == (u_batch.unsqueeze(1) - 1),
        )
        mask = torch.logical_not(torch.logical_and(t_exceeded_t_batch.unsqueeze(1), u_can_be_final)).long()
        e_current = e_current * mask

        v_prev_shifted = torch.roll(v_prev, shifts=1, dims=1)
        v_prev_shifted[:, 0] = V_NEGATIVE_NUM

        v_prev_shifted2 = torch.roll(v_prev, shifts=2, dims=1)
        v_prev_shifted2[:, :2] = V_NEGATIVE_NUM
        v_prev_shifted2.masked_fill_(letter_repetition_mask, V_NEGATIVE_NUM)

        v_prev_dup = torch.cat(
            (v_prev.unsqueeze(2), v_prev_shifted.unsqueeze(2), v_prev_shifted2.unsqueeze(2)), dim=2
        )
        candidates_v_current = v_prev_dup + e_current.unsqueeze(2)
        v_prev, bp_relative = torch.max(candidates_v_current, dim=2)
        backpointers_rel[:, t, :] = bp_relative

    alignments_batch: list[list[int]] = []
    for b in range(bsz):
        t_b = int(t_batch[b])
        u_b = int(u_batch[b])

        if u_b == 1:
            current_u = 0
        else:
            current_u = int(torch.argmax(v_prev[b, u_b - 2 : u_b])) + u_b - 2

        alignment_b = [current_u]
        for t in range(t_max - 1, 0, -1):
            current_u = current_u - int(backpointers_rel[b, t, current_u])
            alignment_b.insert(0, current_u)

        alignments_batch.append(alignment_b[:t_b])

    return alignments_batch


__all__ = [
    "Segment",
    "Word",
    "add_t_start_end_to_utt_obj",
    "get_single_sample_batch_variables",
    "viterbi_decoding",
]
