#!/usr/bin/env python3
"""
ForcedAligner 後處理腳本

將 Qwen3-ForcedAligner 的簡體字元級輸出轉換為：
  - 各 section 起訖時間 + durationInFrames
  - 詞級字幕 timestamps（繁體中文）

用法：
  python process_alignment.py \
    --alignment alignment.txt \
    --script script.txt \
    [--fps 30] \
    [--output timing.json]

輸入格式：
  alignment.txt: ForcedAligner 輸出，每行 [start - end] 字
  script.txt:    繁體講稿，含 [SECTION] markers

輸出格式：
  {
    "sections": [{"marker", "startMs", "endMs", "durationInFrames"}],
    "captions": [{"text", "startMs", "endMs"}],
    "totalDurationMs": ...,
    "totalDurationFrames": ...
  }
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Chinese punctuation + standard punctuation + whitespace
PUNCTUATION = set(
    "，、。！？：；「」『』（）【】《》〈〉…—～·"
    ",.!?:;\"'()[]{}!?。，"
    "\n\r\t "
)


@dataclass
class AlignedChar:
    char: str       # simplified char from aligner
    start: float    # seconds
    end: float      # seconds
    trad_char: str = ""


@dataclass
class Section:
    marker: str
    text: str           # original text (with punctuation)
    pure_text: str      # no punctuation / whitespace
    char_start: int     # index in global pure text
    char_end: int       # exclusive


# ── Parsing ──────────────────────────────────────────────

def parse_aligner_output(text: str) -> list[AlignedChar]:
    """Parse '[start - end] char' format, one per line."""
    pattern = r"\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s*(.)"
    chars = []
    for m in re.finditer(pattern, text):
        chars.append(AlignedChar(
            char=m.group(3),
            start=float(m.group(1)),
            end=float(m.group(2)),
        ))
    return chars


def strip_punctuation(text: str) -> str:
    return "".join(c for c in text if c not in PUNCTUATION)


def parse_script(text: str) -> tuple[list[Section], str]:
    """Parse script with [MARKER] section headers.
    Returns (sections, global_pure_text).
    """
    marker_re = r"\[([^\]]+)\]"
    markers = list(re.finditer(marker_re, text))

    if not markers:
        pure = strip_punctuation(text)
        return [Section("FULL", text.strip(), pure, 0, len(pure))], pure

    sections: list[Section] = []
    global_pure = ""

    for i, m in enumerate(markers):
        name = m.group(1)
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        raw = text[start:end].strip()
        pure = strip_punctuation(raw)

        char_start = len(global_pure)
        global_pure += pure
        sections.append(Section(name, raw, pure, char_start, char_start + len(pure)))

    return sections, global_pure


# ── Processing ───────────────────────────────────────────

def deduplicate(chars: list[AlignedChar]) -> list[AlignedChar]:
    """Remove duplicate segments (detected by timestamp regression)."""
    if not chars:
        return chars

    result = [chars[0]]
    furthest_end = chars[0].end

    for c in chars[1:]:
        if c.start >= furthest_end:
            result.append(c)
            furthest_end = c.end
        else:
            # timestamp went backwards → duplicate segment, skip
            furthest_end = max(furthest_end, c.end)

    return result


def map_traditional(
    chars: list[AlignedChar], pure_trad: str
) -> list[AlignedChar]:
    """Map traditional characters onto aligned chars by index position."""
    if len(chars) != len(pure_trad):
        print(
            f"WARNING: 字數不符 — aligner {len(chars)} 字 vs 講稿 {len(pure_trad)} 字",
            file=sys.stderr,
        )
        # show first mismatch for debugging
        for i in range(min(len(chars), len(pure_trad))):
            chars[i].trad_char = pure_trad[i]
        min_len = min(len(chars), len(pure_trad))
        if len(chars) > min_len:
            print(
                f"  aligner 多出: {''.join(c.char for c in chars[min_len:min_len+20])}",
                file=sys.stderr,
            )
        else:
            print(f"  講稿多出: {pure_trad[min_len:min_len+20]}", file=sys.stderr)
        return chars[:min_len]

    for i, c in enumerate(chars):
        c.trad_char = pure_trad[i]
    return chars


def build_sections_timing(
    sections: list[Section],
    chars: list[AlignedChar],
    fps: int,
) -> list[dict]:
    """Build contiguous section timings (no gaps between sections)."""
    results = []

    for i, sec in enumerate(sections):
        if sec.char_start >= len(chars):
            break
        sec_chars = chars[sec.char_start : min(sec.char_end, len(chars))]
        if not sec_chars:
            continue

        # Start: first char of this section (or 0 for the first section)
        start_ms = 0 if i == 0 else round(sec_chars[0].start * 1000)

        # End: first char of NEXT section, or last char's end
        if i + 1 < len(sections) and sections[i + 1].char_start < len(chars):
            next_start = chars[sections[i + 1].char_start].start
            end_ms = round(next_start * 1000)
        else:
            end_ms = round(sec_chars[-1].end * 1000)

        duration_frames = max(1, round((end_ms - start_ms) / 1000 * fps))

        results.append({
            "marker": sec.marker,
            "startMs": start_ms,
            "endMs": end_ms,
            "durationMs": end_ms - start_ms,
            "durationInFrames": duration_frames,
        })

    return results


def _split_by_punctuation(text: str) -> list[str]:
    """Split text into phrases at punctuation boundaries.
    Returns list of non-empty phrases (punctuation stripped).
    """
    phrases = re.split(r"[，、。！？：；「」『』（）【】《》〈〉…—～,.!?:;\"'()\[\]{}\s]+", text)
    return [p for p in phrases if p]


_REAL_ESTATE_WORDS = [
    "採光", "通風", "落地窗", "衛浴", "更衣間", "中島", "乾濕分離",
    "坪數", "公設比", "管理費", "車位", "平面車位", "機械車位",
    "電梯大樓", "透天", "華廈", "公寓", "套房",
    "高鐵站", "捷運站", "國道",
    "毛坯", "精裝", "裝潢", "翻新", "整理",
    "虛擬裝潢", "開價", "售價", "降價", "成交價",
    "不動產", "房仲", "信義房屋", "永慶不動產",
    "採光充足", "格局方正", "視野開闊", "交通方便",
    "生活機能", "雙面採光",
]


def build_captions(
    sections: list[Section],
    chars: list[AlignedChar],
) -> list[dict]:
    """Use jieba word segmentation to build word-level captions.

    Strategy: split section text by punctuation FIRST, then run jieba on
    each phrase independently. This prevents cross-sentence merging
    (e.g., "萬電梯" or "五三零五永慶").
    """
    import jieba

    # Suppress jieba's loading messages
    jieba.setLogLevel(20)

    for w in _REAL_ESTATE_WORDS:
        jieba.add_word(w)

    captions: list[dict] = []

    for sec in sections:
        if sec.char_start >= len(chars):
            break
        sec_chars = chars[sec.char_start : min(sec.char_end, len(chars))]
        if not sec_chars:
            continue

        # Split original text (with punctuation) into phrases
        phrases = _split_by_punctuation(sec.text)
        char_idx = 0  # position within sec_chars

        for phrase in phrases:
            phrase_len = len(phrase)
            if char_idx + phrase_len > len(sec_chars):
                break

            # Run jieba on each phrase independently
            words = list(jieba.cut(phrase))
            word_offset = 0

            for word in words:
                wlen = len(word)
                if char_idx + word_offset + wlen > len(sec_chars):
                    break

                word_chars = sec_chars[char_idx + word_offset : char_idx + word_offset + wlen]
                trad_word = "".join(c.trad_char for c in word_chars)

                captions.append({
                    "text": trad_word,
                    "startMs": round(word_chars[0].start * 1000),
                    "endMs": round(word_chars[-1].end * 1000),
                })
                word_offset += wlen

            char_idx += phrase_len

    return captions


# ── Main ─────────────────────────────────────────────────

def process(
    aligner_text: str,
    script_text: str,
    fps: int = 30,
) -> dict:
    # 1. Parse
    raw_chars = parse_aligner_output(aligner_text)
    sections, pure_trad = parse_script(script_text)

    print(f"Aligner 原始字數: {len(raw_chars)}", file=sys.stderr)

    # 2. Dedup
    chars = deduplicate(raw_chars)
    removed = len(raw_chars) - len(chars)
    if removed:
        print(f"去重移除: {removed} 字", file=sys.stderr)
    print(f"去重後字數: {len(chars)}", file=sys.stderr)
    print(f"講稿純文字: {len(pure_trad)} 字", file=sys.stderr)

    # 3. Map traditional chars
    chars = map_traditional(chars, pure_trad)

    # 4. Build section timings
    section_timings = build_sections_timing(sections, chars, fps)

    # 5. Build captions
    captions = build_captions(sections, chars)

    # 6. Summary
    total_ms = round(chars[-1].end * 1000) if chars else 0
    total_frames = round(total_ms / 1000 * fps)

    return {
        "sections": section_timings,
        "captions": captions,
        "totalDurationMs": total_ms,
        "totalDurationFrames": total_frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ForcedAligner 後處理：簡體字元級 → 繁體詞級 + section 時長"
    )
    parser.add_argument("--alignment", required=True, help="ForcedAligner 輸出檔")
    parser.add_argument("--script", required=True, help="原始講稿（含 [SECTION] markers）")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", default="-", help="輸出 JSON（- = stdout）")
    args = parser.parse_args()

    aligner_text = Path(args.alignment).read_text(encoding="utf-8")
    script_text = Path(args.script).read_text(encoding="utf-8")

    result = process(aligner_text, script_text, args.fps)

    output_json = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output == "-":
        sys.stdout.reconfigure(encoding="utf-8")
        print(output_json)
    else:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"✓ 輸出已寫入 {args.output}", file=sys.stderr)

    # Print summary
    print(f"\n=== 摘要 ===", file=sys.stderr)
    print(f"總時長: {result['totalDurationMs']}ms ({result['totalDurationFrames']} frames)", file=sys.stderr)
    print(f"Sections: {len(result['sections'])}", file=sys.stderr)
    for s in result["sections"]:
        dur_s = s["durationMs"] / 1000
        print(f"  [{s['marker']}] {dur_s:.1f}s = {s['durationInFrames']} frames", file=sys.stderr)
    print(f"Captions: {len(result['captions'])} 個詞", file=sys.stderr)


if __name__ == "__main__":
    main()
