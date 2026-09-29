"""ComfyUI 프롬프트 문법을 저장된 attention 토큰에 연결한다."""

from comfy.sd1_clip import escape_important, token_weights, unescape_important

from anima_heatmap import find_token_spans


def split_phrases(text):
    phrases, current = [], []
    depth = 0
    escaped = False
    for char in text:
        if not escaped and char == "(":
            depth += 1
        elif not escaped and char == ")":
            depth = max(0, depth - 1)
        if char == "\n" or (char == "," and depth == 0 and not escaped):
            if "".join(current).strip():
                phrases.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        escaped = char == "\\" and not escaped
    if "".join(current).strip():
        phrases.append("".join(current).strip())
    return phrases


def resolve_phrase(token_map, phrase, occurrence="all"):
    # 문자 괄호는 보존한다. 정확한 표기가 없을 때 ComfyUI 강조 문법을 적용한다.
    literal = unescape_important(escape_important(phrase))
    parsed = "".join(unescape_important(part) for part, _ in token_weights(escape_important(phrase), 1.0))
    for candidate in dict.fromkeys((literal, parsed)):
        try:
            find_token_spans(token_map, candidate)
        except ValueError:
            continue
        indices = find_token_spans(token_map, candidate, occurrence)
        return indices, candidate
    raise ValueError(f"{phrase!r}을 선택한 프롬프트에서 찾지 못했습니다. "
                     "긍정/부정 선택과 철자를 확인하세요. Text Encode의 token_table을 Preview Any로 볼 수 있습니다.")


def describe_selection(token_map, phrase, indices, matched):
    lines = [f"입력: {phrase}", f"검색 표현: {matched}", f"선택 위치: {','.join(map(str, indices))}"]
    lines.extend(f"  [{i}] id={token_map['ids'][i]} token={token_map['pieces'][i]!r}" for i in indices)
    return "\n".join(lines)
