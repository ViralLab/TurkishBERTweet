import json
import os

dir = os.path.dirname(__file__)
EMOJI_DATA_PATH = os.path.join(dir, "emojis_tr_twitter.json")

with open(EMOJI_DATA_PATH, "r") as f:
    emojis = json.load(f)

_SEARCH_TREE = None


def _get_search_tree():
    global _SEARCH_TREE
    if _SEARCH_TREE is None:
        _SEARCH_TREE = {}
        for emj in emojis:
            sub_tree = _SEARCH_TREE
            lastidx = len(emj) - 1
            for i, char in enumerate(emj):
                if char not in sub_tree:
                    sub_tree[char] = {}
                sub_tree = sub_tree[char]
                if i == lastidx:
                    sub_tree["data"] = emojis[emj]

    return _SEARCH_TREE


def demojize(
    string,
    delimiters=("<emoji> ", " </emoji>"),
    language="tr",
    version=None,
    handle_version=None,
):
    if language == "alias":
        language = "tr"
        _use_aliases = True
    else:
        _use_aliases = False
    tree = _get_search_tree()
    result = []
    i = 0
    length = len(string)
    while i < length:
        consumed = False
        char = string[i]
        if char in tree:
            j = i + 1
            sub_tree = tree[char]
            while j < length and string[j] in sub_tree:
                sub_tree = sub_tree[string[j]]
                j += 1
            if "data" in sub_tree:
                emj_data = sub_tree["data"]
                code_points = string[i:j]
                replace_str = None
                if version is not None and emj_data["E"] > version:
                    if callable(handle_version):
                        emj_data = emj_data.copy()
                        emj_data["match_start"] = i
                        emj_data["match_end"] = j
                        replace_str = handle_version(code_points, emj_data)
                    elif handle_version is not None:
                        replace_str = str(handle_version)
                    else:
                        replace_str = None
                elif language in emj_data:
                    if _use_aliases and "alias" in emj_data:
                        replace_str = (
                            delimiters[0] + emj_data["alias"][0][:-1] + delimiters[1]
                        )
                    else:
                        replace_str = (
                            delimiters[0] + emj_data[language][1:-1] + delimiters[1]
                        )
                else:
                    # The emoji exists, but it is not translated, so we keep the emoji
                    replace_str = code_points

                i = j - 1
                consumed = True
                if replace_str:
                    result.append(replace_str)

        if not consumed and char != "\ufe0e" and char != "\ufe0f":
            result.append(char)
        i += 1

    return "".join(result)
