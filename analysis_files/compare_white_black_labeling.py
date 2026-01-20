import os
import json
import math
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional


ROOT = os.path.dirname(os.path.abspath(__file__))
WHITE_ROOT = os.path.join(ROOT, "white_box_medqa-edited")
BLACK_ROOT = os.path.join(ROOT, "analysis_runs")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm_case(text: str) -> str:
    if text is None:
        return ""
    # replace non-breaking spaces, collapse whitespace
    return " ".join(text.replace("\u00a0", " ").split())


def strip_prompt_wrapper(text: str) -> str:
    """Strip the instruction/template wrapper added by PROMPT_TEMPLATE, if present.

    We look for the CASE PRESENTATION block used in multi_model_pipeline.PROMPT_TEMPLATE
    and extract only the raw case text between the CASE PRESENTATION header and the
    OUTPUT TEMPLATE header.
    """
    if not text:
        return ""

    case_block = "----------------------------------------\nCASE PRESENTATION\n----------------------------------------\n"
    out_block = "\n\n----------------------------------------\nOUTPUT TEMPLATE\n----------------------------------------\n"

    try:
        start = text.index(case_block) + len(case_block)
        end = text.index(out_block, start)
        inner = text[start:end]
        return inner.strip()
    except ValueError:
        # Markers not found; return original text
        return text


def best_alignment_correlation(a: List[int], b: List[int], max_shift: int = 10) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """Kept for reference, but no longer used (we now compare transitions without alignment)."""
    la, lb = len(a), len(b)
    if la < 2 or lb < 2:
        return None, None, None
    best_corr: Optional[float] = None
    best_shift = 0
    best_overlap = 0
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            sa, sb = 0, shift
        else:
            sa, sb = -shift, 0
        overlap = min(la - sa, lb - sb)
        if overlap <= 0:
            continue
        matches = sum(1 for i in range(overlap) if a[sa + i] == b[sb + i])
        corr = matches / overlap
        if best_corr is None or corr > best_corr:
            best_corr = corr
            best_shift = shift
            best_overlap = overlap
    return best_corr, best_shift, best_overlap


def find_other_idx(w_state_to_idx: Dict[str, int], b_state_to_idx: Dict[str, int]) -> Optional[int]:
    for name, idx in w_state_to_idx.items():
        if name.lower().startswith("other"):
            return idx
    for name, idx in b_state_to_idx.items():
        if name.lower().startswith("other"):
            return idx
    return None


def filter_seq(seq: Optional[List[int]], other_idx: Optional[int]) -> bool:
    if seq is None or len(seq) < 5:
        return False
    if other_idx is not None and len(seq) > 0:
        frac_other = sum(1 for x in seq if x == other_idx) / len(seq)
        if frac_other > 0.25:
            return False
    return True


def analyze_model(model: str) -> Dict[str, Any]:
    white_path = os.path.join(WHITE_ROOT, model, "medqa_with_second_responses", "results.labeled.json")
    black_path = os.path.join(BLACK_ROOT, model, "medqa_with_second_responses", "results.labeled.json")

    if not (os.path.isfile(white_path) and os.path.isfile(black_path)):
        return {"model": model, "error": "missing_results"}

    white = load_json(white_path)
    black = load_json(black_path)

    w_states = white.get("state_order")
    b_states = black.get("state_order")
    w_state_to_idx = white.get("state_to_idx", {})
    b_state_to_idx = black.get("state_to_idx", {})

    if not w_states or not b_states:
        return {"model": model, "error": "missing_state_order"}

    n_states = len(w_states)
    other_idx = find_other_idx(w_state_to_idx, b_state_to_idx)

    # index traces by normalized case_prompt (strip generation template if present)
    w_by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in white.get("traces", []):
        raw = t.get("case_prompt", "")
        k = norm_case(strip_prompt_wrapper(raw))
        if k:
            w_by_case[k].append(t)

    b_by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in black.get("traces", []):
        raw = t.get("case_prompt", "")
        k = norm_case(strip_prompt_wrapper(raw))
        if k:
            b_by_case[k].append(t)

    common_cases = sorted(set(w_by_case.keys()) & set(b_by_case.keys()))

    matched_pairs: List[Tuple[List[int], List[int]]] = []
    for k in common_cases:
        wl = sorted(w_by_case[k], key=lambda x: x.get("sample_index", 0))
        bl = sorted(b_by_case[k], key=lambda x: x.get("sample_index", 0))
        m = min(len(wl), len(bl))
        for i in range(m):
            ws = wl[i].get("sequence")
            bs = bl[i].get("sequence")
            if filter_seq(ws, other_idx) and filter_seq(bs, other_idx):
                matched_pairs.append((ws, bs))

    trans_white = [[0.0 for _ in range(n_states)] for _ in range(n_states)]
    trans_black = [[0.0 for _ in range(n_states)] for _ in range(n_states)]

    for ws, bs in matched_pairs:
        for seq, mat in ((ws, trans_white), (bs, trans_black)):
            for i in range(len(seq) - 1):
                a = seq[i]
                b = seq[i + 1]
                # Some pipelines may encode states as lists (multi-label); skip those for transitions
                if isinstance(a, list):
                    if not a:
                        continue
                    a = a[0]
                if isinstance(b, list):
                    if not b:
                        continue
                    b = b[0]
                if 0 <= a < n_states and 0 <= b < n_states:
                    mat[a][b] += 1.0

    # KL over flattened transition distributions (white -> black)
    fw = [x for row in trans_white for x in row]
    fb = [x for row in trans_black for x in row]
    sum_fw, sum_fb = sum(fw), sum(fb)
    if sum_fw == 0 or sum_fb == 0:
        kl_wb = None
        kl_bw = None
    else:
        eps = 1e-8
        Pw = [(x + eps) / (sum_fw + eps * len(fw)) for x in fw]
        Pb = [(x + eps) / (sum_fb + eps * len(fb)) for x in fb]
        kl_wb = sum(p * math.log(p / q) for p, q in zip(Pw, Pb))
        kl_bw = sum(q * math.log(q / p) for p, q in zip(Pw, Pb))

    result: Dict[str, Any] = {
        "model": model,
        "kl_white_to_black": kl_wb,
        "kl_black_to_white": kl_bw,
    }
    return result


def discover_models() -> List[str]:
    models: List[str] = []
    if not os.path.isdir(WHITE_ROOT) or not os.path.isdir(BLACK_ROOT):
        return models
    for model in os.listdir(WHITE_ROOT):
        white_path = os.path.join(WHITE_ROOT, model, "medqa_with_second_responses", "results.labeled.json")
        black_path = os.path.join(BLACK_ROOT, model, "medqa_with_second_responses", "results.labeled.json")
        if os.path.isfile(white_path) and os.path.isfile(black_path):
            models.append(model)
    return sorted(models)


def main() -> None:
    models = discover_models()
    results = [analyze_model(m) for m in models]

    # Prepare nicely formatted text table; only transition-based KL, no alignment-based metrics
    rows = []
    wb_values = []  # collect KL(white→black) across models for a global mean
    for r in results:
        if "error" in r:
            rows.append((r["model"], None, None, True))
        else:
            kl_wb = r["kl_white_to_black"]
            kl_bw = r["kl_black_to_white"]
            rows.append((r["model"], kl_wb, kl_bw, False))
            if kl_wb is not None:
                wb_values.append(kl_wb)

    model_col_width = max(len("Model"), max(len(m) for m, *_ in rows))

    header = (
        f"{'Model':<{model_col_width}}  "
        f"{'KL':>12}  "
    )
    print(header)
    print("-" * (len(header) + 4))

    for model, kl_wb, kl_bw, is_err in rows:
        if is_err:
            kl_wb_s, kl_bw_s = "NA", "NA"
        else:
            kl_wb_s = "NA" if kl_wb is None else f"{kl_wb:.6f}"
        print(
            f"{model:<{model_col_width}}  "
            f"{kl_wb_s:>18}  "
        )

    # Global mean KL across models
    if wb_values:
        mean_wb = sum(wb_values) / len(wb_values)
        print()
        print(f"Mean KL across models: {mean_wb:.6f}")


if __name__ == "__main__":
    main()
