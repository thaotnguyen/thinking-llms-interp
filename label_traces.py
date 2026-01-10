import os
import json
import time
from string import Template
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import numpy as np
from tqdm import tqdm

# OpenAI SDK works with DeepSeek via base_url
try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit("Please install openai>=1.40.0. pip install -r requirements.txt")

PROMPT_TEMPLATE = (
    """
You are an expert in interpreting how language models solve medical diagnostic cases using multi-step reasoning. Your task is to analyze a Chain-of-Thought (CoT) reasoning trace, break it up into sentences, and label each sentence with:

1. **function_tags**: One or more labels that describe what this sentence is *doing* functionally in the reasoning process.

2. **depends_on**: A list of earlier sentence indices that this sentence directly depends on — meaning it uses information, results, or logic introduced in those earlier sentences.

This annotation will be used to build a dependency graph and perform causal analysis, so please be precise and conservative: only mark a sentence as dependent on another if its reasoning clearly uses a previous step's result or idea.

---

### Function Tags (you may assign multiple per sentence if appropriate):

1. initialization: Openers or setup of the reasoning process (e.g., "I'll think step by step", outlining the approach). 
2. case_setup: Parsing, restating, or structuring the case details (initial reading/comprehension of the vignette). 
3. hypothesis_generation: Proposing diagnostic hypotheses or differential items without yet weighing evidence. 
4. hypothesis_weighing: Weighing or comparing hypotheses using available evidence; synthesizing and narrowing options; self-checks and re-evaluations included here. 
5. stored_medical_knowledge: Recalling general medical facts, disease associations, or epidemiology not specific to a single case detail. 
6. case_interpretation: Interpreting facts of the case, and not just restating them or connecting them to potential diagnosis. 
7. prompting_reconsideration: Issues a metacognitive prompt to pause, widen the differential, or redirect the analysis without yet proposing a concrete alternative hypothesis. 
8. diagnostic_commitment: Committing to a specific leading diagnosis. 
9. final_answer_emission: Formatting the final diagnosis into the desired answer format, i.e. "Now, for the output, I need to provide internal reasoning within <think> tags and the final diagnosis within <answer> tags."
10. other: Use only if the sentence does not fit any of the above tags or is purely stylistic. 

--- 

### depends_on Instructions:

For each sentence, include a list of earlier sentence indices that the reasoning in this sentence *uses*. For example:
- If sentence 9 determines that a differential diagnosis fits the case based on a case interpretation in sentence 4 and a recalled medical fact in sentence 5, then `depends_on: [4, 5]`
- If sentence 24 verifies a medical fact from sentence 23, then `depends_on: [23]`
- If there's no clear dependency (e.g. a general plan or recall), use an empty list: `[]`
- If sentence 13 is based on information in sentence 11, which in turn uses information from sentence 7, then `depends_on: [11, 7]`

Important Notes:
- Make sure to include all dependencies for each sentence. 
- Include both long-range and short-range dependencies.
- Do NOT forget about long-range dependencies. 
- Try to be as comprehensive as possible.
- Make sure there is always a path from earlier sentences (e.g. problem_setup and/or medical interpretation/synthesis) to the final answer.

---

### Output Format: Return a single dictionary with one entry per sentence, where each entry has: - the sentence index (as the key, converted to a string), - a dictionary with: - "function_tags": list of tag strings. Here's the expected format:

{{
    "1": {{
    "text": "First, I need to read the case presentation carefully and identify key details to arrive at the most likely diagnosis.",
    "function_tags": ["initialization"],
    "depends_on": []
    }},
    "2": {{
    "text": "The patient is a 38-year-old Sudanese woman with a 1-year history of a progressively enlarging swelling in the left upper thigh and hip.",
    "function_tags": ["case_setup"],
    "depends_on": []
    }},
    "3": {{
    "text": "It started small, and she deferred intervention until mild discomfort brought her back.",
    "function_tags": ["case_setup"],
    "depends_on": ["2"]
    }}, ...(cont.) 
    "20": {{
    "text": "However, synovial sarcoma is typically malignant and might show more aggressive features.",
    "function_tags": ["stored_medical_knowledge"],
    "depends_on": ["19"]
    }},
    "21": {{
    "text": "Here, the mass is well-circumscribed, and there's no evidence of bony destruction or malignancy in the lymph node biopsy.",
    "function_tags": ["hypothesis_weighing"],
    "depends_on": ["4", "12", "15", "19", "20"]
    }},
    "22": {{
    "text": "Another possibility is myositis ossificans, but that usually follows trauma and has a zonal pattern of calcification, not typically "chicken-wire."",
    "function_tags": ["hypothesis_generation", "hypothesis_weighing"],
    "depends_on": ["11", "16"]
    }}, ...(cont.)
    "73": {{
    "text": "Perhaps it's a calcifying aponeurotic fibroma, but that's more common in hands and feet.",
    "function_tags": ["hypothesis_generation", "hypothesis_weighing"],
    "depends_on": ["4", "70"]
    }},
    "74": {{
    "text": "I think synovial sarcoma is the best bet.",
    "function_tags": ["diagnostic_commitment"],
    "depends_on": ["70", "71", "72", "73"]
    }},
    "75": {{
    "text": "So, my internal reasoning points to synovial sarcoma.",
    "function_tags": ["diagnostic_commitment"],
    "depends_on": ["74"]
    }}
}}

Here is the medical case:

[PROBLEM]
{case_prompt}

Here is the full Chain of Thought:

[REASONING TRACE]
{reasoning_trace}

Now label each sentence with function tags and dependencies.
"""
).strip()

TAG_COLORS = {
    # Canonical long-name categories
    "initialization": "#f9cb9c",
    "case_setup": "#fce5cd",
    "hypothesis_generation": "#fff2cc",
    "hypothesis_weighing": "#d0e0e3",
    "stored_medical_knowledge": "#d9ead3",
    "case_interpretation": "#F4CCCC",
    "diagnostic_commitment": "#cfe2f3",
    "final_answer_emission": "#ead1dc",
    "other": "#EFEFEF",
    # Back-compat: short codes/legacy keys map to same colors
    "IN": "#f9cb9c",
    "PO": "#f9cb9c",  # mapped to initialization
    "DA": "#fce5cd",  # mapped to case_setup
    "HG": "#fff2cc",
    "HT": "#d0e0e3",  # mapped to hypothesis_weighing
    "SK": "#d9ead3",  # mapped to stored_medical_knowledge
    "DC": "#cfe2f3",
    "FA": "#ead1dc",
    "OT": "#EFEFEF",
}


def build_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY environment variable.")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return client


# Thread-local client to avoid creating too many connections/file descriptors
_TLS = threading.local()


def get_thread_client() -> OpenAI:
    cli = getattr(_TLS, "client", None)
    if cli is None:
        cli = build_client()
        _TLS.client = cli
    return cli


def call_deepseek(client: Optional[OpenAI], case_prompt: str, reasoning_trace: str, model: str = "deepseek-chat", retries: int = 3, sleep: float = 2.0, jitter: float = 0.25, max_tokens: int = 8000) -> Optional[Dict[str, Any]]:
    # Build content using the simplified function-tags-only prompt (not the DAG prompt)
    content = PROMPT_TEMPLATE.format(case_prompt=case_prompt, reasoning_trace=reasoning_trace)
    for attempt in range(retries):
        try:
            if client is None:
                client = get_thread_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": content},
                ],
                max_tokens=max_tokens,
                stream=False,
            )
            text = resp.choices[0].message.content or ""
            text_stripped = text.strip()

            # Remove markdown code fences if present
            if text_stripped.startswith("```"):
                # Drop first line fence
                first_nl = text_stripped.find("\n")
                if first_nl != -1:
                    text_stripped = text_stripped[first_nl + 1 :]
                # Drop trailing fence
                if text_stripped.endswith("```"):
                    text_stripped = text_stripped[:-3]

            # Try direct JSON
            try:
                return json.loads(text_stripped)
            except Exception:
                print("Failed to parse JSON directly")
                pass

            # Fallback: extract the largest {...} JSON object
            import re
            matches = list(re.finditer(r"\{[\s\S]*\}", text_stripped))
            if matches:
                candidate = matches[0].group(0)
                try:
                    return json.loads(candidate)
                except Exception:
                    pass
            # Give up for this attempt
            raise ValueError("DeepSeek response could not be parsed as JSON")
        except Exception as e:
            # Handle too many open files with additional backoff
            print(e)
            try:
                import errno
                if isinstance(e, OSError) and getattr(e, 'errno', None) == errno.EMFILE:
                    # exponential backoff on EMFILE
                    time.sleep(3.0 * (attempt + 1))
                    continue
            except Exception:
                pass
            if attempt == retries - 1:
                print(f"DeepSeek call failed after {retries} attempts: {e}")
                return None
            # normal backoff + jitter
            import random
            time.sleep(sleep * (attempt + 1) + random.uniform(0, jitter))
    return None


def normalize_labels(obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Ensure each value has keys: text (str|None), function_tags (List[str]).
    Accepts cases where function_tags is a string; converts to list. Keys must be strings.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in obj.items():
        key = str(k)
        if not isinstance(v, dict):
            v = {"text": None, "function_tags": v}
        text = v.get("text")
        tags = v.get("function_tags", [])
        if isinstance(tags, str):
            tags = [tags]
        # Map all inputs to canonical long-name categories used in the new prompt
        allowed_to_canonical = {
            # Canonical long names
            "initialization": "initialization",
            "case_setup": "case_setup",
            "hypothesis_generation": "hypothesis_generation",
            "hypothesis_weighing": "hypothesis_weighing",
            "stored_medical_knowledge": "stored_medical_knowledge",
            "case_interpretation": "case_interpretation",
            "diagnostic_commitment": "diagnostic_commitment",
            "final_answer_emission": "final_answer_emission",
            "other": "other",
            # Older long names -> canonical
            "process_organization": "initialization",
            "data_acquisition": "case_setup",
            "stored_knowledge": "stored_medical_knowledge",
            "hypothesis_testing": "hypothesis_weighing",
            "diagnosis": "diagnostic_commitment",
            # Short codes -> canonical
            "IN": "initialization", "INIT": "initialization", "init": "initialization",
            "PO": "initialization",
            "DA": "case_setup",
            "HG": "hypothesis_generation",
            "HT": "hypothesis_weighing",
            "SK": "stored_medical_knowledge",
            "DC": "diagnostic_commitment",
            "FA": "final_answer_emission",
            "OT": "other",
            # Older short codes
            "EG": "case_setup",  # evidence_gathering
            "ES": "hypothesis_weighing",  # evidence_synthesis
            "WD": "hypothesis_weighing",  # weighing data
            "MF": "stored_medical_knowledge",
        }
        alias_map = {
            # legacy long-form aliases
            "evidence_gathering": "case_setup",
            "problem_setup": "case_setup",
            "plan_generation": "initialization",
            "fact_retrieval": "stored_medical_knowledge",
            "stored knowledge": "stored_medical_knowledge",
            "medical knowledge": "stored_medical_knowledge",
            "knowledge": "stored_medical_knowledge",
            "evidence_synthesis": "hypothesis_weighing",
            "synthesis": "hypothesis_weighing",
            "weighing": "hypothesis_weighing",
            "hypothesis testing": "hypothesis_weighing",
            "testing": "hypothesis_weighing",
            "diagnostic commitment": "diagnostic_commitment",
            # final answer variants
            "final answer emission": "final_answer_emission",
            "final_answer_emission": "final_answer_emission",
            "final answer": "final_answer_emission",
            "final_answer": "final_answer_emission",
            "answer emission": "final_answer_emission",
            "answer_emission": "final_answer_emission",
            # legacy medical facts -> stored_medical_knowledge
            "medical facts": "stored_medical_knowledge",
            "medical_facts": "stored_medical_knowledge",
            "medicalfacts": "stored_medical_knowledge",
        }
        norm_tags: List[str] = []
        for t in tags:
            canonical: Optional[str] = None
            if isinstance(t, str):
                t_str = t.strip()
                t_low = t_str.lower()
                t_key_variants = [
                    t_str,
                    t_low,
                    t_str.replace(" ", "_"),
                    t_low.replace(" ", "_"),
                    t_str.replace("-", "_"),
                    t_low.replace("-", "_"),
                ]
                for key in t_key_variants:
                    if key in allowed_to_canonical:
                        canonical = allowed_to_canonical[key]
                        break
                    if key in alias_map:
                        canonical = alias_map[key]
                        break
            elif t in allowed_to_canonical:  # non-string but hashable present
                canonical = allowed_to_canonical.get(t)  # type: ignore[arg-type]

            if canonical is None:
                # Fallback to 'other' if unknown
                canonical = "other"

            if canonical not in norm_tags:
                norm_tags.append(canonical)
        
        # Preserve depends_on field (list of dependency indices as strings)
        depends_on = v.get("depends_on", [])
        if not isinstance(depends_on, list):
            depends_on = []
        # Normalize all elements to strings
        depends_on_normalized = [str(d) for d in depends_on]
        
        out[key] = {"text": text, "function_tags": norm_tags, "depends_on": depends_on_normalized}
    return out


def label_csv(
    csv_path: str = "results.csv",
    out_jsonl: str = "labeled_traces.jsonl",
    out_csv: str = "results.labeled.csv",
    out_html: Optional[str] = None,
    cache_dir: str = ".cache/deepseek_labels",
    resume: bool = True,
    limit: Optional[int] = None,
    workers: int = 4,
    model: str = "deepseek-chat",
    retries: int = 3,
    retry_sleep: float = 2.0,
    retry_jitter: float = 0.25,
    sample_per_accuracy: Optional[int] = None,
    seed: int = 42,
    sample_per_case_accuracy_traces: Optional[int] = None,
    case_accuracy_levels: Optional[str] = "all",
    regenerate_html_only: bool = False,
    max_tokens: int = 8000,
):
    os.makedirs(cache_dir, exist_ok=True)
    
    # If regenerate_html_only, load the labeled CSV and regenerate HTML
    if regenerate_html_only:
        if not os.path.exists(out_csv):
            raise FileNotFoundError(f"Cannot regenerate HTML: {out_csv} does not exist. Run without --regenerate_html_only first.")
        print(f"Loading labeled data from {out_csv}...")
        out_df = pd.read_csv(out_csv)
        print(f"Loaded {len(out_df)} labeled traces.")
        
        # Generate sentence-level CSV and HTML from loaded data
        recs: List[Dict[str, Any]] = []
        html_cases: List[Dict[str, Any]] = []

        def _sentence_sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, Any]:
            idx, _ = item
            try:
                return (0, int(idx))
            except Exception:
                return (1, str(idx))

        for _, r in out_df.iterrows():
            sentence_records: List[Dict[str, Any]] = []
            if "label_json" in r and pd.notna(r["label_json"]):
                label_obj = json.loads(r["label_json"]) if isinstance(r["label_json"], str) else r["label_json"]
                # Normalize tags (convert legacy names/codes to canonical)
                label_obj = normalize_labels(label_obj)
                ordered_sentences = sorted(label_obj.items(), key=_sentence_sort_key)

                for idx_str, entry in ordered_sentences:
                    tags = entry.get("function_tags", [])
                    primary = None
                    if isinstance(tags, list) and tags:
                        primary = tags[0]
                    recs.append({
                        "pmcid": r["pmcid"],
                        "sample_index": r["sample_index"],
                        "sentence_index": int(idx_str) if str(idx_str).isdigit() else idx_str,
                        "sentence_text": entry.get("text"),
                        "function_tags": ",".join(entry.get("function_tags", [])),
                        "primary_state": primary,
                        "verified_correct": r.get("verified_correct"),
                    })
                    sentence_records.append({
                        "index": str(idx_str),
                        "text": entry.get("text") or "",
                        "tags": entry.get("function_tags", []),
                    })
            else:
                # Fallback: include a single untagged sentence from the reasoning trace if available
                raw_text = str(r.get("reasoning_trace") or "")
                sentence_records.append({
                    "index": "0",
                    "text": raw_text,
                    "tags": [],
                })

            verified_val = r.get("verified_correct")
            if isinstance(verified_val, bool):
                verified_correct = verified_val
            elif pd.notna(verified_val):
                verified_correct = str(verified_val).strip().lower() in {"true", "t", "1", "yes", "y"}
            else:
                verified_correct = None

            sample_idx = r.get("sample_index")
            if pd.notna(sample_idx):
                try:
                    sample_idx_str = str(int(sample_idx))
                except Exception:
                    sample_idx_str = str(sample_idx)
            else:
                sample_idx_str = ""

            html_cases.append({
                "pmcid": str(r.get("pmcid")),
                "sample_index": sample_idx_str,
                "verified_correct": verified_correct,
                "sentences": sentence_records,
            })

        sentences_csv_path = out_csv.replace(".csv", ".sentences.csv")
        if recs:
            pd.DataFrame(recs).to_csv(sentences_csv_path, index=False)
            print(f"Regenerated sentences CSV: {sentences_csv_path}")

        def _generate_html(cases: List[Dict[str, Any]], html_path: str) -> None:
            if not cases:
                print("No cases to generate HTML for.")
                return

            try:
                # Use ensure_ascii=True to avoid encoding issues in HTML
                cases_payload = json.dumps(cases, ensure_ascii=True)
                colors_payload = json.dumps(TAG_COLORS, ensure_ascii=True)
            except Exception as e:
                print(f"Error serializing data for HTML: {e}")
                return

            html_template = Template(
                """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DeepSeek Trace Labels</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; color: #222; }
label { margin-right: 8px; font-weight: bold; }
select { margin-right: 16px; padding: 4px 6px; }
.controls { margin-bottom: 20px; display: flex; align-items: center; flex-wrap: wrap; gap: 12px; }
.legend { margin-bottom: 16px; display: flex; gap: 12px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 6px; }
.color-swatch { width: 24px; height: 16px; border: 1px solid #999; border-radius: 4px; }
#case-meta { margin-bottom: 16px; font-weight: bold; }
#sentences { line-height: 1.6; font-size: 15px; }
.sentence-text { display: inline; padding: 2px 4px; border-radius: 3px; }
</style>
</head>
<body>
<h1>DeepSeek Trace Labels</h1>
<div class="controls">
    <label for="correctness-select">Correctness</label>
    <select id="correctness-select">
        <option value="all">All</option>
        <option value="true">Correct</option>
        <option value="false">Incorrect</option>
    </select>
    <label for="pmcid-select">PMCID</label>
    <select id="pmcid-select"></select>
    <label for="trace-select">Trace</label>
    <select id="trace-select"></select>
</div>
<div class="legend" id="legend"></div>
<div id="case-meta"></div>
<div id="sentences"></div>
<script>
const CASES = $cases_payload;
const TAG_COLORS = $colors_payload;

function escapeHtml(str) {
    return str.replace(/[&<>"']/g, function(tag) {
        const chars = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};
        return chars[tag] || tag;
    });
}

function renderLegend() {
    const legend = document.getElementById('legend');
    legend.innerHTML = '';
    Object.entries(TAG_COLORS).forEach(function(entry) {
        const tag = entry[0];
        const color = entry[1];
        const item = document.createElement('div');
        item.className = 'legend-item';
        const swatch = document.createElement('div');
        swatch.className = 'color-swatch';
        swatch.style.backgroundColor = color;
        const label = document.createElement('span');
        label.textContent = tag;
        item.appendChild(swatch);
        item.appendChild(label);
        legend.appendChild(item);
    });
}

function filteredCases(correctness) {
    if (correctness === 'all') {
        return CASES;
    }
    const expected = correctness === 'true';
    return CASES.filter(function(entry) {
        return entry.verified_correct === expected;
    });
}

function populatePmcidOptions() {
    const correctness = document.getElementById('correctness-select').value;
    const pmcidSelect = document.getElementById('pmcid-select');
    const filtered = filteredCases(correctness);
    const pmcids = Array.from(new Set(filtered.map(function(entry) { return entry.pmcid; }))).sort();
    pmcidSelect.innerHTML = '';
    pmcids.forEach(function(pmcid) {
        const opt = document.createElement('option');
        opt.value = pmcid;
        opt.textContent = pmcid;
        pmcidSelect.appendChild(opt);
    });
    populateTraceOptions();
}

function populateTraceOptions() {
    const correctness = document.getElementById('correctness-select').value;
    const pmcid = document.getElementById('pmcid-select').value;
    const traceSelect = document.getElementById('trace-select');
    const filtered = filteredCases(correctness).filter(function(entry) {
        return entry.pmcid === pmcid;
    });
    traceSelect.innerHTML = '';
    filtered.forEach(function(entry) {
        const option = document.createElement('option');
        const si = (entry.sample_index ?? '');
        option.value = String(si);
        option.textContent = (si !== '') ? ('Trace ' + si) : 'Trace';
        traceSelect.appendChild(option);
    });
    rendersentences();
}

function rendersentences() {
    const correctness = document.getElementById('correctness-select').value;
    const pmcid = document.getElementById('pmcid-select').value;
    const traceSelect = document.getElementById('trace-select');
    const sampleIndex = traceSelect.value;
    const cases = filteredCases(correctness).filter(function(entry) {
        return entry.pmcid === pmcid;
    });
    const activeCase = cases.find(function(entry) {
        return String(entry.sample_index ?? '') === sampleIndex;
    }) || cases[0];
    const meta = document.getElementById('case-meta');
    const container = document.getElementById('sentences');

    if (!activeCase) {
        container.innerHTML = '<em>No matching cases for this selection.</em>';
        meta.textContent = '';
        return;
    }

    const correctnessLabel = activeCase.verified_correct === true ? 'Correct trace' : activeCase.verified_correct === false ? 'Incorrect trace' : 'Correctness unknown';
    const traceLabel = activeCase.sample_index ? 'Trace ' + activeCase.sample_index : 'Trace';
    meta.textContent = traceLabel + ' • ' + correctnessLabel;

    container.innerHTML = '';
    activeCase.sentences.forEach(function(sentence, idx) {
        const textSpan = document.createElement('span');
        textSpan.className = 'sentence-text';
        const primaryTag = (sentence.tags || [])[0];
        if (primaryTag && TAG_COLORS[primaryTag]) {
            textSpan.style.backgroundColor = TAG_COLORS[primaryTag];
        } else {
            textSpan.style.backgroundColor = '#efefef';
        }
        const safeText = escapeHtml(sentence.text || '').replace(/\\n/g, '<br>');
        textSpan.innerHTML = (safeText || '<em>No text</em>');
        container.appendChild(textSpan);
        
        // Add a space between sentences for readability
        if (idx < activeCase.sentences.length - 1) {
            container.appendChild(document.createTextNode(' '));
        }
    });
}

document.getElementById('correctness-select').addEventListener('change', function() {
    populatePmcidOptions();
});

document.getElementById('pmcid-select').addEventListener('change', function() {
    populateTraceOptions();
});

document.getElementById('trace-select').addEventListener('change', function() {
    rendersentences();
});

renderLegend();
populatePmcidOptions();
</script>
</body>
</html>
                """
            )

            html_content = html_template.substitute(
                cases_payload=cases_payload,
                colors_payload=colors_payload,
            )

            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"Generated HTML: {html_path}")

        html_output_path = out_html or out_csv.replace(".csv", ".html")
        _generate_html(html_cases, html_output_path)
        print(f"HTML regeneration complete. Total cases: {len(html_cases)}")
        return
    
    df = pd.read_csv(csv_path)
    needed_cols = ["pmcid", "sample_index", "case_prompt", "true_diagnosis", "predicted_diagnosis", "reasoning_trace", "verification_response", "verified_correct"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # Optional stratified sampling by trace-level accuracy (True/False) — legacy mode
    if sample_per_accuracy is not None and sample_per_case_accuracy_traces is None:
        def _to_bool(x):
            if isinstance(x, bool):
                return x
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            s = str(x).strip().lower()
            if s in ("true", "t", "1", "yes", "y"): return True
            if s in ("false", "f", "0", "no", "n"): return False
            try:
                return float(s) > 0.5
            except Exception:
                return None
        vc = df["verified_correct"].apply(_to_bool)
        df = df[~vc.isna()].copy()
        df["verified_correct"] = vc.values
        parts = []
        for val in [True, False]:
            g = df[df["verified_correct"] == val]
            if len(g) == 0:
                continue
            n = min(len(g), int(sample_per_accuracy))
            parts.append(g.sample(n=n, random_state=seed))
        if parts:
            df = pd.concat(parts, ignore_index=True)
            # Shuffle combined sample for fairness
            df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        else:
            # nothing matched; keep empty to no-op gracefully
            df = df.head(0)

    # Sampling by per-case accuracy level (0..10): pick N traces per level
    if sample_per_case_accuracy_traces is not None:
        # Coerce verified_correct to bool first
        def _to_bool2(x):
            if isinstance(x, bool):
                return x
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            s = str(x).strip().lower()
            if s in ("true", "t", "1", "yes", "y"): return True
            if s in ("false", "f", "0", "no", "n"): return False
            try:
                return float(s) > 0.5
            except Exception:
                return None
        vc = df["verified_correct"].apply(_to_bool2)
        df = df[~vc.isna()].copy()
        df["verified_correct"] = vc.values

        # Compute per-case accuracy count and traces-per-case
        case_correct = df.groupby("pmcid")["verified_correct"].sum(min_count=1).fillna(0).astype(int)
        case_counts = df.groupby("pmcid").size().astype(int)

        # Levels to sample
        if case_accuracy_levels is None or str(case_accuracy_levels).strip().lower() == "all":
            levels = sorted(case_correct.unique().tolist())
        else:
            try:
                levels = [int(x) for x in str(case_accuracy_levels).split(",")]
            except Exception:
                levels = sorted(case_correct.unique().tolist())

        sampled_parts = []
        rng = np.random.default_rng(seed)
        for L in levels:
            pmcids_L = case_correct[case_correct == L].index.tolist()
            if not pmcids_L:
                continue
            rng.shuffle(pmcids_L)
            # Accumulate pmcids until reaching >= target traces
            target = int(sample_per_case_accuracy_traces)
            chosen = []
            total_traces = 0
            for pm in pmcids_L:
                chosen.append(pm)
                total_traces += int(case_counts.get(pm, 0))
                if total_traces >= target:
                    break
            sub = df[df["pmcid"].isin(chosen)].copy()
            # If overshoot, downsample to exactly target
            if len(sub) > target:
                sub = sub.sample(n=target, random_state=seed)
            sampled_parts.append(sub)

        if sampled_parts:
            df = pd.concat(sampled_parts, ignore_index=True)
            df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        else:
            df = df.head(0)

    outputs: List[Dict[str, Any]] = []

    it = df.iterrows()
    if limit is not None:
        it = zip(range(limit), df.head(limit).iterrows())
        it = ((i, row) for i, (_, row) in it)
    else:
        it = ((i, row) for i, (_, row) in enumerate(df.iterrows()))

    # Worker function for parallel processing
    def process_row(idx: int, row: pd.Series) -> Optional[Dict[str, Any]]:
        key = f"{row['pmcid']}__{row['sample_index']}"
        cache_path = os.path.join(cache_dir, f"{key}.json")

        # # Resume from cache file if present
        # if resume and os.path.exists(cache_path):
        #     try:
        #         with open(cache_path, "r") as f:
        #             obj = json.load(f)
        #         labels = normalize_labels(obj)
        #         return {**row.to_dict(), "label_json": json.dumps(labels, ensure_ascii=False)}
        #     except Exception:
        #         pass

        # Randomized small sleep to reduce burstiness when many threads start
        time.sleep(0.05 * (idx % 5))

        obj = call_deepseek(None, str(row["case_prompt"]), str(row["reasoning_trace"]), model=model, retries=retries, sleep=retry_sleep, jitter=retry_jitter, max_tokens=max_tokens)
        if obj is None:
            return None
        # labels = normalize_labels(obj)
        # try:
        #     with open(cache_path, "w") as f:
        #         json.dump(labels, f)
        # except Exception:
        #     # non-fatal
        #     print("Failed to write cache file")
        #     pass
        return {**row.to_dict(), "label_json": json.dumps(obj, ensure_ascii=False)}

    tasks: List[Tuple[int, pd.Series]] = []
    if limit is not None:
        for i, (_, row) in enumerate(df.head(limit).iterrows()):
            tasks.append((i, row))
    else:
        for i, (_, row) in enumerate(df.iterrows()):
            tasks.append((i, row))

    if workers <= 1:
        for i, row in tqdm(tasks, desc="Labeling traces (seq)"):
            out = process_row(i, row)
            if out is not None:
                outputs.append(out)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_row, i, row): i for i, row in tasks}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Labeling traces ({workers} workers)"):
                try:
                    out = fut.result()
                    if out is not None:
                        outputs.append(out)
                except Exception as e:
                    # non-fatal: collect and continue
                    print(f"Error processing row {futs[fut]}: {e}")
                    continue

    out_df = pd.DataFrame(outputs)
    out_df.to_csv(out_csv, index=False)

    # Also write sentence-level long-form CSV
    recs: List[Dict[str, Any]] = []
    html_cases: List[Dict[str, Any]] = []

    def _sentence_sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, Any]:
        idx, _ = item
        try:
            return (0, int(idx))
        except Exception:
            return (1, str(idx))

    for _, r in out_df.iterrows():
        label_obj = json.loads(r["label_json"]) if isinstance(r["label_json"], str) else r["label_json"]
        # Support both dict mapping index->entry and list of entries
        if isinstance(label_obj, dict):
            items_iter = sorted(label_obj.items(), key=_sentence_sort_key)
        elif isinstance(label_obj, list):
            items_iter = [(str(i), entry) for i, entry in enumerate(label_obj)]
        else:
            # Unknown format; skip this row
            continue

        sentence_records: List[Dict[str, Any]] = []
        for idx_str, entry in items_iter:
            # Normalize entry: if it's a list, interpret as [text, tags?] or fallback to text only
            if isinstance(entry, list):
                # Best-effort normalization for list-shaped outputs
                text_val = None
                tags_val: List[str] = []
                if entry:
                    # first element as text if string
                    if isinstance(entry[0], str):
                        text_val = entry[0]
                    # search any list element that is a list of strings for tags
                    for el in entry:
                        if isinstance(el, list) and all(isinstance(t, str) for t in el):
                            tags_val = [str(t) for t in el]
                            break
                entry = {"text": text_val or "", "function_tags": tags_val}
            elif not isinstance(entry, dict):
                # Fallback: convert scalars to dict with text
                entry = {"text": str(entry), "function_tags": []}

            tags = entry.get("function_tags", [])
            primary = None
            if isinstance(tags, list) and tags:
                primary = tags[0]
            recs.append({
                "pmcid": r["pmcid"],
                "sample_index": r["sample_index"],
                "sentence_index": int(idx_str) if str(idx_str).isdigit() else idx_str,
                "sentence_text": entry.get("text"),
                "function_tags": ",".join(entry.get("function_tags", [])),
                "primary_state": primary,
                "verified_correct": r.get("verified_correct"),
            })
            sentence_records.append({
                "index": str(idx_str),
                "text": entry.get("text") or "",
                "tags": entry.get("function_tags", []),
            })

        verified_val = r.get("verified_correct")
        if isinstance(verified_val, bool):
            verified_correct = verified_val
        elif pd.notna(verified_val):
            verified_correct = str(verified_val).strip().lower() in {"true", "t", "1", "yes", "y"}
        else:
            verified_correct = None

        sample_idx = r.get("sample_index")
        if pd.notna(sample_idx):
            try:
                sample_idx_str = str(int(sample_idx))
            except Exception:
                sample_idx_str = str(sample_idx)
        else:
            sample_idx_str = ""

        html_cases.append({
            "pmcid": str(r.get("pmcid")),
            "sample_index": sample_idx_str,
            "verified_correct": verified_correct,
            "sentences": sentence_records,
        })

    sentences_csv_path = out_csv.replace(".csv", ".sentences.csv")
    if recs:
        pd.DataFrame(recs).to_csv(sentences_csv_path, index=False)

    def _generate_html(cases: List[Dict[str, Any]], html_path: str) -> None:
        if not cases:
            return

        try:
            # Use ensure_ascii=True to avoid encoding issues in HTML
            cases_payload = json.dumps(cases, ensure_ascii=True)
            colors_payload = json.dumps(TAG_COLORS, ensure_ascii=True)
        except Exception:
            return

        html_template = Template(
            """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DeepSeek Trace Labels</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; color: #222; }
label { margin-right: 8px; font-weight: bold; }
select { margin-right: 16px; padding: 4px 6px; }
.controls { margin-bottom: 20px; display: flex; align-items: center; flex-wrap: wrap; gap: 12px; }
.legend { margin-bottom: 16px; display: flex; gap: 12px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 6px; }
.color-swatch { width: 24px; height: 16px; border: 1px solid #999; border-radius: 4px; }
#case-meta { margin-bottom: 16px; font-weight: bold; }
#sentences { line-height: 1.6; font-size: 15px; }
.sentence-text { display: inline; padding: 2px 4px; border-radius: 3px; }
</style>
</head>
<body>
<h1>DeepSeek Trace Labels</h1>
<div class="controls">
    <label for="correctness-select">Correctness</label>
    <select id="correctness-select">
        <option value="all">All</option>
        <option value="true">Correct</option>
        <option value="false">Incorrect</option>
    </select>
    <label for="pmcid-select">PMCID</label>
    <select id="pmcid-select"></select>
    <label for="trace-select">Trace</label>
    <select id="trace-select"></select>
</div>
<div class="legend" id="legend"></div>
<div id="case-meta"></div>
<div id="sentences"></div>
<script>
const CASES = $cases_payload;
const TAG_COLORS = $colors_payload;

function escapeHtml(str) {
    return str.replace(/[&<>"']/g, function(tag) {
        const chars = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};
        return chars[tag] || tag;
    });
}

function renderLegend() {
    const legend = document.getElementById('legend');
    legend.innerHTML = '';
    Object.entries(TAG_COLORS).forEach(function(entry) {
        const tag = entry[0];
        const color = entry[1];
        if (tag === 'WD') {
            return;
        }
        const item = document.createElement('div');
        item.className = 'legend-item';
        const swatch = document.createElement('div');
        swatch.className = 'color-swatch';
        swatch.style.backgroundColor = color;
        const label = document.createElement('span');
        label.textContent = tag;
        item.appendChild(swatch);
        item.appendChild(label);
        legend.appendChild(item);
    });
}

function filteredCases(correctness) {
    if (correctness === 'all') {
        return CASES;
    }
    const expected = correctness === 'true';
    return CASES.filter(function(entry) {
        return entry.verified_correct === expected;
    });
}

function populatePmcidOptions() {
    const correctness = document.getElementById('correctness-select').value;
    const pmcidSelect = document.getElementById('pmcid-select');
    const filtered = filteredCases(correctness);
    const pmcids = Array.from(new Set(filtered.map(function(entry) { return entry.pmcid; }))).sort();
    pmcidSelect.innerHTML = '';
    pmcids.forEach(function(pmcid) {
        const opt = document.createElement('option');
        opt.value = pmcid;
        opt.textContent = pmcid;
        pmcidSelect.appendChild(opt);
    });
    populateTraceOptions();
}

function populateTraceOptions() {
    const correctness = document.getElementById('correctness-select').value;
    const pmcid = document.getElementById('pmcid-select').value;
    const traceSelect = document.getElementById('trace-select');
    const filtered = filteredCases(correctness).filter(function(entry) {
        return entry.pmcid === pmcid;
    });
    traceSelect.innerHTML = '';
    filtered.forEach(function(entry) {
        const option = document.createElement('option');
        option.value = entry.sample_index || '';
        option.textContent = entry.sample_index ? 'Trace ' + entry.sample_index : 'Trace';
        traceSelect.appendChild(option);
    });
    rendersentences();
}

function rendersentences() {
    const correctness = document.getElementById('correctness-select').value;
    const pmcid = document.getElementById('pmcid-select').value;
    const traceSelect = document.getElementById('trace-select');
    const sampleIndex = traceSelect.value;
    const cases = filteredCases(correctness).filter(function(entry) {
        return entry.pmcid === pmcid;
    });
    const activeCase = cases.find(function(entry) {
        return entry.sample_index === sampleIndex;
    }) || cases[0];
    const meta = document.getElementById('case-meta');
    const container = document.getElementById('sentences');

    if (!activeCase) {
        container.innerHTML = '<em>No matching cases for this selection.</em>';
        meta.textContent = '';
        return;
    }

    const correctnessLabel = activeCase.verified_correct === true ? 'Correct trace' : activeCase.verified_correct === false ? 'Incorrect trace' : 'Correctness unknown';
    const traceLabel = activeCase.sample_index ? 'Trace ' + activeCase.sample_index : 'Trace';
    meta.textContent = traceLabel + ' • ' + correctnessLabel;

    container.innerHTML = '';
    activeCase.sentences.forEach(function(sentence, idx) {
        const textSpan = document.createElement('span');
        textSpan.className = 'sentence-text';
        const primaryTag = (sentence.tags || [])[0];
        if (primaryTag && TAG_COLORS[primaryTag]) {
            textSpan.style.backgroundColor = TAG_COLORS[primaryTag];
        } else {
            textSpan.style.backgroundColor = '#efefef';
        }
        const safeText = escapeHtml(sentence.text || '').replace(/\n/g, '<br>');
        textSpan.innerHTML = (safeText || '<em>No text</em>');
        container.appendChild(textSpan);
        
        // Add a space between sentences for readability
        if (idx < activeCase.sentences.length - 1) {
            container.appendChild(document.createTextNode(' '));
        }
    });
}

document.getElementById('correctness-select').addEventListener('change', function() {
    populatePmcidOptions();
});

document.getElementById('pmcid-select').addEventListener('change', function() {
    populateTraceOptions();
});

document.getElementById('trace-select').addEventListener('change', function() {
    rendersentences();
});

renderLegend();
populatePmcidOptions();
</script>
</body>
</html>
            """
        )

        html_content = html_template.substitute(
            cases_payload=cases_payload,
            colors_payload=colors_payload,
        )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    html_output_path = out_html or out_csv.replace(".csv", ".html")
    _generate_html(html_cases, html_output_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Label reasoning traces with DeepSeek Chat")
    p.add_argument("--csv", default="results.csv")
    p.add_argument("--out_csv", default="results.labeled.csv")
    p.add_argument("--out_html", default="labeled_traces.html")
    p.add_argument("--out_jsonl", default="labeled_traces.jsonl")
    p.add_argument("--cache_dir", default=".cache/deepseek_labels")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=1000, help="Number of parallel workers (1 for sequential)")
    p.add_argument("--model", type=str, default="deepseek-chat", help="Model name for DeepSeek endpoint")
    p.add_argument("--retries", type=int, default=3, help="API call retries per item")
    p.add_argument("--retry_sleep", type=float, default=2.0, help="Base backoff sleep seconds")
    p.add_argument("--retry_jitter", type=float, default=0.25, help="Random jitter seconds added to backoff")
    p.add_argument("--sample_per_accuracy", type=int, default=None, help="Sample N examples from each accuracy level (verified_correct True/False)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--sample_per_case_accuracy_traces", type=int, default=None, help="Sample N traces for each per-case accuracy level (0..10)")
    p.add_argument("--case_accuracy_levels", type=str, default="all", help="Comma-separated levels (e.g., '0,5,10') or 'all'")
    p.add_argument("--regenerate_html_only", default=False,action="store_true", help="Skip labeling and only regenerate HTML from existing labeled CSV")
    args = p.parse_args()

    label_csv(
        csv_path=args.csv,
        out_csv=args.out_csv,
        out_html=args.out_html,
        out_jsonl=args.out_jsonl,
        cache_dir=args.cache_dir,
        resume=not args.no_resume,
        limit=args.limit,
        workers=max(1, args.workers),
        model=args.model,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        retry_jitter=args.retry_jitter,
        sample_per_accuracy=args.sample_per_accuracy,
        seed=args.seed,
        sample_per_case_accuracy_traces=args.sample_per_case_accuracy_traces,
        case_accuracy_levels=args.case_accuracy_levels,
        regenerate_html_only=args.regenerate_html_only,
    )
