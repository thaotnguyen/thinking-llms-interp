import argparse
import dataclasses
import json
import os
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tempfile
from typing import ClassVar, Literal

import boltons.iterutils
import joblib
import numpy as np
import openai
import pandas as pd
import pydantic
import scipy.optimize
import tensorflow as tf
import tqdm


class StrictModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")


SentenceLabel = Literal["clinical_reasoning", "no_information"]

# fmt: off
class SentenceVerdict(StrictModel): index: int; label: SentenceLabel
class SentenceVerdicts(StrictModel): verdicts: list[SentenceVerdict]
class Label(StrictModel): title: str; description: str
class LabelJudgement(StrictModel): score: float; guidance: str
# GroupFeedback / TaxonomyJudgement belonged to the old per-group LLM-rubric
# taxonomy judge, replaced by the sentence-placement accuracy judge below.
# class GroupFeedback(StrictModel): group_index: int; guidance: str
# class TaxonomyJudgement(StrictModel):
#     score: float; satisfied: bool; group_feedback: list[GroupFeedback]
class TaxonomyLabel(StrictModel): group_index: int; title: str; description: str
class TaxonomyRewrite(StrictModel): groups: list[TaxonomyLabel]
class SentenceCategory(StrictModel): index: int; category_index: int
class SentenceCategories(StrictModel): assignments: list[SentenceCategory]
# fmt: on


def memory_cache(func):
    return joblib.Memory(
        tempfile.mkdtemp(prefix="medinterp-unconfigured-cache."), verbose=0
    ).cache(func)


def parallel_threads(**kwargs):
    return joblib.Parallel(
        n_jobs=64, prefer="threads", require="sharedmem", **kwargs
    )


class LLMQuery:
    def __init__(
        self,
        instructions,
        input_key,
        output_model,
        model=None,
        token_budgets=None,
    ):
        self.instructions = instructions
        self.input_key = input_key
        self.model = model or "gpt-4o-mini"
        self.token_budgets = token_budgets or [220, 420, 720]
        self.output_model = output_model

    def run(self, client, payload):
        for max_tokens in self.token_budgets:
            try:
                response = client.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.instructions},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {self.input_key: payload},
                                default=lambda obj: obj.model_dump(),
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    response_format=self.output_model,
                    max_completion_tokens=max_tokens,
                )
            except openai.LengthFinishReasonError:
                continue
            if parsed := response.choices[0].message.parsed:
                return parsed
        raise RuntimeError(
            f"{self.output_model.__name__} did not return structured output"
        )


NO_INFORMATION_JUDGE = LLMQuery(
    input_key="sentences",
    token_budgets=[4000, 8000, 12000],
    instructions=(
        "Classify each sentence. Use clinical_reasoning for patient facts, "
        "medical terms, findings, diagnoses, evidence, uncertainty, answers, "
        "recall, or reasoning. Use no_information only for pure formatting, "
        "copied instructions, or generic process text with no case, medical, "
        "diagnostic, evidence, uncertainty, or reasoning content. An example "
        "is 'Hmm.'. Do not judge quality or redundancy. When in doubt, choose "
        "clinical_reasoning."
    ),
    output_model=SentenceVerdicts,
)
# LABELLER / JUDGE drove the old per-dimension and per-group draft/critique
# loop (see the commented-out generate_label / dimension_labels below). The new
# taxonomy scheme names all groups jointly from sampled sentences instead.
# LABELLER = LLMQuery(
#     input_key="label",
#     instructions=(
#         "Draft one medical reasoning-step label. Evidence blocks are learned "
#         "signals: examples show behavior and non_examples show contrast. For "
#         "groups, name the recurring behavior across signals. Follow "
#         "revision_task, stay distinct from other_groups, avoid quality or "
#         "outcome titles and incidental details, and return title/description."
#     ),
#     output_model=Label,
# )
# JUDGE = LLMQuery(
#     input_key="label",
#     instructions=(
#         "Judge one medical reasoning-step label as a professional editor. Use "
#         "only this item's evidence. Enforce professional medical language, "
#         "evidence fit, specificity, concise Title Case, and clinician-facing "
#         "contrast. Penalize broad, case-specific, quality/outcome, generic, "
#         "over-modified, or prompt/input-material wording. Guidance should be "
#         "2 or 3 sentences the next draft can follow."
#     ),
#     output_model=LabelJudgement,
# )
# Old LLM-rubric taxonomy judge, replaced by taxonomy_accuracy (sentence
# placement accuracy) below.
# TAXONOMY_JUDGE = LLMQuery(
#     input_key="taxonomy",
#     instructions=(
#         "Judge one medical reasoning taxonomy as a whole. Score whether groups "
#         "name reasoning behaviors, use clear medical taxonomy language, "
#         "preserve meaning/granularity, remain orthogonal, and form a useful "
#         "reasoning progression. Return one 2- or 3-sentence feedback item for "
#         "every group_index."
#     ),
#     output_model=TaxonomyJudgement,
# )
TAXONOMY_PROPOSER = LLMQuery(
    input_key="taxonomy",
    token_budgets=[2000, 4000, 8000],
    instructions=(
        "Propose one general medical reasoning taxonomy. Each group provides "
        "example reasoning sentences drawn from that category. Return every "
        "group_index with a crisp Title Case reasoning-behavior title and a "
        "clinician-facing description naming the recurring behavior across its "
        "examples. Keep groups distinct; avoid quality/outcome or case-specific "
        "wording."
    ),
    output_model=TaxonomyRewrite,
)
TAXONOMY_SYNTHESIZER = LLMQuery(
    input_key="taxonomy",
    token_budgets=[2000, 4000, 8000],
    instructions=(
        "Synthesize one final medical reasoning taxonomy. You are given "
        "group_indices (the exact, complete set of valid group_index values) "
        "and several candidate_taxonomies, each produced independently and "
        "each already covering every group_index as a coherent whole. "
        "Considering all candidate taxonomies together, produce ONE final "
        "taxonomy: for every group_index in group_indices choose or merge into "
        "a single best Title Case reasoning-behavior title and clinician-facing "
        "description. Resolve overlap holistically across the whole taxonomy, "
        "not group by group, so the final groups are mutually distinct and "
        "orthogonal. Return exactly one entry per group_index in group_indices "
        "-- no more, no fewer, and no other values."
    ),
    output_model=TaxonomyRewrite,
)
CATEGORY_CLASSIFIER = LLMQuery(
    input_key="task",
    token_budgets=[2000, 4000, 8000],
    instructions=(
        "Assign each sentence to the single best-fitting category. You are "
        "given a taxonomy (category_index, title, description) and a list of "
        "indexed sentences. Return one category_index for every sentence index."
    ),
    output_model=SentenceCategories,
)
TAXONOMY_REWRITER = LLMQuery(
    input_key="taxonomy",
    model="gpt-4o-mini",
    token_budgets=[2000, 4000, 8000],
    instructions=(
        "Rewrite one finalized general medical reasoning taxonomy. Preserve "
        "each group meaning and granularity. Return every group_index. Use "
        "crisp Title Case reasoning behavior names and clear clinician-facing "
        "descriptions. Do not merge, split, reorder, omit, or add groups."
    ),
    output_model=TaxonomyRewrite,
)


@memory_cache
def load_sentence_judgements(sentences_df):
    judgements_df = pd.DataFrame(
        index=sentences_df.index, data={"is_no_information": pd.NA}
    )
    client = openai.OpenAI(timeout=60.0)
    with tqdm.tqdm(
        total=len(sentences_df), desc="sentence_judgements", unit="sentence"
    ) as bar:
        parallel = parallel_threads(return_as="generator_unordered")
        for judgement_rows in parallel(
            joblib.delayed(judge_payloads)(client, payload)
            for payload in (
                list(payload)
                for payload in boltons.iterutils.chunked_iter(
                    zip(sentences_df.index, sentences_df["text"]), 200
                )
            )
        ):
            for index, is_no_information in judgement_rows:
                judgements_df.at[index, "is_no_information"] = is_no_information
            bar.update(len(judgement_rows))
    if judgements_df["is_no_information"].isna().any():
        raise ValueError("sentence judgement omitted rows")
    return judgements_df.sort_index()


def judge_payloads(client, payload):
    data = NO_INFORMATION_JUDGE.run(
        client,
        [
            {"index": payload_index, "text": text}
            for payload_index, (_row_index, text) in enumerate(payload)
        ],
    )
    verdict_by_index = {
        verdict.index: verdict.label == "no_information"
        for verdict in data.verdicts
        if 0 <= verdict.index < len(payload)
    }
    # Guarantee every sentence in the batch gets a verdict: any index the judge
    # omits defaults to clinical_reasoning (is_no_information=False), matching
    # NO_INFORMATION_JUDGE's own policy ("When in doubt, choose clinical_reasoning").
    return [
        (row_index, verdict_by_index.get(payload_index, False))
        for payload_index, (row_index, _text) in enumerate(payload)
    ]


@dataclasses.dataclass
class LabelPool:
    source_model: str
    signal_id: int
    example_pool: list

    n_pool: ClassVar[int] = 1024

    @classmethod
    def from_scores(cls, sentences_df, model, dimension, pool_ix, score_i):
        # Positives only: the top-n_pool highest-scoring sentences for this
        # dimension. Negatives are no longer used by the taxonomy scheme.
        order_i = np.argsort(score_i)
        return cls(
            model,
            int(dimension),
            sentences_df.loc[
                pool_ix.take(order_i[::-1][: cls.n_pool]),
                "text",
            ].tolist(),
        )

    # def sample(self, sample_count):
    #     return {
    #         "source_model": self.source_model,
    #         "signal_id": self.signal_id,
    #         "examples": np.random.choice(
    #             self.example_pool, sample_count, replace=True
    #         ).tolist(),
    #     }


# def sample_label_evidence(pools, sample_count=20):
#     return [
#         pool.sample(count)
#         for pool, count in zip(
#             pools,
#             np.bincount(
#                 np.random.choice(len(pools), sample_count, replace=True),
#                 minlength=len(pools),
#             ),
#         )
#         if count
#     ]


# generate_label / dimension_labels implemented the old per-dimension and
# per-group draft/critique naming. The taxonomy is now named jointly from
# sampled sentences (see build_taxonomy), so per-dimension labels are dropped.
# def generate_label(client, pools, payload, review_rounds=3):
#     pools = list(pools)
#     attempts = []
#     for review_round in range(review_rounds):
#         label_payload = dict(
#             payload,
#             review_round=review_round,
#             evidence=sample_label_evidence(pools),
#         )
#         label = LABELLER.run(
#             client,
#             (
#                 label_payload
#                 if not attempts
#                 else dict(
#                     label_payload,
#                     previous_label=attempts[-1][0],
#                     revision_task=dict(
#                         label_payload.get("revision_task", {}),
#                         label_feedback=attempts[-1][1],
#                     ),
#                 )
#             ),
#         )
#         feedback = JUDGE.run(
#             client,
#             dict(
#                 payload,
#                 review_round=review_round,
#                 evidence=sample_label_evidence(pools),
#                 proposed_label=label,
#             ),
#         )
#         attempts.append((label, feedback))
#     best_label, _feedback = max(attempts, key=lambda attempt: attempt[1].score)
#     return best_label


# @memory_cache
# def dimension_labels(label_pool_by_dimension):
#     client = openai.OpenAI(timeout=60.0)
#     pools = list(label_pool_by_dimension.values())
#     labels = parallel_threads()(
#         joblib.delayed(generate_label)(
#             client,
#             [pool],
#             {"source_model": pool.source_model, "signal_id": pool.signal_id},
#         )
#         for pool in pools
#     )
#     return pd.DataFrame(
#         dict(
#             model=pool.source_model,
#             dimension=pool.signal_id,
#             **label.model_dump(),
#         )
#         for pool, label in zip(pools, labels)
#     ).set_index(["model", "dimension"])


def group_sentences_by_model(group_df, label_pool_by_dimension):
    # For one taxonomy group, gather positive example sentences per source
    # model by concatenating the example_pools of its member dimensions.
    by_model = {}
    for model, dimension in group_df.index:
        pool = label_pool_by_dimension[(model, int(dimension))]
        by_model.setdefault(model, []).extend(pool.example_pool)
    return by_model


def sample_group(by_model, count, even):
    # even=False: sample `count` sentences from the pooled union of all models.
    # even=True: split `count` as evenly as possible across models (balanced
    # per-model coverage), sampling each model's share separately.
    models = sorted(by_model)
    if not even:
        pooled = [text for model in models for text in by_model[model]]
        return np.random.choice(pooled, count, replace=True).tolist()
    base, remainder = divmod(count, len(models))
    texts = []
    for model_i, model in enumerate(models):
        share = base + (1 if model_i < remainder else 0)
        if share:
            texts.extend(
                np.random.choice(by_model[model], share, replace=True).tolist()
            )
    return texts


def taxonomy_accuracy(client, taxonomy_labels, by_model_by_group, per_group):
    # Score a taxonomy by how well an LLM can place held-out sentences into the
    # right category. Sample per_group sentences per category evenly across
    # models, then classify them (batched) and return placement accuracy.
    items = [
        (label.group_index, text)
        for label in taxonomy_labels
        for text in sample_group(
            by_model_by_group[label.group_index], per_group, even=True
        )
    ]
    np.random.shuffle(items)
    taxonomy_payload = [
        {
            "category_index": label.group_index,
            "title": label.title,
            "description": label.description,
        }
        for label in taxonomy_labels
    ]
    correct = 0
    parallel = parallel_threads(return_as="generator_unordered")
    for batch_correct in parallel(
        joblib.delayed(classify_batch)(client, taxonomy_payload, batch)
        for batch in boltons.iterutils.chunked_iter(list(enumerate(items)), 64)
    ):
        correct += batch_correct
    return correct / len(items) if items else 0.0


def classify_batch(client, taxonomy_payload, batch):
    # batch: list of (global_index, (true_group_index, text)).
    data = CATEGORY_CLASSIFIER.run(
        client,
        {
            "taxonomy": taxonomy_payload,
            "sentences": [
                {"index": local_index, "text": text}
                for local_index, (_gi, (_true, text)) in enumerate(batch)
            ],
        },
    )
    predicted = {
        assignment.index: assignment.category_index
        for assignment in data.assignments
        if 0 <= assignment.index < len(batch)
    }
    # An omitted sentence defaults to -1 and simply counts as incorrect.
    return sum(
        predicted.get(local_index, -1) == true
        for local_index, (_gi, (true, _text)) in enumerate(batch)
    )


@memory_cache
def build_taxonomy(
    label_pool_by_dimension,
    dimensions_df,
    taxonomy_rounds=3,
    proposal_rounds=8,
    sentences_per_group=64,
):
    client = openai.OpenAI(timeout=60.0)
    group_rows = list(dimensions_df.groupby("group", sort=True))
    groups = [int(group) for group, _group_df in group_rows]
    by_model_by_group = {
        int(group): group_sentences_by_model(group_df, label_pool_by_dimension)
        for group, group_df in group_rows
    }
    best_score = -1.0
    best_taxonomy_labels = None
    for _taxonomy_round in range(taxonomy_rounds):
        # Propose a full taxonomy proposal_rounds times, each from a fresh
        # 64-sentence-per-group sample fed jointly into one prompt.
        proposals = parallel_threads()(
            joblib.delayed(TAXONOMY_PROPOSER.run)(
                client,
                {
                    "groups": [
                        {
                            "group_index": group,
                            "examples": sample_group(
                                by_model_by_group[group],
                                sentences_per_group,
                                even=False,
                            ),
                        }
                        for group in groups
                    ]
                },
            )
            for _ in range(proposal_rounds)
        )
        # Keep each proposal round as a coherent whole (all 6 group titles chosen
        # together) rather than re-bucketing candidates per group, which would
        # lose the cross-group distinctness each round already reasoned about.
        synthesized = TAXONOMY_SYNTHESIZER.run(
            client,
            {
                "group_indices": groups,
                "candidate_taxonomies": [
                    {
                        "groups": [
                            {
                                "group_index": label.group_index,
                                "title": label.title,
                                "description": label.description,
                            }
                            for label in proposal.groups
                        ]
                    }
                    for proposal in proposals
                ]
            },
        ).groups
        # Guard against a hallucinated/omitted group_index: fall back to that
        # group's first proposal-round candidate if the synthesizer drops it.
        synthesized_by_group = {
            label.group_index: label
            for label in synthesized
            if label.group_index in by_model_by_group
        }
        taxonomy_labels = [
            synthesized_by_group.get(
                group,
                next(
                    label
                    for proposal in proposals
                    for label in proposal.groups
                    if label.group_index == group
                ),
            )
            for group in groups
        ]
        score = taxonomy_accuracy(
            client, taxonomy_labels, by_model_by_group, sentences_per_group
        )
        print("taxonomy_round accuracy", score, flush=True)
        if score > best_score:
            best_score = score
            best_taxonomy_labels = taxonomy_labels
    rewrite = TAXONOMY_REWRITER.run(client, {"groups": best_taxonomy_labels})
    return {label.group_index: label for label in rewrite.groups}


def parallel_model_stage(sentences_df, dimensions_df, devices, stage, *args):
    return pd.concat(
        joblib.Parallel(
            n_jobs=len(devices), prefer="threads", require="sharedmem"
        )(
            joblib.delayed(stage)(
                model_df,
                dimensions_df.loc[model],
                devices[model_i % len(devices)],
                *args,
            )
            for model_i, (model, model_df) in enumerate(
                sentences_df.groupby(level="model", sort=True)
            )
        )
    )


def load_inputs(sentences_path):
    sentences_df = pd.read_csv(sentences_path).set_index(
        ["pmcid", "model", "sentence_index"], drop=False
    )
    if sentences_df.index.has_duplicates:
        raise ValueError("duplicate sentence index")
    case_model_counts = sentences_df.groupby(level="pmcid")["model"].nunique()
    cases = case_model_counts[
        case_model_counts == sentences_df["model"].nunique()
    ].index
    sentences_df = sentences_df[sentences_df["pmcid"].isin(cases)]
    sentences_df = sentences_df.sort_index()
    sentences_df["dimensions"] = sentences_df["dimensions"].apply(
        lambda value: tf.constant(
            json.loads(value) if isinstance(value, str) else value,
            dtype=tf.float32,
        )
    )
    return sentences_df


def build_dimensions(sentences_df):
    rows = []
    label_pool_by_dimension = {}
    for model, model_df in sentences_df.groupby(level="model", sort=True):
        semantic_df = model_df.loc[
            ~model_df["is_no_information"].to_numpy(dtype=bool)
        ]
        if semantic_df.empty:
            raise ValueError(f"model has no semantic sentences model={model}")
        x_di = np.stack(
            [value.numpy() for value in semantic_df["dimensions"]], axis=1
        )
        mean_d = x_di.mean(axis=1)
        std_d = x_di.std(axis=1)
        if not np.all(std_d > 1e-12):
            raise ValueError(f"model has zero-variance dimensions {model}")
        z_di = (x_di - mean_d[:, None]) / std_d[:, None]
        trace_offsets_i = semantic_df.groupby(level="pmcid", sort=True).size()
        trace_pmcid_list = trace_offsets_i.index.tolist()
        split_i = np.cumsum(trace_offsets_i.to_numpy())[:-1]
        for dimension in range(int(z_di.shape[0])):
            z_t = dict(
                zip(trace_pmcid_list, np.split(z_di[dimension], split_i))
            )
            rows.append({"model": model, "dimension": dimension, "z_t": z_t})
            label_pool_by_dimension[(model, dimension)] = LabelPool.from_scores(
                sentences_df,
                model,
                dimension,
                semantic_df.index,
                z_di[dimension],
            )
    dimensions_df = pd.DataFrame(rows).set_index(["model", "dimension"])
    dimensions_df["global_d"] = np.arange(len(dimensions_df))
    return dimensions_df, label_pool_by_dimension


def trace_progress_i(row_count, progress_bins):
    position_i = tf.range(row_count, dtype=tf.float32) + 0.5
    return tf.cast(tf.floor(position_i * progress_bins / row_count), tf.int32)


def trace_z_id(dimensions_df, pmcid):
    # A (model, case) trace can be entirely no_information; its pmcid is then
    # absent from z_t (which only spans semantic sentences). Fall back to an
    # empty score vector so the trace is handled purely via its no_information
    # one-hot states downstream.
    empty = np.zeros((0,), dtype=np.float32)
    return tf.stack(
        [row["z_t"].get(pmcid, empty) for _dimension, row in dimensions_df.iterrows()],
        axis=1,
    )


def trace_state_is(score_id, no_information_t):
    no_information_t = tf.convert_to_tensor(no_information_t)
    no_information_i = tf.where(no_information_t)
    state_count = score_id.shape[1] + 1
    semantic_is = tf.concat(
        [
            tf.nn.softmax(score_id, axis=1),
            tf.zeros((len(score_id), 1), dtype=tf.float32),
        ],
        axis=1,
    )
    state_is = tf.tensor_scatter_nd_update(
        tf.zeros((len(no_information_t), state_count), dtype=tf.float32),
        tf.where(~no_information_t),
        semantic_is,
    )
    return tf.tensor_scatter_nd_update(
        state_is,
        no_information_i,
        tf.one_hot(
            tf.fill((len(no_information_i),), state_count - 1),
            state_count,
        ),
    )


def model_phase(
    model_df,
    dimensions_df,
    device,
    transition_bandwidth,
    progress_bins,
):
    with tf.device(device):
        trace_data_list = []
        dimension_list = []
        dimension_d = tf.convert_to_tensor(
            dimensions_df.index.to_numpy(), dtype=tf.int32
        )
        for pmcid, trace_df in model_df.groupby(level="pmcid", sort=True):
            no_information_t = trace_df["is_no_information"].to_numpy(
                dtype=bool
            )
            z_id = trace_z_id(dimensions_df, pmcid)
            semantic_i = tf.where(tf.convert_to_tensor(~no_information_t))[:, 0]
            dimension_i = tf.tensor_scatter_nd_update(
                tf.fill((len(trace_df),), -1),
                semantic_i[:, None],
                tf.gather(dimension_d, tf.argmax(z_id, axis=1)),
            )
            dimension_list.append(
                pd.Series(
                    dimension_i.numpy(), index=trace_df.index, name="dimension"
                )
            )
            trace_data_list.append(
                (
                    trace_state_is(z_id, no_information_t),
                    trace_progress_i(len(trace_df), progress_bins),
                )
            )
        phase_i = tf.concat(
            fit_phases(
                trace_data_list,
                transition_bandwidth,
                progress_bins,
            ),
            axis=0,
        )
    return pd.DataFrame(
        {
            "dimension": pd.concat(dimension_list),
            "phase": pd.Series(list(tf.unstack(phase_i)), index=model_df.index),
        }
    )


def fit_phases(
    trace_data_list,
    transition_bandwidth,
    progress_bins,
):
    length_t = tf.convert_to_tensor(
        [len(state_is) for state_is, _progress_i in trace_data_list],
        dtype=tf.int32,
    )
    state_its = tf.ragged.stack(
        [state_is for state_is, _progress_i in trace_data_list]
    ).to_tensor()
    progress_it = tf.ragged.stack(
        [progress_i for _state_is, progress_i in trace_data_list]
    ).to_tensor()
    max_length = int(tf.shape(state_its)[1].numpy())
    state_prior_s = tf.maximum(tf.reduce_sum(state_its, axis=[0, 1]), 1e-12)
    state_prior_s = state_prior_s / tf.reduce_sum(state_prior_s)
    progress_itb = tf.one_hot(progress_it, progress_bins)
    emission_bs = tf.tile(0.1 * state_prior_s[None, :], (progress_bins, 1))
    emission_bs = emission_bs + tf.einsum(
        "itb,its->bs", progress_itb, state_its
    )
    emission_bs = tf.linalg.normalize(emission_bs, ord=1, axis=1)[0]
    counts_bbss = tf.tile(
        0.1 * emission_bs[None, :, None, :],
        (progress_bins, 1, tf.shape(state_its)[2], 1),
    )
    counts_bbss = counts_bbss + tf.einsum(
        "ita,itb,itr,its->abrs",
        progress_itb[:, :-1],
        progress_itb[:, 1:],
        state_its[:, :-1],
        state_its[:, 1:],
    )
    transition_bbss = tf.linalg.normalize(counts_bbss, ord=1, axis=3)[0]
    bin_b = tf.range(progress_bins, dtype=tf.float32)
    expected_step_t = tf.cast(progress_bins - 1, tf.float32) / tf.cast(
        length_t - 1, tf.float32
    )
    center_tb = tf.minimum(
        tf.constant(progress_bins - 1, dtype=tf.float32),
        bin_b[None, :] + expected_step_t[:, None],
    )
    transition_tbb = tf.exp(
        -tf.abs(bin_b[None, None, :] - center_tb[:, :, None])
        / transition_bandwidth
    )
    transition_tbb = tf.linalg.normalize(transition_tbb, ord=1, axis=2)[0]
    likelihood_its = state_its / state_prior_s
    joint_ts = emission_bs[0][None, :] * likelihood_its[:, 0]
    joint_tbs = tf.pad(
        joint_ts[:, None, :],
        [[0, 0], [0, progress_bins - 1], [0, 0]],
    )
    joint_tbs = tf.linalg.normalize(joint_tbs, ord=1, axis=[1, 2])[0]
    phase_rows = [tf.reduce_sum(joint_tbs, axis=2)]
    valid_it = tf.sequence_mask(length_t, max_length)
    for i in range(1, max_length):
        update_tbs = tf.einsum(
            "tar,tab,abrs->tbs", joint_tbs, transition_tbb, transition_bbss
        )
        update_tbs = update_tbs * likelihood_its[:, i, None, :]
        update_tbs = update_tbs / tf.maximum(
            tf.reduce_sum(update_tbs, axis=[1, 2], keepdims=True), 1e-12
        )
        joint_tbs = tf.where(valid_it[:, i, None, None], update_tbs, joint_tbs)
        phase_rows.append(tf.reduce_sum(joint_tbs, axis=2))
    phase_itb = tf.stack(phase_rows, axis=1)
    return [
        phase_itb[trace_i, :length]
        for trace_i, length in enumerate(length_t.numpy())
    ]


def universal_group_phase(
    sentences_df,
    dimensions_df,
    devices,
    transition_bandwidth,
    progress_bins,
):
    trace_data_list = []
    for trace_i, (trace_key, trace_df) in enumerate(
        sentences_df.groupby(level=["pmcid", "model"], sort=True)
    ):
        with tf.device(devices[trace_i % len(devices)]):
            pmcid, model = trace_key
            no_information_t = trace_df["is_no_information"].to_numpy(
                dtype=bool
            )
            z_ig = tf.stack(
                [
                    tf.reduce_max(trace_z_id(group_df, pmcid), axis=1)
                    for _group, group_df in dimensions_df.loc[model].groupby(
                        "group", sort=True
                    )
                ],
                axis=1,
            )
            trace_data_list.append(
                (
                    trace_state_is(z_ig, no_information_t),
                    trace_progress_i(len(trace_df), progress_bins),
                )
            )
    with tf.device(devices[0]):
        phase_i = tf.concat(
            fit_phases(
                trace_data_list,
                transition_bandwidth,
                progress_bins,
            ),
            axis=0,
        )
    return pd.Series(
        list(tf.unstack(phase_i)),
        index=sentences_df.index,
        name="unified_phase",
    )


def project_groups(assignment_dg, dimensions_df, group_g):
    assignment_dg = assignment_dg.numpy()
    groups_d = group_g[np.argmax(assignment_dg, axis=1)]
    for _model, model_dims_df in dimensions_df.groupby(
        level="model", sort=True
    ):
        global_d = model_dims_df["global_d"].to_numpy()
        dimension_i, group_i = scipy.optimize.linear_sum_assignment(
            -assignment_dg[global_d]
        )
        groups_d[global_d[dimension_i]] = group_g[group_i]
    return pd.Series(groups_d, dimensions_df.index, name="group")


def relaxed_progress_groups(
    sentences_df,
    dimensions_df,
    initial_groups_t,
    devices,
    transition_bandwidth,
    progress_bins,
):
    steps = 40
    trace_batch_size = 32
    max_beta = 4.0
    coverage_weight = 100.0
    group_g = np.asarray(sorted(pd.unique(initial_groups_t)))
    with tf.device(devices[0]):
        initial_position_i = tf.convert_to_tensor(
            np.searchsorted(group_g, initial_groups_t),
            dtype=tf.int32,
        )
        logits = tf.Variable(
            -1.0 + 2.0 * tf.one_hot(initial_position_i, len(group_g))
        )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.08)
    with tf.device(devices[0]):
        model_mask_md = tf.constant(
            pd.get_dummies(
                dimensions_df.index.get_level_values("model")
            ).T.to_numpy(),
            dtype=tf.float32,
        )
    trace_rows = []
    for trace_i, ((model, pmcid), trace_df) in enumerate(
        sentences_df.groupby(level=["model", "pmcid"], sort=True)
    ):
        with tf.device(devices[trace_i % len(devices)]):
            model_dims_df = dimensions_df.loc[model]
            no_information_t = trace_df["is_no_information"].to_numpy(
                dtype=bool
            )
            trace_rows.append(
                {
                    "global_d": tf.convert_to_tensor(
                        model_dims_df["global_d"].to_numpy(), dtype=tf.int32
                    ),
                    "no_information_t": tf.convert_to_tensor(
                        no_information_t, dtype=tf.bool
                    ),
                    "progress_i": trace_progress_i(
                        len(trace_df), progress_bins
                    ),
                    "target_ib": tf.stack(trace_df["phase"].tolist()),
                    "z_id": trace_z_id(model_dims_df, pmcid),
                }
            )
    bin_b = tf.range(progress_bins, dtype=tf.float32)
    kernel_bb = tf.exp(
        -tf.abs(bin_b[:, None] - bin_b[None, :]) / transition_bandwidth
    )
    for step in range(steps + 1):
        temperature = 1.0 + max(0.0, 1.0 - step / steps)
        if trace_batch_size < len(trace_rows):
            batch_i = tf.random.shuffle(tf.range(len(trace_rows)))[
                :trace_batch_size
            ].numpy()
            batch = [trace_rows[i] for i in batch_i]
        else:
            batch = trace_rows
        with tf.GradientTape() as tape:
            assignment_dg = tf.nn.softmax(logits / temperature, axis=1)
            state_list = []
            for i, row in enumerate(batch):
                with tf.device(devices[i % len(devices)]):
                    log_assignment_dg = tf.math.log(
                        tf.gather(assignment_dg, row["global_d"]) + 1e-12
                    )
                    group_z_ig = tf.reduce_logsumexp(
                        max_beta * row["z_id"][:, :, None]
                        + log_assignment_dg[None, :, :],
                        axis=1,
                    )
                    group_z_ig = group_z_ig / max_beta
                    state_list.append(
                        trace_state_is(group_z_ig, row["no_information_t"])
                    )
            phase_rows = fit_phases(
                list(zip(state_list, [row["progress_i"] for row in batch])),
                transition_bandwidth,
                progress_bins,
            )
            progress_loss = tf.add_n(
                [
                    tf.einsum(
                        "ib,bc,ic->",
                        phase_ib - row["target_ib"],
                        kernel_bb,
                        phase_ib - row["target_ib"],
                    )
                    for row, phase_ib in zip(batch, phase_rows)
                ]
            ) / sum(len(phase_ib) for phase_ib in phase_rows)
            coverage_loss = tf.reduce_mean(
                tf.square(
                    tf.nn.relu(0.75 - tf.matmul(model_mask_md, assignment_dg))
                )
            )
            loss = progress_loss + coverage_weight * coverage_loss
        if step < steps:
            gradients = tape.gradient(loss, [logits])
            optimizer.apply_gradients(zip(gradients, [logits]))
    return project_groups(assignment_dg, dimensions_df, group_g)


def progress_json(values):
    return [float(f"{float(value):.5g}") for value in values.numpy()]


def output_data(output_dir, sentences_df, dimensions_df, labels_by_group):
    trace_dir = os.path.join(output_dir, "trace_cases")
    if os.path.exists(trace_dir):
        shutil.rmtree(trace_dir)
    os.makedirs(trace_dir)

    trace_rows = []
    models = sorted(pd.unique(sentences_df["model"]))
    semantic_df = sentences_df[
        ~sentences_df["is_no_information"].to_numpy(dtype=bool)
    ]
    examples_by_group_model = {
        (int(group), model): [
            {"model": model, "case": pmcid, "text": row["text"]}
            for (pmcid, _model, _sentence_index), row in group_df.iterrows()
        ]
        for (group, model), group_df in semantic_df.groupby(
            [semantic_df["group_index"], semantic_df["model"]], sort=True
        )
    }
    for pmcid, trace_df in sentences_df.groupby(level="pmcid", sort=True):
        model_rows_by_model = {}
        for model, model_df in trace_df.groupby(level="model", sort=True):
            model_rows_by_model[model] = [
                {
                    "section": int(row["sentence_index"]),
                    "text": row["text"],
                    "dimension": int(row["dimension"]),
                    "group": int(row["group_index"]),
                    "is_no_information": bool(row["is_no_information"]),
                    "model_progress": progress_json(row["phase"]),
                    "unified_progress": progress_json(row["unified_phase"]),
                }
                for _row_index, row in model_df.iterrows()
            ]
        trace_file = f"trace_cases/{pmcid}.json"
        with open(
            os.path.join(output_dir, trace_file), "w", encoding="utf-8"
        ) as trace_output:
            json.dump(
                {"models": model_rows_by_model},
                trace_output,
                separators=(",", ":"),
            )
        trace_rows.append(
            {
                "pmcid": pmcid,
                "file": trace_file,
                "counts": {
                    model: len(rows)
                    for model, rows in model_rows_by_model.items()
                },
            }
        )

    taxonomy_rows = []
    for group, group_df in dimensions_df.groupby("group", sort=True):
        group = int(group)
        label = labels_by_group[group]
        taxonomy_rows.append(
            dict(
                group_index=group,
                # Per-dimension labels removed; only the member (model, dimension)
                # pairs are recorded now (titles/descriptions no longer computed).
                dimension_members=[
                    {"model": model, "dimension": int(dimension)}
                    for (model, dimension), row in group_df.iterrows()
                ],
                title=label.title,
                description=label.description,
                count=sum(
                    len(examples_by_group_model.get((group, model), []))
                    for model in models
                ),
                examples={
                    model: {"count": len(rows), "sample": rows[:8]}
                    for model in models
                    if (rows := examples_by_group_model.get((group, model)))
                },
            )
        )
    return {
        "models": models,
        "traces": trace_rows,
        "summary": {
            "groups": dimensions_df["group"].nunique(),
            "annotations": len(sentences_df),
            "traces": len(trace_rows),
        },
        "taxonomy": {"groups": taxonomy_rows},
    }


def run(
    sentences_path,
    output_dir,
    regenerate_cache=None,
    transition_bandwidth=1.0,
    progress_bins=12,
    group_count=6,
):
    os.makedirs(output_dir, exist_ok=True)
    stages = {
        "sentence_judgements": load_sentence_judgements,
        # "dimension_labels": dimension_labels,  # per-dimension labelling removed
        "taxonomy": build_taxonomy,
    }
    regenerate_cache = set(regenerate_cache or [])
    if "all" in regenerate_cache:
        regenerate_cache = set(stages)
    memory = joblib.Memory(output_dir, verbose=0)
    for stage in stages.values():
        stage.store_backend = memory.store_backend
    for name in regenerate_cache:
        stages[name].clear(warn=False)
    tf.random.set_seed(42)
    np.random.seed(42)
    devices = [
        device.name for device in tf.config.list_logical_devices("GPU")
    ] or ["/CPU:0"]
    print("tensorflow_devices", ", ".join(devices), flush=True)
    sentences_df = load_inputs(sentences_path)
    judgements_path = os.path.join(output_dir, "sentence_judgements.csv")
    if "sentence_judgements" not in regenerate_cache and os.path.exists(
        judgements_path
    ):
        judgements_df = pd.read_csv(
            judgements_path,
            dtype={"pmcid": str, "model": str, "sentence_index": int},
        )
    else:
        judgements_df = load_sentence_judgements(sentences_df).reset_index()
        judgements_df.to_csv(judgements_path, index=False)
    sentences_df["is_no_information"] = judgements_df.set_index(
        ["pmcid", "model", "sentence_index"]
    ).loc[sentences_df.index, "is_no_information"]
    dimensions_df, label_pool_by_dimension = build_dimensions(sentences_df)
    sentences_df[["dimension", "phase"]] = parallel_model_stage(
        sentences_df,
        dimensions_df,
        devices,
        model_phase,
        transition_bandwidth,
        progress_bins,
    )
    # Per-dimension labelling removed; the taxonomy is named jointly from
    # sampled sentences in build_taxonomy.
    # dimensions_df[["title", "description"]] = dimension_labels(
    #     label_pool_by_dimension
    # ).loc[dimensions_df.index, ["title", "description"]]
    min_dimension_count = dimensions_df.groupby(level="model").size().min()
    group_count = min(int(group_count), int(min_dimension_count))
    initial_groups_t = (
        dimensions_df.groupby(level="model").cumcount() % group_count
    ).rename("group")
    dimensions_df["group"] = relaxed_progress_groups(
        sentences_df,
        dimensions_df,
        initial_groups_t,
        devices,
        transition_bandwidth,
        progress_bins,
    )
    sentences_df["group_index"] = [
        (
            -1
            if dimension < 0
            else dimensions_df.at[(model, int(dimension)), "group"]
        )
        for (_pmcid, model, _sentence_index), dimension in sentences_df[
            "dimension"
        ].items()
    ]
    sentences_df["unified_phase"] = universal_group_phase(
        sentences_df,
        dimensions_df,
        devices,
        transition_bandwidth,
        progress_bins,
    )
    dimensions_df = dimensions_df.drop(columns=["z_t"])
    labels_by_group = build_taxonomy(label_pool_by_dimension, dimensions_df)
    data = output_data(output_dir, sentences_df, dimensions_df, labels_by_group)
    data_path = os.path.join(output_dir, "data.json")
    with open(data_path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, separators=(",", ":"))
    print("data=" + data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sentences_path")
    parser.add_argument("output_dir")
    parser.add_argument("--regenerate-cache", action="append", default=[])
    parser.add_argument("--group-count", type=int, default=6)
    run(**vars(parser.parse_args()))
