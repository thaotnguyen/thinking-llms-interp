import dataclasses
import json
import os
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from typing import ClassVar, Literal

import joblib
import numpy as np
import openai
import pandas as pd
import pydantic
import scipy.optimize
import tensorflow as tf

import pypelite


class StrictModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")


SentenceLabel = Literal["clinical_reasoning", "no_information"]

# fmt: off
class SentenceVerdict(StrictModel): index: int; label: SentenceLabel
class SentenceVerdicts(StrictModel): verdicts: list[SentenceVerdict]
class Label(StrictModel): title: str; description: str
class LabelJudgement(StrictModel): score: float; guidance: str
class GroupFeedback(StrictModel): group_index: int; guidance: str
class TaxonomyJudgement(StrictModel):
    score: float; satisfied: bool; group_feedback: list[GroupFeedback]
class TaxonomyLabel(StrictModel): group_index: int; title: str; description: str
class TaxonomyRewrite(StrictModel): groups: list[TaxonomyLabel]
# fmt: on


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
LABELLER = LLMQuery(
    input_key="label",
    instructions=(
        "Draft one medical reasoning-step label. Evidence blocks are learned "
        "signals: examples show behavior and non_examples show contrast. For "
        "groups, name the recurring behavior across signals. Follow "
        "revision_task, stay distinct from other_groups, avoid quality or "
        "outcome titles and incidental details, and return title/description."
    ),
    output_model=Label,
)
JUDGE = LLMQuery(
    input_key="label",
    instructions=(
        "Judge one medical reasoning-step label as a professional editor. Use "
        "only this item's evidence. Enforce professional medical language, "
        "evidence fit, specificity, concise Title Case, and clinician-facing "
        "contrast. Penalize broad, case-specific, quality/outcome, generic, "
        "over-modified, or prompt/input-material wording. Guidance should be "
        "2 or 3 sentences the next draft can follow."
    ),
    output_model=LabelJudgement,
)
TAXONOMY_JUDGE = LLMQuery(
    input_key="taxonomy",
    instructions=(
        "Judge one medical reasoning taxonomy as a whole. Score whether groups "
        "name reasoning behaviors, use clear medical taxonomy language, "
        "preserve meaning/granularity, remain orthogonal, and form a useful "
        "reasoning progression. Return one 2- or 3-sentence feedback item for "
        "every group_index."
    ),
    output_model=TaxonomyJudgement,
)
TAXONOMY_REWRITER = LLMQuery(
    input_key="taxonomy",
    model="gpt-5.5",
    token_budgets=[2000, 4000, 8000],
    instructions=(
        "Rewrite one finalized general medical reasoning taxonomy. Preserve "
        "each group meaning and granularity. Return every group_index. Use "
        "crisp Title Case reasoning behavior names and clear clinician-facing "
        "descriptions. Do not merge, split, reorder, omit, or add groups."
    ),
    output_model=TaxonomyRewrite,
)


@pypelite.stage(
    name="sentence_judgements",
    key=("pmcid", "model", "sentence_index"),
    batch="records",
    batch_size=200,
    workers=64,
)
def load_sentence_judgements(records):
    client = openai.OpenAI(timeout=60.0)
    payload_rows = [
        (
            (
                record["pmcid"],
                record["model"],
                record["sentence_index"],
            ),
            record["text"],
        )
        for record in records
    ]
    judgement = NO_INFORMATION_JUDGE.run(
        client,
        [
            {"index": payload_index, "text": text}
            for payload_index, (_row_index, text) in enumerate(payload_rows)
        ],
    )
    rows = []
    for verdict in judgement.verdicts:
        pmcid, model, sentence_index = payload_rows[verdict.index][0]
        rows.append(
            {
                "pmcid": pmcid,
                "model": model,
                "sentence_index": sentence_index,
                "is_no_information": verdict.label == "no_information",
            }
        )
    return rows


@dataclasses.dataclass
class LabelPool:
    example_pool: list
    non_example_pool: list

    n_pool: ClassVar[int] = 128

    @classmethod
    def from_scores(cls, sentences_df, pool_ix, score_i):
        order_i = np.argsort(score_i)
        return cls(
            sentences_df.loc[
                pool_ix.take(order_i[::-1][: cls.n_pool]),
                "text",
            ].tolist(),
            sentences_df.loc[
                pool_ix.take(order_i[: cls.n_pool]),
                "text",
            ].tolist(),
        )

    @classmethod
    def merge_pools(cls, pools):
        pools = list(pools)
        if len(pools) == 1:
            return pools[0]
        return cls(
            [example for pool in pools for example in pool.example_pool],
            [
                non_example
                for pool in pools
                for non_example in pool.non_example_pool
            ],
        )

    def sample(self, sample_count):
        return {
            "examples": np.random.choice(
                self.example_pool, sample_count, replace=True
            ).tolist(),
            "non_examples": np.random.choice(
                self.non_example_pool, sample_count, replace=True
            ).tolist(),
        }


def generate_label(client, pools, label_context, review_rounds=3):
    label_pool = LabelPool.merge_pools(pools)
    attempts = []
    for review_round in range(review_rounds):
        label_evidence = label_pool.sample(20)
        judge_evidence = label_pool.sample(20)
        label_payload = dict(
            label_context,
            review_round=review_round,
            evidence=label_evidence,
        )
        label = LABELLER.run(
            client,
            (
                label_payload
                if not attempts
                else dict(
                    label_payload,
                    previous_label=attempts[-1][0],
                    revision_task=dict(
                        label_payload.get("revision_task", {}),
                        label_feedback=attempts[-1][1],
                    ),
                )
            ),
        )
        feedback = JUDGE.run(
            client,
            dict(
                label_context,
                review_round=review_round,
                evidence=judge_evidence,
                proposed_label=label,
            ),
        )
        attempts.append((label, feedback))
    best_label, _feedback = max(attempts, key=lambda attempt: attempt[1].score)
    return best_label


@pypelite.stage(
    name="dimension_labels",
    key=("model", "dimension"),
    batch="dimensions_df",
    workers=64,
)
def dimension_labels(dimensions_df):
    client = openai.OpenAI(timeout=60.0)
    return [
        pd.Series(
            generate_label(
                client,
                [row["pool"]],
                {
                    "source_model": row["model"],
                    "signal_id": row["dimension"],
                },
            ).model_dump(),
            name=(row["model"], int(row["dimension"])),
        )
        for row in dimensions_df
    ]


def taxonomy_key(dimensions_df, **_kwargs):
    return tuple(
        (
            int(group),
            tuple(
                (model, int(dimension)) for model, dimension in group_df.index
            ),
        )
        for group, group_df in dimensions_df.groupby("group", sort=True)
    )


def dimension_embedding_key(dimensions_df):
    return tuple(
        (model, int(dimension), row["title"], row["description"])
        for (model, dimension), row in dimensions_df.iterrows()
    )


@pypelite.stage(name="dimension_embeddings", key=dimension_embedding_key)
def dimension_embeddings(dimensions_df):
    response = openai.OpenAI(timeout=120.0).embeddings.create(
        model="text-embedding-3-small",
        input=[
            f"Title: {row['title']}\nDescription: {row['description']}"
            for _dimension, row in dimensions_df.iterrows()
        ],
    )
    embedding_de = np.asarray(
        [item.embedding for item in response.data], dtype=np.float32
    )
    return embedding_de / np.linalg.norm(embedding_de, axis=1, keepdims=True)


def sentence_embedding_key(sentences_df):
    return tuple(
        (pmcid, model, int(sentence_index), row["text"])
        for (pmcid, model, sentence_index), row in sentences_df.iterrows()
    )


@pypelite.stage(name="sentence_embeddings", key=sentence_embedding_key)
def sentence_embeddings(sentences_df):
    client = openai.OpenAI(timeout=120.0)
    embedding_rows = []
    text_list = sentences_df["text"].tolist()
    for start in range(0, len(text_list), 512):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text_list[start : start + 512],
        )
        embedding_rows.extend(item.embedding for item in response.data)
    return np.asarray(embedding_rows, dtype=np.float32)


def sentence_steering_vectors(sentences_df, dimensions_df, embedding_se):
    direction_list = []
    for model, model_df in sentences_df.groupby(level="model", sort=True):
        model_dimensions_df = dimensions_df.loc[model]
        model_embedding_se = embedding_se[
            sentences_df.index.get_level_values("model") == model
        ]
        case_direction_list = []
        offset = 0
        for pmcid, case_df in model_df.groupby(level="pmcid", sort=True):
            row_count = len(case_df)
            case_embedding_te = model_embedding_se[offset : offset + row_count]
            offset += row_count
            case_embedding_te = case_embedding_te - case_embedding_te.mean(
                axis=0, keepdims=True
            )
            activation_td = np.stack(
                [
                    model_dimensions_df.at[dimension, "z_t"][pmcid]
                    for dimension in model_dimensions_df.index
                ],
                axis=1,
            )
            activation_td = activation_td - activation_td.mean(
                axis=0, keepdims=True
            )
            case_direction_list.append(
                np.einsum("td,te->de", activation_td, case_embedding_te)
                / (np.abs(activation_td).sum(axis=0)[:, None] + 1e-12)
            )
        direction_list.append(np.mean(case_direction_list, axis=0))
    direction_de = np.concatenate(direction_list)
    return direction_de / np.linalg.norm(direction_de, axis=1, keepdims=True)


@pypelite.stage(name="taxonomy", key=taxonomy_key)
def build_taxonomy(
    label_pool_by_dimension,
    dimensions_df,
    taxonomy_rounds=4,
    review_rounds=4,
):
    client = openai.OpenAI(timeout=60.0)
    taxonomy_judgement = None
    taxonomy_label_list = None
    best_score = -1.0
    best_taxonomy_labels = None
    for taxonomy_round in range(taxonomy_rounds):
        group_rows = list(dimensions_df.groupby("group", sort=True))
        guidance_by_group = previous_by_group = {}
        if taxonomy_judgement is not None:
            guidance_by_group = {
                item.group_index: item
                for item in taxonomy_judgement.group_feedback
            }
            previous_by_group = {
                label.group_index: label for label in taxonomy_label_list
            }
        labels = [
            generate_label(
                client,
                [
                    label_pool_by_dimension[(model, int(dimension))]
                    for model, dimension in group_df.index
                ],
                dict(
                    group_index=int(group),
                    prior_summaries=[
                        {
                            "source_model": model,
                            "signal_id": int(dimension),
                            "title": row["title"],
                            "description": row["description"],
                        }
                        for (model, dimension), row in group_df.iterrows()
                    ],
                    **(
                        {
                            "previous_label": previous_by_group[int(group)],
                            "revision_task": {
                                "group_feedback": guidance_by_group[int(group)],
                                "other_groups": [
                                    label
                                    for label in taxonomy_label_list
                                    if label.group_index != int(group)
                                ],
                            },
                        }
                        if taxonomy_label_list is not None
                        else {}
                    ),
                ),
                review_rounds,
            )
            for group, group_df in group_rows
        ]
        taxonomy_label_list = [
            TaxonomyLabel(
                group_index=int(group),
                **label.model_dump(),
            )
            for (group, _group_df), label in zip(group_rows, labels)
        ]
        taxonomy_judgement = TAXONOMY_JUDGE.run(
            client,
            {"groups": taxonomy_label_list},
        )
        if taxonomy_judgement.score > best_score:
            best_score = taxonomy_judgement.score
            best_taxonomy_labels = taxonomy_label_list
        if taxonomy_judgement.satisfied:
            break
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
    if pmcid not in next(iter(dimensions_df["z_t"].values)):
        return tf.zeros((0, len(dimensions_df)), dtype=tf.float32)
    return tf.stack(
        [row["z_t"][pmcid] for _dimension, row in dimensions_df.iterrows()],
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
    state_count = int(tf.shape(trace_data_list[0][0])[1].numpy())
    state_prior_s = tf.zeros(state_count, dtype=tf.float32)
    emission_bs = tf.zeros((progress_bins, state_count), dtype=tf.float32)
    counts_bbss = tf.zeros(
        (progress_bins, progress_bins, state_count, state_count),
        dtype=tf.float32,
    )
    for state_is, progress_i in trace_data_list:
        progress_ib = tf.one_hot(progress_i, progress_bins)
        state_prior_s = state_prior_s + tf.reduce_sum(state_is, axis=0)
        emission_bs = emission_bs + tf.einsum(
            "ib,is->bs", progress_ib, state_is
        )
        counts_bbss = counts_bbss + tf.einsum(
            "ia,ib,ir,is->abrs",
            progress_ib[:-1],
            progress_ib[1:],
            state_is[:-1],
            state_is[1:],
        )
    state_prior_s = tf.maximum(state_prior_s, 1e-12)
    state_prior_s = state_prior_s / tf.reduce_sum(state_prior_s)
    emission_bs = emission_bs + 0.1 * state_prior_s[None, :]
    emission_bs = tf.linalg.normalize(emission_bs, ord=1, axis=1)[0]
    counts_bbss = counts_bbss + tf.tile(
        0.1 * emission_bs[None, :, None, :],
        (progress_bins, 1, state_count, 1),
    )
    transition_bbss = tf.linalg.normalize(counts_bbss, ord=1, axis=3)[0]
    bin_b = tf.range(progress_bins, dtype=tf.float32)
    result_list = []
    for start in range(0, len(trace_data_list), 8):
        batch = trace_data_list[start : start + 8]
        length_t = tf.convert_to_tensor(
            [len(state_is) for state_is, _progress_i in batch],
            dtype=tf.int32,
        )
        state_its = tf.ragged.stack(
            [state_is for state_is, _progress_i in batch]
        ).to_tensor()
        max_length = int(tf.shape(state_its)[1].numpy())
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
                "tar,tab,abrs->tbs",
                joint_tbs,
                transition_tbb,
                transition_bbss,
            )
            update_tbs = update_tbs * likelihood_its[:, i, None, :]
            update_tbs = update_tbs / tf.maximum(
                tf.reduce_sum(update_tbs, axis=[1, 2], keepdims=True),
                1e-12,
            )
            joint_tbs = tf.where(
                valid_it[:, i, None, None], update_tbs, joint_tbs
            )
            phase_rows.append(tf.reduce_sum(joint_tbs, axis=2))
        phase_itb = tf.stack(phase_rows, axis=1)
        result_list.extend(
            phase_itb[trace_i, :length]
            for trace_i, length in enumerate(length_t.numpy())
        )
    return result_list


def universal_group_phase(
    sentences_df,
    dimensions_df,
    devices,
    transition_bandwidth,
    progress_bins,
):
    trace_data_list = []
    group_list = sorted(pd.unique(dimensions_df["group"]))
    for trace_i, (trace_key, trace_df) in enumerate(
        sentences_df.groupby(level=["pmcid", "model"], sort=True)
    ):
        with tf.device(devices[trace_i % len(devices)]):
            pmcid, model = trace_key
            no_information_t = trace_df["is_no_information"].to_numpy(
                dtype=bool
            )
            group_dimensions = dict(
                tuple(dimensions_df.loc[model].groupby("group", sort=True))
            )
            z_ig = tf.stack(
                [
                    (
                        tf.reduce_max(
                            trace_z_id(group_dimensions[group], pmcid), axis=1
                        )
                        if group in group_dimensions
                        else tf.fill((int((~no_information_t).sum()),), -1e9)
                    )
                    for group in group_list
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
    embedding_de=None,
    semantic_weight=0.0,
    entropy_weight=0.0,
    return_metrics=False,
):
    steps = 80
    case_batch_size = 8
    evaluation_case_count = 256
    max_beta = 4.0
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
        model_group_mass_m = tf.reduce_sum(model_mask_md, axis=1) / len(group_g)
        if embedding_de is not None:
            relatedness_dd = tf.maximum(
                tf.matmul(embedding_de, embedding_de, transpose_b=True), 0.0
            )
            cross_model_dd = 1.0 - tf.matmul(
                model_mask_md, model_mask_md, transpose_a=True
            )
    case_rows = {}
    for trace_i, ((model, pmcid), trace_df) in enumerate(
        sentences_df.groupby(level=["model", "pmcid"], sort=True)
    ):
        with tf.device(devices[trace_i % len(devices)]):
            model_dims_df = dimensions_df.loc[model]
            no_information_t = trace_df["is_no_information"].to_numpy(
                dtype=bool
            )
            case_rows.setdefault(pmcid, []).append(
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
    case_rows = list(case_rows.values())

    def objective(assignment_dg, batch):
        rows = [row for case in batch for row in case]
        state_list = []
        activation_list = []
        for i, row in enumerate(rows):
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
                semantic_progress_i = tf.boolean_mask(
                    row["progress_i"], ~row["no_information_t"]
                )
                activation_list.append(
                    tf.math.unsorted_segment_mean(
                        group_z_ig,
                        semantic_progress_i,
                        progress_bins,
                    )
                )
                state_list.append(
                    trace_state_is(group_z_ig, row["no_information_t"])
                )
        phase_rows = fit_phases(
            list(zip(state_list, [row["progress_i"] for row in rows])),
            transition_bandwidth,
            progress_bins,
        )
        progress_total = tf.add_n(
            [
                tf.reduce_sum(
                    tf.einsum(
                        "ib,bc,ic->i",
                        phase_ib,
                        kernel_bb,
                        row["target_ib"],
                    )
                    / tf.sqrt(
                        tf.einsum(
                            "ib,bc,ic->i",
                            phase_ib,
                            kernel_bb,
                            phase_ib,
                        )
                        * tf.einsum(
                            "ib,bc,ic->i",
                            row["target_ib"],
                            kernel_bb,
                            row["target_ib"],
                        )
                    )
                )
                for row, phase_ib in zip(rows, phase_rows)
            ]
        )
        correlation_list = []
        offset = 0
        for case in batch:
            activation_pbg = tf.stack(
                activation_list[offset : offset + len(case)]
            )
            offset += len(case)
            centered_pbg = activation_pbg - tf.reduce_mean(
                activation_pbg, axis=1, keepdims=True
            )
            normalized_pbg = tf.math.l2_normalize(centered_pbg, axis=1)
            correlation_ppg = tf.einsum(
                "pbg,qbg->pqg", normalized_pbg, normalized_pbg
            )
            correlation_list.append(
                tf.reduce_mean(
                    tf.boolean_mask(
                        correlation_ppg,
                        ~tf.eye(len(case), dtype=tf.bool),
                    )
                )
            )
        return (
            progress_total,
            sum(len(phase_ib) for phase_ib in phase_rows),
            tf.add_n(correlation_list),
            len(correlation_list),
        )

    for step in range(steps + 1):
        temperature = 0.1 + 1.9 * max(0.0, 1.0 - step / steps)
        if case_batch_size < len(case_rows):
            batch_i = tf.random.shuffle(tf.range(len(case_rows)))[
                :case_batch_size
            ].numpy()
            batch = [case_rows[i] for i in batch_i]
        else:
            batch = case_rows
        with tf.GradientTape() as tape:
            assignment_dg = tf.exp(logits / temperature)
            for _ in range(10):
                assignment_dg = assignment_dg / tf.matmul(
                    model_mask_md,
                    tf.matmul(model_mask_md, assignment_dg)
                    / model_group_mass_m[:, None],
                    transpose_a=True,
                )
                assignment_dg = assignment_dg / tf.reduce_sum(
                    assignment_dg, axis=1, keepdims=True
                )
            (
                progress_total,
                progress_count,
                correlation_total,
                correlation_count,
            ) = objective(assignment_dg, batch)
            progress_similarity = progress_total / progress_count
            correlation = correlation_total / correlation_count
            loss = -0.5 * (progress_similarity + correlation)
            assignment_entropy = -tf.reduce_mean(
                tf.reduce_sum(
                    assignment_dg * tf.math.log(assignment_dg + 1e-12),
                    axis=1,
                )
            ) / tf.math.log(tf.cast(len(group_g), tf.float32))
            loss = loss - entropy_weight * assignment_entropy
            semantic_similarity = tf.constant(0.0)
            if embedding_de is not None:
                coassignment_dd = tf.matmul(
                    assignment_dg, assignment_dg, transpose_b=True
                )
                semantic_similarity = tf.reduce_sum(
                    cross_model_dd * relatedness_dd * coassignment_dd
                ) / tf.reduce_sum(cross_model_dd * coassignment_dd)
                loss = loss - semantic_weight * semantic_similarity
        if step < steps:
            gradients = tape.gradient(loss, [logits])
            optimizer.apply_gradients(zip(gradients, [logits]))
    groups_t = project_groups(assignment_dg, dimensions_df, group_g)
    if not return_metrics:
        return groups_t
    progress_total = tf.constant(0.0)
    correlation_total = tf.constant(0.0)
    progress_count = 0
    correlation_count = 0
    evaluation_rows = [
        case_rows[i]
        for i in np.linspace(
            0,
            len(case_rows) - 1,
            min(evaluation_case_count, len(case_rows)),
            dtype=int,
        )
    ]
    for start in range(0, len(evaluation_rows), case_batch_size):
        batch_metrics = objective(
            assignment_dg, evaluation_rows[start : start + case_batch_size]
        )
        progress_total += batch_metrics[0]
        progress_count += batch_metrics[1]
        correlation_total += batch_metrics[2]
        correlation_count += batch_metrics[3]
    progress_similarity = progress_total / progress_count
    correlation = correlation_total / correlation_count
    semantic_similarity = tf.constant(0.0)
    if embedding_de is not None:
        coassignment_dd = tf.matmul(
            assignment_dg, assignment_dg, transpose_b=True
        )
        semantic_similarity = tf.reduce_sum(
            cross_model_dd * relatedness_dd * coassignment_dd
        ) / tf.reduce_sum(cross_model_dd * coassignment_dd)
    loss = -0.5 * (progress_similarity + correlation)
    assignment_entropy = -tf.reduce_mean(
        tf.reduce_sum(
            assignment_dg * tf.math.log(assignment_dg + 1e-12), axis=1
        )
    ) / tf.math.log(tf.cast(len(group_g), tf.float32))
    loss = loss - entropy_weight * assignment_entropy
    loss = loss - semantic_weight * semantic_similarity
    return groups_t, {
        "loss": float(loss.numpy()),
        "progress_similarity": float(progress_similarity.numpy()),
        "activation_correlation": float(correlation.numpy()),
        "semantic_similarity": float(semantic_similarity.numpy()),
        "assignment_entropy": float(assignment_entropy.numpy()),
    }


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
            [
                semantic_df["group_index"],
                semantic_df.index.get_level_values("model"),
            ],
            sort=True,
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
                dimension_labels=[
                    {
                        "model": model,
                        "dimension": int(dimension),
                        "title": row["title"],
                        "description": row["description"],
                    }
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
    transition_bandwidth=1.0,
    progress_bins=12,
    group_count=11,
    group_count_range=None,
    semantic_embedding_source="sentence",
    semantic_case_count=128,
    semantic_weight=0.1,
    entropy_weight=0.0,
    refresh=None,
    clean=None,
    skip=None,
    until=None,
    read_only=False,
):
    os.makedirs(output_dir, exist_ok=True)
    refresh = list(refresh or [])
    with pypelite.pipeline(
        output_dir,
        refresh=refresh,
        clean=clean,
        skip=skip,
        until=until,
        read_only=read_only,
    ):
        tf.random.set_seed(42)
        np.random.seed(42)
        devices = [
            device.name for device in tf.config.list_logical_devices("GPU")
        ] or ["/CPU:0"]
        print("tensorflow_devices", ", ".join(devices), flush=True)
        sentences_df = load_inputs(sentences_path)
        judgements_path = os.path.join(output_dir, "sentence_judgements.csv")
        if not {"all", "sentence_judgements"}.intersection(
            refresh
        ) and os.path.exists(judgements_path):
            judgements_df = pd.read_csv(
                judgements_path,
                dtype={"pmcid": str, "model": str, "sentence_index": int},
            )
        else:
            judgements_df = pd.DataFrame(
                load_sentence_judgements(
                    [
                        {
                            "pmcid": pmcid,
                            "model": model,
                            "sentence_index": int(sentence_index),
                            "text": row["text"],
                        }
                        for (
                            pmcid,
                            model,
                            sentence_index,
                        ), row in sentences_df.iterrows()
                    ]
                )
            )
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
        dimension_labels_df = pd.concat(
            dimension_labels(
                dimensions_df.reset_index()[["model", "dimension"]].assign(
                    pool=[
                        label_pool_by_dimension[(model, int(dimension))]
                        for model, dimension in dimensions_df.index
                    ]
                )
            ),
            axis=1,
        ).T
        dimension_labels_df.index = pd.MultiIndex.from_tuples(
            dimension_labels_df.index,
            names=["model", "dimension"],
        )
        dimensions_df = dimensions_df.join(
            dimension_labels_df[["title", "description"]]
        )
        if semantic_embedding_source == "summary":
            embedding_de = dimension_embeddings(
                dimensions_df[["title", "description"]]
            )
        elif semantic_embedding_source == "sentence":
            semantic_pmcids = sorted(
                sentences_df.index.get_level_values("pmcid").unique()
            )[:semantic_case_count]
            semantic_df = sentences_df[
                sentences_df.index.get_level_values("pmcid").isin(
                    semantic_pmcids
                )
                & ~sentences_df["is_no_information"].to_numpy(dtype=bool)
            ]
            embedding_de = sentence_steering_vectors(
                semantic_df,
                dimensions_df,
                sentence_embeddings(semantic_df[["text"]]),
            )
        else:
            raise ValueError(
                "semantic embedding source must be 'sentence' or 'summary'"
            )
        np.save(
            os.path.join(output_dir, "dimension_semantic_embeddings.npy"),
            embedding_de,
        )
        max_dimension_count = dimensions_df.groupby(level="model").size().max()
        if group_count_range:
            if (
                group_count_range[0] < 1
                or group_count_range[0] > group_count_range[1]
            ):
                raise ValueError(
                    "group count range must be positive and increasing"
                )
            if group_count_range[0] > max_dimension_count:
                raise ValueError(
                    "group count range exceeds available dimensions"
                )
            group_counts = range(
                group_count_range[0],
                min(group_count_range[1], int(max_dimension_count)) + 1,
            )
        else:
            group_counts = [min(int(group_count), int(max_dimension_count))]
        sweep_rows = []
        groups_by_count = {}
        for candidate_count in group_counts:
            tf.random.set_seed(42)
            np.random.seed(42)
            initial_groups_t = (
                dimensions_df.groupby(level="model").cumcount()
                % candidate_count
            ).rename("group")
            groups_by_count[candidate_count], metrics = relaxed_progress_groups(
                sentences_df=sentences_df,
                dimensions_df=dimensions_df,
                initial_groups_t=initial_groups_t,
                devices=devices,
                transition_bandwidth=transition_bandwidth,
                progress_bins=progress_bins,
                embedding_de=embedding_de,
                semantic_weight=semantic_weight,
                entropy_weight=entropy_weight,
                return_metrics=True,
            )
            sweep_rows.append({"group_count": candidate_count, **metrics})
            print("group_count_metrics", json.dumps(sweep_rows[-1]), flush=True)
        selected = min(sweep_rows, key=lambda row: row["loss"])
        dimensions_df["group"] = groups_by_count[selected["group_count"]]
        with open(
            os.path.join(output_dir, "group_count_sweep.json"),
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                {
                    "objective": (
                        "-0.5 * (progress_similarity + "
                        "activation_correlation) - semantic_weight * "
                        "semantic_similarity - entropy_weight * "
                        "assignment_entropy"
                    ),
                    "semantic_weight": semantic_weight,
                    "entropy_weight": entropy_weight,
                    "selected_group_count": selected["group_count"],
                    "rows": sweep_rows,
                },
                output_file,
                indent=2,
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
        data = output_data(
            output_dir, sentences_df, dimensions_df, labels_by_group
        )
        data_path = os.path.join(output_dir, "data.json")
        with open(data_path, "w", encoding="utf-8") as output_file:
            json.dump(data, output_file, separators=(",", ":"))
        print("data=" + data_path)


if __name__ == "__main__":
    parser = pypelite.argument_parser()
    parser.add_argument("sentences_path")
    parser.add_argument("output_dir")
    parser.add_argument("--group-count", type=int, default=11)
    parser.add_argument("--group-count-range", nargs=2, type=int)
    parser.add_argument(
        "--semantic-embedding-source",
        choices=["sentence", "summary"],
        default="sentence",
    )
    parser.add_argument("--semantic-case-count", type=int, default=128)
    parser.add_argument("--semantic-weight", type=float, default=0.1)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    run(**vars(parser.parse_args()))
