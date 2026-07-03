import unittest

from responses import extract_thinking_process, _strip_trailing_answer_line


class ExtractThinkingProcessTests(unittest.TestCase):
    def test_deepseek_output_template_trace(self):
        # This trace mirrors the example provided in the user request.
        trace = (
            "Read the following case presentation and give the most likely diagnosis.\n"
            "First, provide your internal reasoning for the diagnosis within the tags <think> ... </think>.\n"
            "Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>.\n\n"
            "----------------------------------------\n"
            "CASE PRESENTATION\n"
            "----------------------------------------\n"
            "A 27-year-old man presents to the emergency department with nausea and vomiting. The patient started experiencing these symptoms shortly after arriving home from going out to eat at a seafood restaurant. His symptoms progressed and now he reports having an odd metallic taste in his mouth, diffuse pruritus, and blurry vision. His temperature is 99.0¬∞F (37.2¬∞C), blood pressure is 120/72 mmHg, pulse is 50/min, respirations are 17/min, and oxygen saturation is 99% on room air. Physical exam reveals bradycardia and an inability of the patient to differentiate hot versus cold; no rash can be appreciated on exam. What is the most likely etiology of this patient‚Äôs symptoms?\n\n"
            "----------------------------------------\n"
            "OUTPUT TEMPLATE\n"
            "----------------------------------------\n"
            "<think>\n"
            "...your internal reasoning for the diagnosis...\n"
            "</think><answer>\n"
            "...the name of the disease/entity...\n"
            "</answer>\n\n"
            "<think>\n"
            "We need to read case: 27-year-old man, nausea, vomiting, metallic taste, diffuse pruritus, blurry vision. Temperature normal, BP normal, pulse 50/min (bradycardia). He has inability to differentiate hot vs cold (thermoesthesia). No rash. Symptoms after eating at seafood restaurant.\n\n"
            "Differential: Seafood poisoning: ciguatera fish poisoning, scombroid fish poisoning, paralytic shellfish poisoning (PSP). Ciguatera: associated with tropical fish (barracuda, snapper). Symptoms: GI (nausea, vomiting), neurological: paresthesias, reversal of temperature sensation (hot feels cold, cold feels hot), pruritus, metallic taste. Bradycardia? Could have bradycardia/hypotension. Also bradycardia maybe due to ciguatoxin. Ciguatera fish poisoning has these signs: GI, neurological, cardiorespiratory, etc. The key sign: altered temperature perception (hot/cold). Also pruritus. Metallic taste. So likely ciguatera.\n\n"
            "Other fish poisoning: scombroid fish poisoning: GI upset, rash, pruritus, but no temperature sensation reversal. Also no bradycardia.\n\n"
            "Paralytic shellfish poisoning (PSP): neurotoxic, cramps, tingling, GI, but not reversal of temperature.\n\n"
            "Anisakiasis? no.\n\n"
            "Also consider shellfish poisoning like domoic acid causes GI but not neurological. So ciguatera is best match.\n\n"
            "Thus diagnosis: ciguatera fish poisoning.\n\n"
            "Let's produce internal reasoning and answer.\n"
            "</think>\n\n"
            "<think>\n"
            "The patient is a young adult who developed gastrointestinal symptoms (nausea, vomiting) soon after dining at a seafood restaurant. The constellation of symptoms now includes an odd metallic taste, diffuse pruritus, blurred vision, bradycardia, and loss of temperature discrimination (inability to distinguish hot from cold). The absence of a rash and the specific sensory disturbance are key. Ciguatera fish poisoning is classically associated with these findings: GI upset, metallic taste, pruritus, neuro‑visceral manifestations such as hot‑cold reversal, blurred vision, and cardiac effects including bradycardia. Other seafood toxins (scombroid, paralytic shellfish) do not produce the hot‑cold reversal. Thus, the most likely diagnosis is ciguatera fish poisoning.\n"
            "<answer>Ciguatera fish poisoning</answer>"
        )

        reasoning = extract_thinking_process(trace)

        # Basic structural checks for the extracted reasoning trace
        self.assertTrue(
            reasoning.startswith("We need to read case: 27-year-old man"),
            msg=f"Unexpected start of reasoning: {reasoning[:120]!r}",
        )
        self.assertIn("Thus diagnosis: ciguatera fish poisoning.", reasoning)
        self.assertIn("Let's produce internal reasoning and answer.", reasoning)
        self.assertNotIn("<answer>", reasoning)
        self.assertNotIn("<think>", reasoning)

    def test_simple_think_block(self):
        trace = "prefix\n<think>\nMy reasoning.\n</think>\n<answer>Final</answer>"
        reasoning = extract_thinking_process(trace)
        self.assertEqual("My reasoning.", reasoning.strip())

    def test_final_response_header_removed(self):
        trace = (
            "prefix\n<think>\nLine one.\n## Final Response\nLine two.\n"
            "</think>\n<answer>X</answer>"
        )
        reasoning = extract_thinking_process(trace)
        self.assertNotIn("## Final Response", reasoning)
        self.assertIn("Line one.", reasoning)
        self.assertIn("Line two.", reasoning)

    def test_final_response_crlf_and_nbsp(self):
        trace = (
            "x\n<think>\nbody\r\n##\xa0Final  Response\r\n"
            "</think>\n<answer>y</answer>"
        )
        reasoning = extract_thinking_process(trace)
        self.assertNotIn("Final Response", reasoning)
        self.assertEqual("body", reasoning)

    def test_chat_marker_prefix_and_trailing_answer(self):
        # Ensure we drop everything before "]</answer><｜Assistant｜><think>\n"
        # and also drop any final <answer>...</answer> line from the result.
        trace = (
            "system and user chatter\n"
            "]</answer><｜Assistant｜><think>\n"
            "Reasoning line one.\n"
            "Reasoning line two.\n"
            "</think>\n"
            "<answer>Final diagnosis</answer>\n"
        )

        reasoning = extract_thinking_process(trace)

        self.assertTrue(reasoning.startswith("Reasoning line one."))
        self.assertIn("Reasoning line two.", reasoning)
        # Final <answer> line should be removed entirely
        self.assertNotIn("Final diagnosis", reasoning)
        self.assertNotIn("<answer>", reasoning)

    def test_ascii_assistant_marker_removed(self):
        trace = (
            "noise and system\n"
            "</answer><| Assistant |><think>\n"
            "Reasoning A.\n"
            "</think>\n"
        )
        reasoning = extract_thinking_process(trace)
        self.assertIn("Reasoning A.", reasoning)
        self.assertNotIn("</answer><| Assistant |><think>", reasoning)

    def test_exact_fullwidth_assistant_marker_removed(self):
        trace = (
            "prefix stuff\n"
            "</answer><｜Assistant｜><think>\n"
            "Reasoning B line.\n"
            "</think>\n"
        )
        reasoning = extract_thinking_process(trace)
        self.assertIn("Reasoning B line.", reasoning)
        self.assertNotIn("</answer><｜Assistant｜><think>\n", reasoning)

    def test_full_deepseek_template_plus_think_marker(self):
        # Ensure we drop everything before the full template block:
        # "<think>...internal reasoning...</think><answer>...name...</answer>\n<think>\n"
        trace = (
            "preamble\n"
            "<think>\n...your internal reasoning for the diagnosis...\n</think><answer>\n"
            "...the name of the disease/entity...\n</answer>\n<think>\n"
            "Actual reasoning first line.\n"
            "Second reasoning line.\n"
            "</think>\n<answer>Final diagnosis</answer>\n"
        )

        reasoning = extract_thinking_process(trace)

        self.assertTrue(reasoning.startswith("Actual reasoning first line."))
        self.assertIn("Second reasoning line.", reasoning)
        # Template markers should not appear in the extracted reasoning
        self.assertNotIn("...your internal reasoning for the diagnosis...", reasoning)
        self.assertNotIn("...the name of the disease/entity...", reasoning)
        self.assertNotIn("<think>\n...your internal reasoning for the diagnosis...", reasoning)

    def test_bare_fullwidth_assistant_think_removed(self):
        trace = (
            "<think>\n"
            "Reasoning before marker.\n"
            "<Assistant0><think>"
            "Reasoning after marker.\n"
            "</think>\n"
        )

        reasoning = extract_thinking_process(trace)

        self.assertIn("Reasoning before marker.", reasoning)
        self.assertIn("Reasoning after marker.", reasoning)
        self.assertNotIn("<Assistant0><think>", reasoning)

    def test_answer_assistant_thinking_prefix(self):
        trace = (
            "chat log stuff\n"
            "</answer>assistant\n\n## Thinking\n\n"
            "This is the reasoning body.\nMore lines here.\n"
        )
        reasoning = extract_thinking_process(trace)
        self.assertTrue(reasoning.startswith("This is the reasoning body."))
        self.assertNotIn("</answer>assistant", reasoning)
        self.assertNotIn("## Thinking", reasoning)

    def test_prefix_fullwidth_assistant_wrapping_think_block(self):
        # Case where the full response starts with the assistant marker
        # followed by a single <think>...</think> block.
        trace = (
            "<｜Assistant｜><think>\n"
            "Okay, so I'm trying to figure out the most likely diagnosis for this patient based on the information provided."
            "\nLet me go through the details step by step.\n"
            "First, the patient is a 35-year-old woman with progressive urinary leakage.\n"
            "</think>\n\nThe most likely diagnosis for the patient is a urinary tract infection (UTI)."
        )

        reasoning = extract_thinking_process(trace)

        # Marker and raw think tags should be gone
        self.assertNotIn("<｜Assistant｜><think>", reasoning)
        self.assertNotIn("<think>", reasoning)
        self.assertNotIn("</think>", reasoning)
        # Reasoning should start from the natural language, not from a tag
        self.assertTrue(
            reasoning.startswith("Okay, so I'm trying to figure out the most likely diagnosis"),
            msg=f"Unexpected reasoning start: {reasoning[:120]!r}",
        )


    def test_trailing_eos_does_not_leak_answer(self):
        # skip_special_tokens=False keeps a terminal EOS after the final </answer>. Every
        # model's EOS variant must be stripped so the trailing-answer strip still fires and the
        # final diagnosis does not leak into the reasoning.
        base = (
            "preamble\n"
            "<think>\n...your internal reasoning for the diagnosis...\n</think><answer>\n"
            "...the name of the disease/entity...\n</answer>\n<think>\n"
            "Actual reasoning first line.\n"
            "Second reasoning line.\n"
            "</think>\n<answer>Final diagnosis</answer>"
        )
        for eos in (
            "<｜end▁of▁sentence｜>",           # deepseek fullwidth (also its pad)
            "<｜end▁of▁sentence｜><｜end▁of▁sentence｜>",  # + padding repeat
            "<|eot_id|>",                       # huatuo / llama-3
            "<|im_end|>",                       # qwq / qwen2
            "<|eot_id|><|end_of_text|>",        # eos then pad
        ):
            with self.subTest(eos=eos):
                reasoning = extract_thinking_process(base + eos)
                self.assertTrue(reasoning.startswith("Actual reasoning first line."))
                self.assertIn("Second reasoning line.", reasoning)
                self.assertNotIn("Final diagnosis", reasoning)
                self.assertNotIn("<answer>", reasoning)
                self.assertNotIn("end▁of▁sentence", reasoning)
                self.assertNotIn("<|", reasoning)

    def test_gpt_oss_harmony_token_form(self):
        # skip_special_tokens=False decode: the harmony channel markers survive verbatim.
        trace = (
            "<|start|>system<|message|>You are ChatGPT.<|end|>"
            "<|start|>user<|message|>Read the case presentation...<|end|>"
            "<|start|>assistant<|channel|>analysis<|message|>"
            "The biopsy shows intratubular vacuolization, typical of acute interstitial nephritis. "
            "NSAIDs are the usual cause.\n"
            "<|end|><|start|>assistant<|channel|>final<|message|>"
            "<answer>NSAID-induced acute interstitial nephritis</answer><|return|>"
        )
        reasoning = extract_thinking_process(trace)
        self.assertTrue(reasoning.startswith("The biopsy shows intratubular vacuolization"))
        self.assertIn("NSAIDs are the usual cause.", reasoning)
        # final-channel answer and all harmony/special tokens must be gone
        self.assertNotIn("NSAID-induced acute interstitial nephritis", reasoning)
        self.assertNotIn("<answer>", reasoning)
        self.assertNotIn("<|", reasoning)
        self.assertNotIn("analysis", reasoning)

    def test_gpt_oss_harmony_legacy_glued_form(self):
        # Backward compatibility: legacy skip_special_tokens=True files stored the channel names
        # glued onto "assistant" with the markers stripped. Still extract the analysis channel.
        trace = (
            "assistantanalysis"
            "The patient has ciguatera fish poisoning based on hot-cold reversal."
            "assistantfinal<answer>Ciguatera fish poisoning</answer>"
        )
        reasoning = extract_thinking_process(trace)
        self.assertTrue(reasoning.startswith("The patient has ciguatera fish poisoning"))
        self.assertNotIn("Ciguatera fish poisoning</answer>", reasoning)
        self.assertNotIn("assistantfinal", reasoning)


class StripTrailingAnswerLineTests(unittest.TestCase):
    def test_plain_balanced_block(self):
        self.assertEqual("reasoning.", _strip_trailing_answer_line("reasoning.\n<answer>Osteosarcoma</answer>"))

    def test_long_multiline_answer_block(self):
        # QwQ real answers can be long/multi-line; the whole balanced block must still go.
        s = ("The ECG and enzymes point to infarction.\n\n\n<answer>\nAcute ST-Elevation Myocardial "
             "Infarction with possible posterior and lateral involvement, leading to cardiogenic "
             "shock and arrhythmias\n</answer>")
        self.assertEqual("The ECG and enzymes point to infarction.", _strip_trailing_answer_line(s))

    def test_dangling_closer_then_real_block(self):
        # QwQ: a spurious </answer> precedes the real answer block.
        s = "...the most plausible diagnosis.\n</answer>\n<answer>\nFactitious hypoglycemia\n</answer>"
        out = _strip_trailing_answer_line(s)
        self.assertEqual("...the most plausible diagnosis.", out)
        self.assertNotIn("</answer>", out)

    def test_sandwiched_bare_answer(self):
        # QwQ: "</answer>\nPTSD\n</answer>" — bare answer between two dangling closers.
        s = "...anxiety disorders without trauma are less likely.\n</answer>\nPTSD\n</answer>"
        out = _strip_trailing_answer_line(s)
        self.assertEqual("...anxiety disorders without trauma are less likely.", out)
        self.assertNotIn("answer", out)
        self.assertNotIn("PTSD", out)

    def test_does_not_eat_earlier_quoted_tag(self):
        # A mid-reasoning quoted <answer>...</answer> must survive; only the trailing answer goes.
        s = "It said <answer>use the name</answer> so I reason further and conclude.\n<answer>Final Dx</answer>"
        out = _strip_trailing_answer_line(s)
        self.assertEqual("It said <answer>use the name</answer> so I reason further and conclude.", out)

    def test_no_trailing_answer_is_unchanged(self):
        s = "Reasoning that simply ends with a sentence."
        self.assertEqual(s, _strip_trailing_answer_line(s))


if __name__ == "__main__":
    unittest.main()
