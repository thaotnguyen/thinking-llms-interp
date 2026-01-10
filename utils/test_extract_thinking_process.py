import unittest

from responses import extract_thinking_process


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


if __name__ == "__main__":
    unittest.main()
