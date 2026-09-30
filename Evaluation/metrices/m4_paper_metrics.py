import os
import tiktoken
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from rouge_score import rouge_scorer
from bert_score import score as calc_bert_score

# ---------------------------------------------------------
# Secure NLTK Download Bypass
# ---------------------------------------------------------
# Create an isolated, secure folder inside the project to bypass Linux permission errors
nltk_dir = os.path.join(os.path.dirname(__file__), 'nltk_data_secure')
os.makedirs(nltk_dir, exist_ok=True)
os.chmod(nltk_dir, 0o700)  # Force strict private permissions (rwx------)

nltk.data.path.append(nltk_dir)
nltk.download('punkt', download_dir=nltk_dir, quiet=True)
nltk.download('punkt_tab', download_dir=nltk_dir, quiet=True)

try:
    import mauve
except ImportError:
    mauve = None
    print("Warning: 'mauve-text' library not found.")
# ---------------------------------------------------------



# 1. ROUGE-L Metric (Paper match: use_stemmer=True)
class PaperMatchedRougeL(BaseMetric):
    def __init__(self, threshold=14.0):
        self.threshold = threshold
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def measure(self, test_case: LLMTestCase):
        scores = self.scorer.score(test_case.expected_output, test_case.actual_output)
        self.score = round(scores['rougeL'].fmeasure * 100, 2)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "ROUGE-L (%)"


# 2. BLEU-4 Metric (Standard 4-gram overlap)
class PaperMatchedBLEU4(BaseMetric):
    def __init__(self, threshold=1.0):
        self.threshold = threshold
        self.smooth = SmoothingFunction().method1

    def measure(self, test_case: LLMTestCase):
        ref = [nltk.word_tokenize(test_case.expected_output.lower())]
        hyp = nltk.word_tokenize(test_case.actual_output.lower())
        
        # Calculate 4-gram BLEU score
        bleu = sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smooth)
        self.score = round(bleu * 100, 2)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "BLEU-4 (%)"


# 3. BERTScore Metric (Semantic Similarity)
class PaperMatchedBERTScore(BaseMetric):
    def __init__(self, model_type="bert-base-uncased", threshold=45.0):
        self.model_type = model_type
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        # Calculates contextual embedding similarity using the standard BERT model
        P, R, F1 = calc_bert_score(
            [test_case.actual_output], 
            [test_case.expected_output], 
            model_type=self.model_type, 
            lang="en", 
            verbose=False
        )
        self.score = round(F1.item() * 100, 2)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "BERTScore (%)"


# 4. MAUVE Metric (Distributional divergence from human text)
class PaperMatchedMAUVE(BaseMetric):
    def __init__(self, threshold=60.0):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        if mauve is None:
            raise ImportError("MAUVE is not installed.")
            
        # MAUVE is technically a corpus-level metric. Running it on single sentences 
        # requires setting a very small quantization bucket size to avoid clustering errors.
        try:
            out = mauve.compute_mauve(
                p_text=[test_case.actual_output], 
                q_text=[test_case.expected_output], 
                device_id=-1, # -1 forces CPU execution to save VRAM
                max_text_length=256,
                num_buckets=2, # Minimized for single-test-case execution
                verbose=False
            )
            self.score = round(out.mauve * 100, 2)
        except Exception as e:
            # Fallback if text is too short to cluster
            print(f"MAUVE calculation bypassed for this tiny snippet: {e}")
            self.score = 0.0

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "MAUVE (%)"


# 5. Token Compression Ratio (Ensures >2.0x Reduction)
class TokenCompressionFactorMetric(BaseMetric):
    def __init__(self, target_factor=2.0):
        self.target_factor = target_factor

    def measure(self, test_case: LLMTestCase):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Extract the texts dynamically from the DeepEval test case fields
        raw_text = test_case.context[0] if test_case.context else ""
        compressed_text = test_case.retrieval_context[0] if test_case.retrieval_context else ""
        
        raw_tokens = len(tokenizer.encode(raw_text))
        compressed_tokens = len(tokenizer.encode(compressed_text))
        
        if compressed_tokens == 0:
            self.score = 0.0
        else:
            self.score = round(raw_tokens / compressed_tokens, 2)
            
        self.success = self.score >= self.target_factor
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Compression Factor (x)"
    


####### update

import collections

# BLEU-1 for LOCOMO (Table III)
class PaperMatchedBLEU1(BaseMetric):
    def __init__(self, threshold=15.0):
        self.threshold = threshold
        self.smooth = SmoothingFunction().method1

    def measure(self, test_case: LLMTestCase):
        ref = [nltk.word_tokenize(test_case.expected_output.lower())]
        hyp = nltk.word_tokenize(test_case.actual_output.lower())
        # Calculate 1-gram BLEU score
        bleu = sentence_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=self.smooth)
        self.score = round(bleu * 100, 2)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "BLEU-1 (%)"

# Token F1 Score for LOCOMO (Table III)
class PaperMatchedF1(BaseMetric):
    def __init__(self, threshold=25.0):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        ref_tokens = nltk.word_tokenize(test_case.expected_output.lower())
        hyp_tokens = nltk.word_tokenize(test_case.actual_output.lower())
        
        common = collections.Counter(ref_tokens) & collections.Counter(hyp_tokens)
        num_same = sum(common.values())
        
        if len(ref_tokens) == 0 or len(hyp_tokens) == 0 or num_same == 0:
            self.score = 0.0
        else:
            precision = 1.0 * num_same / len(hyp_tokens)
            recall = 1.0 * num_same / len(ref_tokens)
            self.score = round((2 * precision * recall) / (precision + recall) * 100, 2)
            
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "F1 Score (%)"