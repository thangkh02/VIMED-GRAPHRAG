
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from backend.app.core.config import settings
from backend.app.models.schemas import (
    StatementVerification,
    VerificationRequest,
    VerificationResponse,
)

# Label indices for roberta-large-mnli output logits.
# The model outputs logits in the order: contradiction, neutral, entailment.
_LABEL_CONTRADICTION = 0
_LABEL_NEUTRAL = 1
_LABEL_ENTAILMENT = 2


class SelfReflectiveCritic:
    """NLI-based verification critic for the Self-MedRAG pipeline.

    Loads a local NLI model once at initialisation time, then exposes
    a ``verify`` method that scores rationale statements against a set
    of context passages and decides whether the overall reasoning is
    sufficiently supported.

    Attributes:
        device: The torch device used for inference (cuda or cpu).
        tau: Per-statement entailment threshold (default 0.5).
        theta: Overall passage-level pass/fail threshold (default 0.7).
    """

    def __init__(self) -> None:
        """Load the NLI model and tokenizer.

        The model is placed on a CUDA device when available, otherwise
        it falls back to CPU.  Thresholds are read from the central
        application settings so they can be adjusted without code
        changes.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_name: str = settings.NLI_MODEL_NAME
        self.tau: float = settings.NLI_SENTENCE_THRESHOLD
        self.theta: float = settings.NLI_PASSAGE_THRESHOLD

        print(f"[SelfReflectiveCritic] Loading NLI model: {model_name}")
        print(f"[SelfReflectiveCritic] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.model.to(self.device)
        self.model.eval()

        print("[SelfReflectiveCritic] Model loaded successfully.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_entailment_score(
        self, premise: str, hypothesis: str
    ) -> float:
        """Compute the entailment probability for a (premise, hypothesis) pair.

        Steps:
            1. Tokenize the pair with truncation to the model max length.
            2. Run a forward pass with gradients disabled.
            3. Apply softmax to the output logits.
            4. Return the probability assigned to the 'entailment' class.

        Args:
            premise: The context passage (evidence text).
            hypothesis: The rationale statement to verify.

        Returns:
            A float in [0, 1] representing entailment confidence.
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=-1)
        entailment_prob: float = probabilities[0, _LABEL_ENTAILMENT].item()

        return entailment_prob

    def _verify_statement(
        self, hypothesis: str, passages: List[str]
    ) -> StatementVerification:
        """Verify a single rationale statement against all passages.

        For each passage the entailment score is computed.  The passage
        with the highest score is selected as the best supporting
        evidence.  The statement is labelled 'Supported' when the best
        score meets or exceeds the sentence threshold (tau), otherwise
        it is labelled 'Unsupported'.

        Args:
            hypothesis: The rationale statement to verify.
            passages: All available context passages.

        Returns:
            A StatementVerification instance with the scoring result.
        """
        best_score: float = 0.0
        best_passage: str = ""

        for passage in passages:
            score = self._compute_entailment_score(
                premise=passage, hypothesis=hypothesis
            )
            if score > best_score:
                best_score = score
                best_passage = passage

        label = "Supported" if best_score >= self.tau else "Unsupported"

        return StatementVerification(
            statement=hypothesis,
            label=label,
            confidence_score=round(best_score, 4),
            best_passage=best_passage,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, request: VerificationRequest) -> VerificationResponse:
        """Run the full Self-MedRAG verification pipeline.

        Algorithm:
            1. For each rationale statement, cross-reference it with
               every context passage using NLI entailment scoring.
            2. Label each statement as Supported or Unsupported based
               on the sentence threshold (tau).
            3. Compute the overall support score S_i as the ratio of
               supported statements to total statements.
            4. Determine the final pass / fail verdict using the
               passage threshold (theta).

        Args:
            request: A VerificationRequest containing the list of
                rationale statements and context passages.

        Returns:
            A VerificationResponse with the verdict, scores, and
            per-statement breakdown.
        """
        supported: List[StatementVerification] = []
        unsupported: List[StatementVerification] = []

        for statement in request.statements:
            result = self._verify_statement(statement, request.passages)
            if result.label == "Supported":
                supported.append(result)
            else:
                unsupported.append(result)

        total = len(request.statements)
        support_score = len(supported) / total if total > 0 else 0.0
        is_passed = support_score >= self.theta

        return VerificationResponse(
            is_passed=is_passed,
            support_score=round(support_score, 4),
            supported_statements=supported,
            unsupported_statements=unsupported,
        )


# Module-level singleton (model loaded once on import).
verification_service = SelfReflectiveCritic()
