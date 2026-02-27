from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from fishfarm.models import GenerationRequest, Message, Model
from fishfarm.tasks.base import Task, TaskResult


def get_accuracy(generated_text: str, target_text: str) -> int:
    if str(generated_text).strip().strip(":`'\"(.) ").lower() == str(target_text).strip().strip(":`'\"(.) ").lower():
        return 1
    try:
        if str(float(generated_text)) == str(target_text).strip():  # For example, "1.0" == "1"
            return 1
        return 0
    except:
        return 0


def get_choice(txt: str) -> str:
    txt = str(txt).strip().strip(":`'\"(.) ").lower()
    CHOICES = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
    ]
    for choice in CHOICES:
        if txt.startswith(choice):
            return choice


def get_choice_accuracy(generated_text: str, target_text: str) -> int:
    if get_choice(generated_text) == get_choice(target_text):
        return 1
    return 0


def get_bool_value_from_text(text):
    """Returns None if there was no meaningful boolean value that could be found."""
    # Convert to string if necessary.
    text = str(text)
    if "1" in text:
        return True
    if "0" in text:
        return False
    if "yes" in text.lower():
        return True
    if "no" in text.lower():
        return False
    if "true" in text.lower():
        return True
    if "false" in text.lower():
        return False
    if "positive" in text.lower():
        return True
    if "negative" in text.lower():
        return False
    if "valid" in text.lower():
        return True
    if "invalid" in text.lower():
        return False
    return None


def get_binary_accuracy_flex(generated_text, target_text):
    """Returns 1 if the generated text and target text are equal in boolean space.

    This is a flexible matching function that can handle a variety of boolean representations.
    - yes/no
    - true/false
    - 1/0
    - positive/negative
    - valid/invalid
    """
    generated_prediction = get_bool_value_from_text(generated_text)
    if generated_prediction is None:
        return 0

    target_prediction = get_bool_value_from_text(target_text)
    if target_prediction is None:
        return 0

    return int(generated_prediction == target_prediction)


@dataclass
class QASample:
    question: str
    answer: str


class QATask(Task):
    def __init__(
        self,
        samples: Sequence[QASample],
        eval_fn: Callable,
        context_messages: Sequence[Message] = (),
    ) -> None:
        self.samples = list(samples)
        self.eval_fn = eval_fn
        self.context_messages = context_messages

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def evaluate(self, model: Model, sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]
        requests = []
        for sample in samples:
            messages = list(self.context_messages)
            messages.append(Message(role="user", content=sample.question))
            requests.append(GenerationRequest(messages=messages))

        sample_details = []
        for sample, result in zip(samples, model.generate(requests)):
            output = result.generation
            is_correct = self.eval_fn(output, sample.answer)
            details = dict(problem=sample.question, output=output, answer=sample.answer, is_correct=is_correct)
            sample_details.append(details)

        agg_metrics = dict(acc=sum(sample["is_correct"] for sample in sample_details) / len(sample_details))
        return TaskResult(aggregate_metrics=agg_metrics, sample_details=sample_details)


TASK_EVAL_FNS = {
    "winogrande": get_choice_accuracy,
    "boolq": get_binary_accuracy_flex,
    "piqa": get_choice_accuracy,
    "hellaswag": get_choice_accuracy,
    "arc": get_choice_accuracy,
    "openbookqa": get_choice_accuracy,
}
