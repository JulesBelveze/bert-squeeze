from ...models.custom_transformers.theseus_bert import TheseusBertEncoder


class ConstantReplacementScheduler:
    """
    Scheduler implementing layer replacement for Theseus Bert.

    It replaces the original layers with the compressed ones using the
    replacement rate p defined as follows:
        p = r if t < replacing_steps else 1
    where r is the replacing rate and t is the current optimization step.

    Args:
        bert_encoder (TheseusBertEncoder):
            Encoder of Theseus Bert
        replacing_rate (float):
            Probability to use within the Bernoulli distribution for layer
            replacement.
        replacing_steps (int):
            Number of steps after which the replacing_rate is set to 1.
            This means that the model will only use the compressed layers.
    """

    def __init__(
        self,
        bert_encoder: TheseusBertEncoder,
        replacing_rate: float,
        replacing_steps: int = None,
    ):
        self.bert_encoder = bert_encoder
        self.replacing_rate = replacing_rate
        self.replacing_steps = replacing_steps
        self.step_counter = 0
        self.bert_encoder.set_replacing_rate(replacing_rate)

    def step(self) -> float:
        """
        Increments the scheduler and will set the replacing_rate to 1.0 when
        t >= replacing_steps otherwise `self.replacing_rate`.

        Returns:
            float:
                Replacing rate
        """
        self.step_counter += 1
        if self.replacing_steps is None or self.replacing_rate == 1.0:
            return self.replacing_rate
        else:
            if self.step_counter >= self.replacing_steps:
                self.bert_encoder.set_replacing_rate(1.0)
                self.replacing_rate = 1.0
            return self.replacing_rate


class LinearReplacementScheduler:
    """
    Scheduler implementing layer replacement for Theseus Bert

    It replaces the original layers with the compressed ones using the
    replacement rate p defined as follow:
        p = min(1, kt + b)
    where k > 0 is the coefficient, b is the basic replacement rate and
    t the current optimization step.

    Args:
        bert_encoder (TheseusBertEncoder):
            Encoder of Theseus Bert
        base_replacing_rate (float):
            Probability to use within the Bernoulli distribution for layer
            replacement.
        coefficient (int):
            Coefficient of the linear scheduler (slope in kt+b).
    """

    def __init__(
        self,
        bert_encoder: TheseusBertEncoder,
        base_replacing_rate: float,
        coefficient: int,
    ):
        self.bert_encoder = bert_encoder
        self.base_replacing_rate = base_replacing_rate
        self.step_counter = 0
        self.coefficient = coefficient
        self.bert_encoder.set_replacing_rate(base_replacing_rate)

    def step(self) -> float:
        """
        Increments the scheduler and will set the replacing_rate to 1.0 when
        t >= replacing_steps otherwise to `kt + b`.

        Returns:
            float:
                replacing rate
        """
        self.step_counter += 1
        current_replacing_rate = min(
            self.coefficient * self.step_counter + self.base_replacing_rate, 1.0
        )
        self.bert_encoder.set_replacing_rate(current_replacing_rate)
        return current_replacing_rate
