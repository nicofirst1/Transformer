from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch


@dataclass
class LoggingStrategy:
    store_source: bool = True
    store_labels: bool = True
    store_preds: bool = True

    def filtered_interaction(
            self,
            source: Optional[torch.Tensor],
            preds: Optional[torch.Tensor],
            labels: Optional[torch.Tensor],
            aux: Dict[str, torch.Tensor],
    ):
        return Interaction(
            source=source if self.store_source else None,
            preds=preds if self.store_preds else None,
            labels=labels if self.store_labels else None,
            aux=aux,
        )

    @classmethod
    def minimal(cls):
        args = [False] * 3
        return cls(*args)

    @classmethod
    def maximal(cls):
        return cls()


@dataclass
class Interaction:
    # incoming data
    source: Optional[torch.Tensor]
    preds: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]
    aux: Dict[str, torch.Tensor]

    @property
    def size(self):
        interaction_fields = [
            self.source,
            self.preds,
            self.labels,
        ]
        for t in interaction_fields:
            if t is not None:
                return t.size(0)
        raise RuntimeError("Cannot determine interaction log size; it is empty.")

    def to(self, *args, **kwargs) -> "Interaction":
        """Moves all stored tensor to a device. For instance, it might be not
        useful to store the interaction logs in CUDA memory."""

        def _to(x):
            if x is None or not torch.is_tensor(x):
                return x
            return x.to(*args, **kwargs)

        self.source = _to(self.source)
        self.preds = _to(self.preds)
        self.labels = _to(self.labels)

        if self.aux:
            self.aux = dict((k, _to(v)) for k, v in self.aux.items())

        return self

    @staticmethod
    def from_iterable(interactions: Iterable["Interaction"]) -> "Interaction":
        """
        >>> a = Interaction(torch.ones(1), None, None, torch.ones(1), torch.ones(1), None, {})
        >>> a.size
        1
        >>> b = Interaction(torch.ones(1), None, None, torch.ones(1), torch.ones(1), None, {})
        >>> c = Interaction.from_iterable((a, b))
        >>> c.size
        2
        >>> c
        Interaction(sender_input=tensor([1., 1.]), receiver_input=None, labels=None, message=tensor([1., 1.]), receiver_output=tensor([1., 1.]), message_length=None, aux={})
        >>> d = Interaction(torch.ones(1), torch.ones(1), None, torch.ones(1), torch.ones(1), None, {})
        >>> _ = Interaction.from_iterable((a, d)) # mishaped, should throw an exception
        Traceback (most recent call last):
        ...
        RuntimeError: Appending empty and non-empty interactions logs. Normally this shouldn't happen!
        """

        def _check_cat(lst):
            if all(x is None for x in lst):
                return None

            return lst

        assert interactions, "list must not be empty"
        assert all(len(x.aux) == len(interactions[0].aux) for x in interactions)

        aux = {}
        for k in interactions[0].aux:
            aux[k] = _check_cat([x.aux[k] for x in interactions])

        return Interaction(
            labels=_check_cat([x.labels for x in interactions]),
            preds=_check_cat([x.preds for x in interactions]),
            source=_check_cat([x.source for x in interactions]),
            aux=aux,
        )

    @staticmethod
    def empty() -> "Interaction":
        return Interaction(None, None, {})
