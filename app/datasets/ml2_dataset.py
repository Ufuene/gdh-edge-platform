"""
ml2_dataset.py

Define os registros destinados ao dataset do ML2.

O ML2 trabalha com contexto temporal.

Existem dois tipos de registros:

    label = 0
        operação normal

    label = 1
        Partial Shading Condition (PSC)

Regra de seleção:

    NORMAL
        somente dados cuja operação não apresenta desvio
        acima do limiar.

    PSC
        somente janelas efetivamente classificadas pelo ML2
        como diagnóstico 1.

Um caso:

    deviation_detected = True
    ml2_diagnosis = 0

é descartado.

Isso evita utilizar uma decisão possivelmente errada do ML2
como fonte de verdade para o dataset normal.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from app.data_manager.sample import Sample


@dataclass(slots=True)
class ML2DatasetWindow:
    """
    Representa uma janela temporal destinada ao dataset ML2.

    A janela contém as amostras efetivamente utilizadas no
    contexto temporal do ML2.
    """

    window_id: str

    window_start: int
    window_end: int

    label: int

    samples: list[Sample]

    @classmethod
    def from_window(
        cls,
        window_id: str,
        window_start: int,
        window_end: int,
        samples: Sequence[Sample],
        label: int,
    ) -> "ML2DatasetWindow":
        """
        Constrói uma janela ML2.

        Parameters
        ----------
        window_id:
            Identificador da janela.

        window_start:
            Índice inicial.

        window_end:
            Índice final.

        samples:
            Amostras pertencentes à janela.

        label:
            0 = normal
            1 = PSC
        """

        if label not in (0, 1):
            raise ValueError("ML2 label deve ser 0 ou 1.")

        return cls(
            window_id=window_id,
            window_start=window_start,
            window_end=window_end,
            label=label,
            samples=list(samples),
        )

    def to_dict(self) -> dict:
        """
        Converte a janela para uma estrutura serializável.
        """

        return {
            "window_id": self.window_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "label": self.label,
            "samples": [
                {
                    "timestamp": sample.timestamp.isoformat(),
                    "irradiance": sample.irradiance,
                    "temperature": sample.temperature,
                    "v_out": sample.v_out,
                    "p_out": sample.p_out,
                }
                for sample in self.samples
            ],
        }
