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

Representação da janela:

    O modelo ML2 utiliza 72 features.

    A janela temporal possui 8 amostras.

    Cada amostra contém 9 grandezas:

        1. Irradiancia
        2. Temperatura
        3. Vout
        4. Iout
        5. Ipv
        6. Vpv
        7. Iload
        8. Pout
        9. Ibat

    Portanto:

        8 × 9 = 72 features

A serialização da janela preserva explicitamente essa
representação para que os dados armazenados no Cloud/S3
possam posteriormente ser utilizados na construção de
novos datasets e modelos ML2.
"""

from dataclasses import dataclass
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

        A ordem das grandezas em cada amostra é preservada
        explicitamente de acordo com a representação canônica
        utilizada pelo dataset ML2:

            1. Irradiancia
            2. Temperatura
            3. Vout
            4. Iout
            5. Ipv
            6. Vpv
            7. Iload
            8. Pout
            9. Ibat

        O modelo ML2 utiliza:

            8 amostras × 9 grandezas = 72 features
        """

        return {
            "window_id": self.window_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "label": self.label,
            "samples": [
                {
                    "timestamp": sample.timestamp.isoformat(),
                    # ==================================================
                    # ORDEM CANÔNICA DO DATASET ML2
                    # ==================================================
                    "Irradiancia": sample.irradiance,
                    "Temperatura": sample.temperature,
                    "Vout": sample.v_out,
                    "Iout": sample.i_out,
                    "Ipv": sample.i_pv,
                    "Vpv": sample.v_pv,
                    "Iload": sample.i_load,
                    "Pout": sample.p_out,
                    "Ibat": sample.i_bat,
                }
                for sample in self.samples
            ],
        }
