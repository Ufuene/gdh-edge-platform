"""
ml1_dataset.py

Define o registro de dados destinado ao dataset de treinamento
do ML1.

O ML1 utiliza somente amostras consideradas saudáveis pelo
detector de desvios.

Regra:

    deviation_detected == False
        ->
    amostra elegível para ML1

A classificação posterior do ML2 NÃO é utilizada para decidir
a elegibilidade do ML1.
"""

from dataclasses import dataclass
from datetime import datetime

from app.data_manager.sample import Sample


@dataclass(slots=True)
class ML1DatasetRecord:
    """
    Registro individual destinado ao dataset do ML1.

    Features do ML1:

        irradiance
        temperature
        v_out

    Target:

        p_out
    """

    timestamp: datetime

    irradiance: float
    temperature: float
    v_out: float

    p_out: float

    @classmethod
    def from_sample(cls, sample: Sample) -> "ML1DatasetRecord":
        """
        Constrói um registro ML1 a partir de uma amostra.

        A validação de elegibilidade é responsabilidade do
        DatasetManager.
        """

        return cls(
            timestamp=sample.timestamp,
            irradiance=sample.irradiance,
            temperature=sample.temperature,
            v_out=sample.v_out,
            p_out=sample.p_out,
        )

    def to_dict(self) -> dict:
        """
        Converte o registro para um dicionário serializável.
        """

        return {
            "timestamp": self.timestamp.isoformat(),
            "irradiance": self.irradiance,
            "temperature": self.temperature,
            "v_out": self.v_out,
            "p_out": self.p_out,
        }
