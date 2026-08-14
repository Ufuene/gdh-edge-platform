"""
inference_result.py

Define a estrutura utilizada para transportar o resultado
do processamento de inferência do Gêmeo Digital Híbrido (GDH).

O InferenceResult agrega os resultados produzidos pelas
diferentes etapas do fluxo:

- ML1 → predicted_power
- DeviationDetector → deviation
- DeviationDetector → deviation_detected
- ML2 → diagnosis

O diagnóstico do ML2 é opcional porque pode ainda não estar
disponível quando a janela temporal necessária não estiver
completa.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class InferenceResult:
    """
    Representa o resultado agregado do processamento de uma amostra.
    """

    predicted_power: float
    deviation: float
    deviation_detected: bool
    diagnosis: Optional[int] = None
