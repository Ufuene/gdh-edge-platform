"""
inference_result.py

Define a estrutura utilizada para transportar o resultado
do processamento de inferência do Gêmeo Digital Híbrido (GDH).

O InferenceResult agrega:

- resultado do ML1;
- resultado do detector de desvios;
- execução/diagnóstico do ML2;
- metadados da janela utilizada pelo ML2;
- amostras da janela utilizada pelo ML2;
- elegibilidade para formação dos datasets.

Regras de elegibilidade:

CASO A — operação saudável
--------------------------------
deviation_detected = False

    ML1 dataset  -> elegível
    ML2 dataset  -> elegível como NORMAL (label 0)

CASO B — desvio diagnosticado como PSC
--------------------------------
deviation_detected = True
ml2_executed = True
diagnosis = 1

    ML1 dataset  -> não elegível
    ML2 dataset  -> elegível como PSC (label 1)

CASO C — desvio diagnosticado como NORMAL
--------------------------------
deviation_detected = True
ml2_executed = True
diagnosis = 0

    ML1 dataset  -> não elegível
    ML2 dataset  -> não elegível

diagnosis=None significa que o ML2 não foi executado.
"""

from dataclasses import dataclass
from typing import Optional

from app.data_manager.sample import Sample


@dataclass(slots=True)
class InferenceResult:
    """
    Resultado agregado do processamento de uma amostra.
    """

    # ==========================================================
    # ML1 / Detector de desvios
    # ==========================================================

    predicted_power: float
    deviation: float
    deviation_detected: bool

    # ==========================================================
    # ML2
    # ==========================================================

    diagnosis: Optional[int] = None

    ml2_executed: bool = False

    # ==========================================================
    # Metadados da janela utilizada pelo ML2
    # ==========================================================

    window_available: bool = False

    window_start: Optional[int] = None
    window_end: Optional[int] = None

    # ==========================================================
    # Amostras da janela utilizada pelo ML2
    #
    # Quando ML2 é executado, contém uma cópia da janela
    # efetivamente enviada ao modelo.
    #
    # Quando ML2 não é executado:
    #
    #     window_samples = None
    # ==========================================================

    window_samples: Optional[list[Sample]] = None

    # ==========================================================
    # Elegibilidade para datasets
    # ==========================================================

    ml1_dataset_eligible: bool = False
    ml2_dataset_eligible: bool = False

    # ==========================================================
    # Label do dataset ML2
    #
    # 0 = Normal
    # 1 = PSC
    # None = não existe registro ML2 elegível
    # ==========================================================

    ml2_label: Optional[int] = None
