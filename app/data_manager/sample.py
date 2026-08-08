"""
sample.py

Define a estrutura de uma amostra adquirida pelo ESP32.

Toda informação que percorre a Plataforma Edge AI do Gêmeo Digital
é representada por um objeto Sample.

Esta classe representa uma única observação do sistema fotovoltaico,
contendo tanto as variáveis medidas pelos sensores quanto os resultados
gerados durante o processamento (ML1, detecção de desvios e ML2).

Importante:
- Esta classe representa apenas uma amostra individual.
- A construção das janelas temporais utilizadas pelo ML2 é responsabilidade
  exclusiva do WindowManager.
- Esta classe não executa cálculos nem contém regras de negócio; ela apenas
  encapsula os dados que circulam pela plataforma.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(slots=True)
class Sample:
    """
    Representa uma única aquisição do sistema fotovoltaico.

    Atributos:
        timestamp: Instante da aquisição.
        irradiance: Irradiância solar (W/m²).
        temperature: Temperatura do módulo (°C).
        v_pv: Tensão do módulo fotovoltaico (V).
        i_pv: Corrente do módulo fotovoltaico (A).
        v_out: Tensão de saída do sistema (V).
        i_out: Corrente de saída do sistema (A).
        i_bat: Corrente da bateria (A).
        i_load: Corrente da carga (A).
        p_out: Potência de saída do sistema (W).

        predicted_power: Potência estimada pelo ML1.
        deviation: Diferença entre P_out e P_ref.
        deviation_detected: Indica se |ΔP| > ε.
        diagnosis: Resultado produzido pelo ML2.
    """

    # ==========================================================
    # Dados provenientes do ESP32
    # ==========================================================

    timestamp: datetime

    irradiance: float
    temperature: float

    v_pv: float
    i_pv: float

    v_out: float
    i_out: float

    i_bat: float
    i_load: float

    p_out: float

    # ==========================================================
    # Resultados produzidos pelo ML1
    # ==========================================================

    predicted_power: Optional[float] = None

    # ==========================================================
    # Resultados produzidos pelo Detector de Desvios
    # ==========================================================

    deviation: Optional[float] = None

    deviation_detected: bool = False

    # ==========================================================
    # Resultado produzido pelo ML2
    # ==========================================================

    diagnosis: Optional[int] = None
