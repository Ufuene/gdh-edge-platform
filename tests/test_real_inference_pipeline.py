"""
test_real_inference_pipeline.py

Teste de integração do pipeline real de inferência do GDH Edge.

Fluxo validado:

    Sample
        ↓
    SampleBuffer
        ↓
    ML1Service + modelo real
        ↓
    DeviationDetector
        ↓
    WindowManager
        ↓
    8 Samples
        ↓
    ML2Service + modelo real
        ↓
    InferenceResult

Este arquivo NÃO altera os testes unitários existentes.
"""

from datetime import datetime
from pathlib import Path

import joblib

from app.data_manager.sample import Sample
from app.data_manager.sample_buffer import SampleBuffer
from app.data_manager.window_manager import WindowManager
from app.inference.deviation_detector import DeviationDetector
from app.inference.inference_orchestrator import InferenceOrchestrator
from app.inference.inference_result import InferenceResult
from app.inference.ml1_service import ML1Service
from app.inference.ml2_service import ML2Service

PROJECT_ROOT = Path("/home/ufuene/gdh-platform")

ML1_MODEL_PATH = PROJECT_ROOT / "models/ml1/RandomForest_ML1_CPU.pkl"
ML2_MODEL_PATH = PROJECT_ROOT / "models/ml2/RandomForest_ML2_cpu.pkl"

POWER_THRESHOLD = 5.0


def create_sample(
    irradiance: float,
    temperature: float,
    v_pv: float,
    i_pv: float,
    v_out: float,
    i_out: float,
    i_bat: float,
    i_load: float,
    p_out: float,
) -> Sample:
    """
    Cria uma Sample para o teste do pipeline real.
    """

    return Sample(
        timestamp=datetime.now(),
        irradiance=irradiance,
        temperature=temperature,
        v_pv=v_pv,
        i_pv=i_pv,
        v_out=v_out,
        i_out=i_out,
        i_bat=i_bat,
        i_load=i_load,
        p_out=p_out,
    )


def load_real_services():
    """
    Carrega os dois modelos reais e cria os serviços ML1 e ML2.
    """

    assert ML1_MODEL_PATH.exists(), f"ML1 model not found: {ML1_MODEL_PATH}"

    assert ML2_MODEL_PATH.exists(), f"ML2 model not found: {ML2_MODEL_PATH}"

    ml1_model = joblib.load(ML1_MODEL_PATH)
    ml2_model = joblib.load(ML2_MODEL_PATH)

    assert ml1_model.n_features_in_ == 3

    assert ml2_model.n_features_in_ == 72
    assert list(ml2_model.classes_) == [0, 1]

    ml1_service = ML1Service(ml1_model)
    ml2_service = ML2Service(ml2_model)

    return ml1_service, ml2_service


def create_pipeline():
    """
    Cria o pipeline completo utilizando os modelos reais.
    """

    ml1_service, ml2_service = load_real_services()

    sample_buffer = SampleBuffer()

    window_manager = WindowManager(
        sample_buffer,
        previous_samples=3,
        future_samples=4,
    )

    deviation_detector = DeviationDetector(
        power_threshold=POWER_THRESHOLD,
    )

    orchestrator = InferenceOrchestrator(
        sample_buffer=sample_buffer,
        ml1_service=ml1_service,
        ml2_service=ml2_service,
        deviation_detector=deviation_detector,
        window_manager=window_manager,
    )

    return (
        orchestrator,
        sample_buffer,
        window_manager,
    )


def test_real_inference_pipeline():
    """
    Executa o pipeline completo usando os dois modelos reais.

    O teste:

    1. cria oito amostras;
    2. processa cada uma pelo ML1 real;
    3. cria intencionalmente um desvio na quarta amostra;
    4. aguarda as quatro amostras futuras;
    5. permite ao WindowManager construir a janela;
    6. executa o ML2 real;
    7. verifica o InferenceResult final.
    """

    (
        orchestrator,
        sample_buffer,
        window_manager,
    ) = create_pipeline()

    results = []

    # ==========================================================
    # Perfil base utilizado para as amostras
    # ==========================================================

    base = {
        "irradiance": 62.2,
        "temperature": 10.86175,
        "v_pv": 23.88,
        "i_pv": 0.2615,
        "v_out": 12.0347,
        "i_out": 1.0,
        "i_bat": -13.5,
        "i_load": 14.5,
    }

    # ==========================================================
    # Gerar as 8 amostras
    #
    # A amostra 3 (quarta posição) será utilizada como
    # ponto de desvio.
    # ==========================================================

    for index in range(8):
        p_out = 15.0

        # Introduz uma redução intencional de potência
        # na quarta amostra.
        if index == 3:
            p_out = 0.0

        sample = create_sample(
            irradiance=base["irradiance"],
            temperature=base["temperature"],
            v_pv=base["v_pv"],
            i_pv=base["i_pv"],
            v_out=base["v_out"],
            i_out=base["i_out"],
            i_bat=base["i_bat"],
            i_load=base["i_load"],
            p_out=p_out,
        )

        result = orchestrator.process_sample(
            sample=sample,
            sample_index=index,
        )

        results.append(result)

    # ==========================================================
    # Verificações do buffer
    # ==========================================================

    assert sample_buffer.size() == 8

    # ==========================================================
    # Deve existir uma janela completa
    # ==========================================================

    assert window_manager.has_complete_window()

    window = window_manager.get_current_window()

    assert window is not None
    assert len(window) == 8

    # ==========================================================
    # O evento deve ter sido identificado na quarta amostra
    # ==========================================================

    assert results[3].deviation_detected is True

    # ==========================================================
    # A primeira vez que a janela fica completa é na amostra 7
    # ==========================================================

    final_result = results[7]

    assert isinstance(final_result, InferenceResult)

    assert final_result.diagnosis in (0, 1)

    # ==========================================================
    # O ML1 deve ter produzido uma potência de referência
    # válida em todas as amostras.
    # ==========================================================

    for result in results:
        assert isinstance(result.predicted_power, float)
        assert result.predicted_power >= 0.0

    # ==========================================================
    # A janela deve conter a amostra de desvio na posição 4.
    # ==========================================================

    assert len(window) == 8

    # O índice 3 corresponde à quarta amostra.
    assert window[3].p_out == 0.0
