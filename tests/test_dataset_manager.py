from datetime import datetime

from app.data_manager.sample import Sample
from app.datasets.dataset_manager import DatasetManager
from app.inference.inference_result import InferenceResult


def create_sample():
    return Sample(
        timestamp=datetime.now(),
        irradiance=800.0,
        temperature=35.0,
        v_pv=30.0,
        i_pv=5.0,
        v_out=14.7,
        i_out=4.0,
        i_bat=2.0,
        i_load=2.0,
        p_out=58.8,
    )


# ==============================================================
# CASO A
# ==============================================================
#
# Operação saudável:
#
#   deviation <= threshold
#   deviation_detected = False
#   ML2 não é executado
#
# A amostra:
#
#   -> pode alimentar ML1
#   -> pode compor a sequência normal utilizada para ML2
#
# IMPORTANTE:
# A formação da janela normal de 8 amostras será testada
# posteriormente. Aqui testamos somente a elegibilidade da
# amostra individual.
# ==============================================================


def test_case_a_healthy_data_is_eligible_for_ml1_and_normal_ml2_sequence():
    result = InferenceResult(
        predicted_power=60.0,
        deviation=0.0,
        deviation_detected=False,
        diagnosis=None,
        ml2_executed=False,
    )

    manager = DatasetManager()
    sample = create_sample()

    # ----------------------------------------------------------
    # ML1
    # ----------------------------------------------------------

    ml1 = manager.create_ml1_record(
        sample=sample,
        result=result,
    )

    assert ml1 is not None

    assert ml1.irradiance == 800.0
    assert ml1.temperature == 35.0
    assert ml1.v_out == 14.7
    assert ml1.p_out == 58.8

    assert manager.is_ml1_eligible(result) is True

    # ----------------------------------------------------------
    # ML2
    #
    # A amostra saudável pode participar da formação da
    # sequência normal do ML2.
    #
    # A janela de 8 amostras será tratada posteriormente.
    # ----------------------------------------------------------

    assert manager.is_ml2_eligible(result) is True

    assert manager.determine_ml2_label(result) == 0


# ==============================================================
# CASO B
# ==============================================================
#
# Desvio detectado e ML2 diagnostica PSC:
#
#   deviation > threshold
#   deviation_detected = True
#   ml2_executed = True
#   diagnosis = 1
#
# Resultado:
#
#   -> não entra no ML1
#   -> entra no ML2 como PSC
# ==============================================================


def test_case_b_psc_is_eligible_only_for_ml2():
    result = InferenceResult(
        predicted_power=60.0,
        deviation=15.0,
        deviation_detected=True,
        diagnosis=1,
        ml2_executed=True,
        window_available=True,
        window_start=0,
        window_end=7,
    )

    manager = DatasetManager()

    assert manager.is_ml1_eligible(result) is False

    assert manager.is_ml2_eligible(result) is True

    assert manager.determine_ml2_label(result) == 1


# ==============================================================
# CASO C
# ==============================================================
#
# Desvio detectado, ML2 executado, mas diagnóstico = NORMAL.
#
#   deviation > threshold
#   deviation_detected = True
#   ml2_executed = True
#   diagnosis = 0
#
# Esse caso é descartado.
#
# Não deve contaminar:
#
#   -> ML1
#   -> ML2
#
# porque o fato de o ML2 dizer "Normal" não transforma
# automaticamente a amostra em uma amostra saudável.
# ==============================================================


def test_case_c_deviation_with_ml2_diagnosis_zero_is_discarded():
    result = InferenceResult(
        predicted_power=60.0,
        deviation=15.0,
        deviation_detected=True,
        diagnosis=0,
        ml2_executed=True,
        window_available=True,
        window_start=0,
        window_end=7,
    )

    manager = DatasetManager()

    assert manager.is_ml1_eligible(result) is False

    assert manager.is_ml2_eligible(result) is False

    assert manager.determine_ml2_label(result) is None


# ==============================================================
# CASO D
# ==============================================================
#
# ML2 não foi executado.
#
# diagnosis = None
# ml2_executed = False
#
# Isso significa que o ML2 está fora do fluxo de inferência
# neste ciclo.
#
# Se o desvio estiver abaixo do limiar, a amostra pode alimentar
# ML1 e pode ser candidata à sequência normal do ML2.
# ==============================================================


def test_ml2_none_means_not_executed():
    result = InferenceResult(
        predicted_power=60.0,
        deviation=2.0,
        deviation_detected=False,
        diagnosis=None,
        ml2_executed=False,
    )

    manager = DatasetManager()

    assert result.diagnosis is None
    assert result.ml2_executed is False

    assert manager.is_ml1_eligible(result) is True

    assert manager.is_ml2_eligible(result) is True

    assert manager.determine_ml2_label(result) == 0
