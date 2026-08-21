from datetime import datetime, timedelta

from app.data_manager.sample import Sample
from app.datasets.dataset_manager import DatasetManager
from app.inference.inference_result import InferenceResult

# ============================================================
# TEST FIXTURES
# ============================================================


def create_sample(sequence: int) -> Sample:
    """
    Cria uma amostra com a cadência temporal operacional
    de 5 segundos.
    """

    return Sample(
        timestamp=datetime(2026, 8, 17) + timedelta(seconds=sequence * 5),
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


def create_healthy_result() -> InferenceResult:
    return InferenceResult(
        predicted_power=60.0,
        deviation=0.0,
        deviation_detected=False,
        diagnosis=None,
        ml2_executed=False,
    )


def create_psc_result() -> InferenceResult:
    return InferenceResult(
        predicted_power=60.0,
        deviation=15.0,
        deviation_detected=True,
        diagnosis=1,
        ml2_executed=True,
    )


# ============================================================
# NORMAL WINDOW
# ============================================================


def test_eight_healthy_samples_create_one_normal_ml2_window():
    """
    Oito amostras saudáveis e temporalmente contínuas devem
    produzir:

        8 registros ML1
        +
        1 janela ML2 NORMAL
    """

    manager = DatasetManager()

    result = create_healthy_result()

    for sequence in range(1, 9):
        ml1_record, ml2_window = manager.process_sample(
            sample=create_sample(sequence),
            result=result,
        )

        assert ml1_record is not None

        if sequence < 8:
            assert ml2_window is None
        else:
            assert ml2_window is not None

    assert len(manager.get_ml1_records()) == 8
    assert len(manager.get_ml2_records()) == 1

    window = manager.get_ml2_records()[0]

    assert window.label == 0
    assert window.window_id == "normal-000001"
    assert window.window_start == 1
    assert window.window_end == 8
    assert len(window.samples) == 8


# ============================================================
# INCOMPLETE WINDOW
# ============================================================


def test_seven_healthy_samples_do_not_create_window():
    manager = DatasetManager()

    result = create_healthy_result()

    for sequence in range(1, 8):
        ml1_record, ml2_window = manager.process_sample(
            sample=create_sample(sequence),
            result=result,
        )

        assert ml1_record is not None
        assert ml2_window is None

    assert len(manager.get_ml1_records()) == 7
    assert len(manager.get_ml2_records()) == 0


# ============================================================
# TEMPORAL GAP
# ============================================================


def test_non_contiguous_telemetry_does_not_create_normal_ml2_window():
    """
    As amostras continuam sendo elegíveis individualmente
    para ML1.

    Porém:

        1 2 3 4 5 6 GAP 17 18

    não pode formar uma janela temporal ML2.

    Portanto:

        ML1 = 8 registros
        ML2 = 0 janelas
    """

    manager = DatasetManager()

    result = create_healthy_result()

    telemetry_sequences = [
        1,
        2,
        3,
        4,
        5,
        6,
        17,
        18,
    ]

    for sequence in telemetry_sequences:
        ml1_record, ml2_window = manager.process_sample(
            sample=create_sample(sequence),
            result=result,
        )

        assert ml1_record is not None
        assert ml2_window is None

    assert len(manager.get_ml1_records()) == 8
    assert len(manager.get_ml2_records()) == 0


# ============================================================
# PSC SAMPLE DOES NOT ENTER NORMAL BUILDER
# ============================================================


def test_psc_sample_does_not_enter_normal_ml2_window():
    """
    Uma amostra diagnosticada como PSC:

        deviation_detected = True
        diagnosis = 1
        ml2_executed = True

    não pode alimentar o construtor de janelas normais.

    Além disso, a ausência dessa amostra cria uma quebra
    temporal na sequência normal.

    Portanto, quatro amostras antes do PSC e quatro depois
    do PSC não devem ser artificialmente combinadas em uma
    janela normal de oito amostras.
    """

    manager = DatasetManager()

    healthy_result = create_healthy_result()
    psc_result = create_psc_result()

    sequence = 1

    # --------------------------------------------------------
    # Quatro amostras saudáveis.
    # --------------------------------------------------------

    for _ in range(4):
        ml1_record, ml2_window = manager.process_sample(
            sample=create_sample(sequence),
            result=healthy_result,
        )

        assert ml1_record is not None
        assert ml2_window is None

        sequence += 1

    # --------------------------------------------------------
    # Uma amostra PSC.
    # --------------------------------------------------------

    ml1_record, ml2_window = manager.process_sample(
        sample=create_sample(sequence),
        result=psc_result,
    )

    assert ml1_record is None
    assert ml2_window is None

    sequence += 1

    # --------------------------------------------------------
    # Quatro novas amostras saudáveis.
    # --------------------------------------------------------

    for _ in range(4):
        ml1_record, ml2_window = manager.process_sample(
            sample=create_sample(sequence),
            result=healthy_result,
        )

        assert ml1_record is not None
        assert ml2_window is None

        sequence += 1

    # --------------------------------------------------------
    # Resultado:
    #
    # 8 registros ML1 saudáveis
    # 0 janelas ML2 normais
    #
    # O PSC não deve ser usado para "preencher" uma janela
    # normal.
    # --------------------------------------------------------

    assert len(manager.get_ml1_records()) == 8
    assert len(manager.get_ml2_records()) == 0


# ============================================================
# TWO NORMAL WINDOWS
# ============================================================


def test_two_non_overlapping_normal_windows_are_created():
    """
    Dezesseis amostras saudáveis e temporalmente contínuas
    devem produzir duas janelas ML2 normais não sobrepostas.
    """

    manager = DatasetManager()

    result = create_healthy_result()

    for sequence in range(1, 17):
        ml1_record, ml2_window = manager.process_sample(
            sample=create_sample(sequence),
            result=result,
        )

        assert ml1_record is not None

        if sequence in (8, 16):
            assert ml2_window is not None
        else:
            assert ml2_window is None

    assert len(manager.get_ml1_records()) == 16
    assert len(manager.get_ml2_records()) == 2

    first_window = manager.get_ml2_records()[0]
    second_window = manager.get_ml2_records()[1]

    assert first_window.window_id == "normal-000001"
    assert first_window.window_start == 1
    assert first_window.window_end == 8

    assert second_window.window_id == "normal-000002"
    assert second_window.window_start == 9
    assert second_window.window_end == 16

    assert first_window.label == 0
    assert second_window.label == 0

    assert len(first_window.samples) == 8
    assert len(second_window.samples) == 8


# ============================================================
# GAP BETWEEN TWO NORMAL SEQUENCES
# ============================================================


def test_gap_between_normal_sequences_resets_ml2_window():
    """
    Uma sequência parcial anterior ao gap não pode ser
    combinada com uma nova sequência posterior ao gap.

    Exemplo:

        1 2 3 4 5 6
                    GAP
                              17 18 19 20 21 22 23 24

    As oito últimas amostras são contínuas e devem formar
    uma nova janela normal.
    """

    manager = DatasetManager()

    result = create_healthy_result()

    sequences = [
        1,
        2,
        3,
        4,
        5,
        6,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    ]

    for sequence in sequences:
        manager.process_sample(
            sample=create_sample(sequence),
            result=result,
        )

    # Todas as amostras continuam elegíveis individualmente
    # para ML1.
    assert len(manager.get_ml1_records()) == 14

    # Somente as oito amostras posteriores ao gap formam
    # uma janela temporal válida.
    assert len(manager.get_ml2_records()) == 1

    window = manager.get_ml2_records()[0]

    assert window.label == 0
    assert window.window_id == "normal-000001"
    assert window.window_start == 1
    assert window.window_end == 8

    assert len(window.samples) == 8

    actual_sequences = [
        int((sample.timestamp - datetime(2026, 8, 17)).total_seconds() / 5)
        for sample in window.samples
    ]

    assert actual_sequences == [
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    ]
