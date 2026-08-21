from datetime import datetime, timedelta

import pytest

from app.data_manager.sample import Sample
from app.datasets.ml2_normal_window_builder import (
    ML2NormalWindowBuilder,
)

# ============================================================
# TEST FIXTURE
# ============================================================


def create_sample(index: int) -> Sample:
    """
    Cria uma amostra com a cadência temporal operacional do sistema.

    O GDH Edge trabalha com uma amostra a cada 5 segundos.

    Portanto:

        index = 1 -> t = 5 s
        index = 2 -> t = 10 s
        index = 3 -> t = 15 s
        ...

    A continuidade temporal é parte do contrato do ML2.
    """

    return Sample(
        timestamp=datetime(2026, 1, 1) + timedelta(seconds=index * 5),
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


# ============================================================
# INITIAL STATE
# ============================================================


def test_builder_starts_empty():
    builder = ML2NormalWindowBuilder()

    assert builder.pending_samples() == 0
    assert builder.ml1_sequence_count() == 0
    assert builder.window_count() == 0
    assert builder.has_complete_window() is False


# ============================================================
# CONTINUOUS TEMPORAL WINDOW
# ============================================================


def test_eight_continuous_samples_form_one_normal_window():
    """
    Oito amostras temporalmente contínuas devem formar
    uma janela ML2 completa.

    Cadência:

        5 s

    Portanto:

        8 samples × 5 s

    formam uma única janela temporal.
    """

    builder = ML2NormalWindowBuilder()

    samples = [create_sample(i) for i in range(1, 9)]

    for sample in samples:
        builder.add_sample(sample)

    assert builder.has_complete_window() is True
    assert builder.pending_samples() == 8
    assert builder.ml1_sequence_count() == 8

    window = builder.build_window()

    assert len(window) == 8
    assert window == samples

    assert builder.pending_samples() == 0
    assert builder.window_count() == 1


# ============================================================
# NON-OVERLAPPING WINDOWS
# ============================================================


def test_windows_are_non_overlapping():
    """
    Dezesseis amostras contínuas devem formar duas janelas
    independentes de oito amostras cada.
    """

    builder = ML2NormalWindowBuilder()

    samples = [create_sample(i) for i in range(1, 17)]

    for sample in samples:
        builder.add_sample(sample)

    first_window = builder.build_window()
    second_window = builder.build_window()

    assert first_window == samples[0:8]
    assert second_window == samples[8:16]

    assert set(id(sample) for sample in first_window).isdisjoint(
        id(sample) for sample in second_window
    )

    assert builder.pending_samples() == 0
    assert builder.window_count() == 2


# ============================================================
# TEMPORAL GAP
# ============================================================


def test_temporal_gap_breaks_normal_window():
    """
    Um gap temporal não pode ser atravessado para completar
    uma janela ML2 NORMAL.

    Sequência:

        1 2 3 4 5 6
                    GAP
                              17 18

    As oito amostras podem continuar sendo elegíveis para ML1,
    mas não podem formar uma única janela temporal ML2.

    Após o gap, uma nova sequência temporal é iniciada.
    """

    builder = ML2NormalWindowBuilder()

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

    samples = [create_sample(i) for i in telemetry_sequences]

    for sample in samples:
        builder.add_sample(sample)

    assert builder.ml1_sequence_count() == 8

    # O gap impede a formação de uma janela de oito amostras.
    assert builder.has_complete_window() is False

    # Somente as amostras posteriores ao gap permanecem
    # na nova sequência temporal.
    assert builder.pending_samples() == 2

    assert builder.window_count() == 0


# ============================================================
# SECOND WINDOW AFTER CONTINUOUS SEQUENCE
# ============================================================


def test_next_eight_continuous_samples_form_second_window():
    """
    Depois de um gap, uma nova sequência contínua pode formar
    uma nova janela independente.

    As seis primeiras amostras pertencem à sequência anterior.
    As oito seguintes formam a primeira janela válida após o gap.
    """

    builder = ML2NormalWindowBuilder()

    first_sequence = [
        1,
        2,
        3,
        4,
        5,
        6,
    ]

    second_sequence = [
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    ]

    samples = [create_sample(i) for i in first_sequence + second_sequence]

    for sample in samples:
        builder.add_sample(sample)

    assert builder.has_complete_window() is True
    assert builder.pending_samples() == 8

    window = builder.build_window()

    expected_samples = [create_sample(i) for i in second_sequence]

    assert window == expected_samples
    assert len(window) == 8

    assert builder.pending_samples() == 0
    assert builder.window_count() == 1


# ============================================================
# INCOMPLETE WINDOW
# ============================================================


def test_incomplete_window_remains_pending():
    """
    Sete amostras contínuas não são suficientes para formar
    uma janela ML2.
    """

    builder = ML2NormalWindowBuilder()

    samples = [create_sample(i) for i in range(1, 8)]

    for sample in samples:
        builder.add_sample(sample)

    assert builder.has_complete_window() is False
    assert builder.pending_samples() == 7
    assert builder.window_count() == 0


# ============================================================
# ADD AND BUILD
# ============================================================


def test_add_sample_and_build_if_ready():
    """
    A oitava amostra contínua deve completar a janela.
    """

    builder = ML2NormalWindowBuilder()

    for index in range(1, 8):
        result = builder.add_sample_and_build_if_ready(create_sample(index))

        assert result is None

    result = builder.add_sample_and_build_if_ready(create_sample(8))

    assert result is not None
    assert len(result) == 8
    assert builder.pending_samples() == 0
    assert builder.window_count() == 1


# ============================================================
# GAP FOLLOWED BY NEW WINDOW
# ============================================================


def test_gap_discards_old_partial_window_and_starts_new_sequence():
    """
    Um gap não deve permitir que uma janela parcial anterior
    seja misturada com uma nova sequência temporal.

    Exemplo:

        1 2 3 4 5
                GAP
                    10 11 12 13 14 15 16 17

    Resultado:

        [10..17]

    é a primeira janela válida.
    """

    builder = ML2NormalWindowBuilder()

    first_sequence = [1, 2, 3, 4, 5]
    second_sequence = [10, 11, 12, 13, 14, 15, 16, 17]

    for index in first_sequence + second_sequence:
        builder.add_sample(create_sample(index))

    assert builder.has_complete_window() is True
    assert builder.pending_samples() == 8

    window = builder.build_window()

    expected = [create_sample(index) for index in second_sequence]

    assert window == expected
    assert len(window) == 8
    assert builder.window_count() == 1
    assert builder.pending_samples() == 0


# ============================================================
# BUILD WITHOUT COMPLETE WINDOW
# ============================================================


def test_build_without_complete_window_raises_error():
    builder = ML2NormalWindowBuilder()

    for index in range(1, 8):
        builder.add_sample(create_sample(index))

    with pytest.raises(RuntimeError):
        builder.build_window()


# ============================================================
# RESET
# ============================================================


def test_reset_clears_builder_state():
    builder = ML2NormalWindowBuilder()

    for index in range(1, 9):
        builder.add_sample(create_sample(index))

    builder.build_window()

    builder.reset()

    assert builder.pending_samples() == 0
    assert builder.ml1_sequence_count() == 0
    assert builder.window_count() == 0
    assert builder.has_complete_window() is False


# ============================================================
# TEMPORAL CADENCE
# ============================================================


def test_continuous_samples_have_five_second_cadence():
    """
    Garante explicitamente o contrato temporal utilizado
    pelos testes do ML2.
    """

    samples = [create_sample(i) for i in range(1, 9)]

    for previous, current in zip(
        samples,
        samples[1:],
    ):
        delta = (current.timestamp - previous.timestamp).total_seconds()

        assert delta == 5.0
