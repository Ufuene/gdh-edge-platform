from datetime import datetime, timedelta

from app.data_manager.sample import Sample
from app.datasets.ml2_normal_window_builder import (
    ML2NormalWindowBuilder,
)


def create_sample(index: int) -> Sample:
    return Sample(
        timestamp=datetime(2026, 1, 1) + timedelta(seconds=index),
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


def test_builder_starts_empty():
    builder = ML2NormalWindowBuilder()

    assert builder.pending_samples() == 0
    assert builder.ml1_sequence_count() == 0
    assert builder.window_count() == 0
    assert builder.has_complete_window() is False


def test_eight_ml1_eligible_samples_form_one_window():
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


def test_windows_are_non_overlapping():
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


def test_telemetry_sequence_gaps_do_not_break_ml1_sequence():
    builder = ML2NormalWindowBuilder()

    # Simula amostras elegíveis para ML1 cuja sequência original
    # de telemetria possui uma interrupção.
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

    window = builder.build_window()

    assert len(window) == 8

    # O construtor trabalha pela ordem de chegada das amostras
    # elegíveis, não pela distância entre seus índices originais.
    assert window == samples


def test_next_eight_eligible_samples_form_second_window():
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
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
    ]

    samples = [create_sample(i) for i in telemetry_sequences]

    for sample in samples:
        builder.add_sample(sample)

    first_window = builder.build_window()
    second_window = builder.build_window()

    assert first_window == samples[0:8]
    assert second_window == samples[8:16]


def test_incomplete_window_remains_pending():
    builder = ML2NormalWindowBuilder()

    samples = [create_sample(i) for i in range(1, 8)]

    for sample in samples:
        builder.add_sample(sample)

    assert builder.has_complete_window() is False
    assert builder.pending_samples() == 7
    assert builder.window_count() == 0


def test_add_sample_and_build_if_ready():
    builder = ML2NormalWindowBuilder()

    for index in range(1, 8):
        result = builder.add_sample_and_build_if_ready(create_sample(index))

        assert result is None

    result = builder.add_sample_and_build_if_ready(create_sample(8))

    assert result is not None
    assert len(result) == 8
    assert builder.pending_samples() == 0
    assert builder.window_count() == 1


def test_build_without_complete_window_raises_error():
    builder = ML2NormalWindowBuilder()

    for index in range(1, 8):
        builder.add_sample(create_sample(index))

    try:
        builder.build_window()
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


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
