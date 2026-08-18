from datetime import datetime, timedelta

from app.data_manager.sample import Sample
from app.datasets.dataset_manager import DatasetManager
from app.inference.inference_result import InferenceResult


def create_sample(sequence: int) -> Sample:
    return Sample(
        timestamp=datetime(2026, 8, 17) + timedelta(seconds=sequence),
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


def test_eight_healthy_samples_create_one_normal_ml2_window():
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


def test_non_contiguous_telemetry_indices_use_ml1_sequence():
    manager = DatasetManager()

    result = create_healthy_result()

    telemetry_sequences = [1, 2, 3, 4, 5, 6, 17, 18]

    for sequence in telemetry_sequences:
        ml1_record, ml2_window = manager.process_sample(
            sample=create_sample(sequence),
            result=result,
        )

        assert ml1_record is not None

    assert len(manager.get_ml1_records()) == 8
    assert len(manager.get_ml2_records()) == 1

    window = manager.get_ml2_records()[0]

    assert window.label == 0
    assert window.window_id == "normal-000001"
    assert window.window_start == 1
    assert window.window_end == 8

    actual_sequences = [
        int((sample.timestamp - datetime(2026, 8, 17)).total_seconds())
        for sample in window.samples
    ]

    assert actual_sequences == telemetry_sequences


def test_psc_samples_do_not_enter_normal_ml2_window():
    manager = DatasetManager()

    healthy_result = create_healthy_result()
    psc_result = create_psc_result()

    sequence = 1

    for _ in range(4):
        manager.process_sample(
            sample=create_sample(sequence),
            result=healthy_result,
        )
        sequence += 1

    manager.process_sample(
        sample=create_sample(sequence),
        result=psc_result,
    )
    sequence += 1

    for _ in range(4):
        manager.process_sample(
            sample=create_sample(sequence),
            result=healthy_result,
        )
        sequence += 1

    assert len(manager.get_ml1_records()) == 8
    assert len(manager.get_ml2_records()) == 1

    window = manager.get_ml2_records()[0]

    assert window.label == 0
    assert window.window_start == 1
    assert window.window_end == 8
    assert len(window.samples) == 8

    actual_sequences = [
        int((sample.timestamp - datetime(2026, 8, 17)).total_seconds())
        for sample in window.samples
    ]

    assert actual_sequences == [1, 2, 3, 4, 6, 7, 8, 9]


def test_two_non_overlapping_normal_windows_are_created():
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
