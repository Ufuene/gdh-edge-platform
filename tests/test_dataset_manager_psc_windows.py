from datetime import datetime, timedelta

from app.data_manager.sample import Sample
from app.datasets.dataset_manager import DatasetManager
from app.inference.inference_result import InferenceResult


def create_sample(index: int) -> Sample:
    return Sample(
        timestamp=datetime(2026, 1, 1) + timedelta(seconds=index),
        irradiance=800.0 + index,
        temperature=35.0,
        v_pv=30.0,
        i_pv=5.0,
        v_out=14.7,
        i_out=4.0,
        i_bat=2.0,
        i_load=2.0,
        p_out=58.8,
    )


def create_psc_result(samples):
    return InferenceResult(
        predicted_power=60.0,
        deviation=15.0,
        deviation_detected=True,
        diagnosis=1,
        ml2_executed=True,
        window_available=True,
        window_start=10,
        window_end=17,
        window_samples=samples,
    )


def test_psc_window_is_registered_with_label_one():
    manager = DatasetManager()

    samples = [create_sample(index) for index in range(8)]

    result = create_psc_result(samples)

    ml1_record, ml2_window = manager.process_sample(
        sample=samples[-1],
        result=result,
    )

    assert ml1_record is None

    assert ml2_window is not None
    assert ml2_window.label == 1
    assert ml2_window.window_id == "psc-000001"

    assert ml2_window.window_start == 10
    assert ml2_window.window_end == 17

    assert len(ml2_window.samples) == 8
    assert ml2_window.samples == samples


def test_psc_samples_do_not_enter_ml1_dataset():
    manager = DatasetManager()

    samples = [create_sample(index) for index in range(8)]

    result = create_psc_result(samples)

    manager.process_sample(
        sample=samples[-1],
        result=result,
    )

    assert manager.get_ml1_records() == []


def test_psc_window_is_registered_in_ml2_dataset():
    manager = DatasetManager()

    samples = [create_sample(index) for index in range(8)]

    result = create_psc_result(samples)

    manager.process_sample(
        sample=samples[-1],
        result=result,
    )

    records = manager.get_ml2_records()

    assert len(records) == 1
    assert records[0].label == 1
    assert records[0].samples == samples


def test_diagnosis_zero_does_not_create_ml2_window():
    manager = DatasetManager()

    samples = [create_sample(index) for index in range(8)]

    result = InferenceResult(
        predicted_power=60.0,
        deviation=15.0,
        deviation_detected=True,
        diagnosis=0,
        ml2_executed=True,
        window_available=True,
        window_start=10,
        window_end=17,
        window_samples=samples,
    )

    ml1_record, ml2_window = manager.process_sample(
        sample=samples[-1],
        result=result,
    )

    assert ml1_record is None
    assert ml2_window is None
    assert manager.get_ml1_records() == []
    assert manager.get_ml2_records() == []


def test_psc_window_without_samples_is_not_registered():
    manager = DatasetManager()

    result = InferenceResult(
        predicted_power=60.0,
        deviation=15.0,
        deviation_detected=True,
        diagnosis=1,
        ml2_executed=True,
        window_available=True,
        window_start=10,
        window_end=17,
        window_samples=None,
    )

    ml1_record, ml2_window = manager.process_sample(
        sample=create_sample(0),
        result=result,
    )

    assert ml1_record is None
    assert ml2_window is None
    assert manager.get_ml1_records() == []
    assert manager.get_ml2_records() == []
