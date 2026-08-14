from app.inference.inference_result import InferenceResult


def test_inference_result_creation():
    """
    Verifica a criação de um resultado de inferência.
    """

    result = InferenceResult(
        predicted_power=118.5,
        deviation=3.2,
        deviation_detected=False,
    )

    assert result.predicted_power == 118.5
    assert result.deviation == 3.2
    assert result.deviation_detected is False
    assert result.diagnosis is None


def test_inference_result_with_diagnosis():
    """
    Verifica a criação de um resultado com diagnóstico ML2.
    """

    result = InferenceResult(
        predicted_power=120.0,
        deviation=20.0,
        deviation_detected=True,
        diagnosis=1,
    )

    assert result.predicted_power == 120.0
    assert result.deviation == 20.0
    assert result.deviation_detected is True
    assert result.diagnosis == 1


def test_inference_result_normal_diagnosis():
    """
    Verifica um diagnóstico NORMAL produzido pelo ML2.
    """

    result = InferenceResult(
        predicted_power=120.0,
        deviation=25.0,
        deviation_detected=True,
        diagnosis=0,
    )

    assert result.diagnosis == 0


def test_inference_result_defaults_diagnosis_to_none():
    """
    Verifica que o diagnóstico é opcional e inicia como None.
    """

    result = InferenceResult(
        predicted_power=100.0,
        deviation=2.0,
        deviation_detected=False,
    )

    assert result.diagnosis is None
