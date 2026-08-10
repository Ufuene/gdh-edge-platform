from app.inference.deviation_detector import DeviationDetector


def test_deviation_detector_creation():

    detector = DeviationDetector()

    assert detector._power_threshold == 5.0


def test_deviation_detector_custom_threshold():

    detector = DeviationDetector(power_threshold=10.0)

    assert detector._power_threshold == 10.0


def test_compute_error_when_measured_power_is_greater():

    detector = DeviationDetector()

    error = detector.compute_error(
        measured_power=120.0,
        predicted_power=118.0,
    )

    assert error == 2.0


def test_compute_error_when_predicted_power_is_greater():

    detector = DeviationDetector()

    error = detector.compute_error(
        measured_power=100.0,
        predicted_power=120.0,
    )

    assert error == 20.0


def test_compute_error_when_powers_are_equal():

    detector = DeviationDetector()

    error = detector.compute_error(
        measured_power=120.0,
        predicted_power=120.0,
    )

    assert error == 0.0


def test_has_deviation_when_error_is_below_threshold():

    detector = DeviationDetector(
        power_threshold=5.0,
    )

    result = detector.has_deviation(
        measured_power=120.0,
        predicted_power=118.0,
    )

    assert result is False


def test_has_deviation_when_error_equals_threshold():

    detector = DeviationDetector(
        power_threshold=5.0,
    )

    result = detector.has_deviation(
        measured_power=120.0,
        predicted_power=115.0,
    )

    assert result is False


def test_has_deviation_when_error_exceeds_threshold():

    detector = DeviationDetector(
        power_threshold=5.0,
    )

    result = detector.has_deviation(
        measured_power=120.0,
        predicted_power=114.0,
    )

    assert result is True


def test_has_deviation_when_predicted_power_is_greater_and_error_is_below_threshold():

    detector = DeviationDetector(
        power_threshold=5.0,
    )

    result = detector.has_deviation(
        measured_power=118.0,
        predicted_power=120.0,
    )

    assert result is False


def test_has_deviation_when_predicted_power_is_greater_and_error_exceeds_threshold():

    detector = DeviationDetector(
        power_threshold=5.0,
    )

    result = detector.has_deviation(
        measured_power=114.0,
        predicted_power=120.0,
    )

    assert result is True
