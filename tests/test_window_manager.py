from app.data_manager.sample_buffer import SampleBuffer
from app.data_manager.window_manager import (
    WindowManager,
    WindowState,
)


def test_window_manager_creation():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    assert manager._sample_buffer == buffer

    assert manager._previous_samples == 3
    assert manager._future_samples == 4

    assert manager._window_size == 8

    assert manager._hysteresis_samples == 8

    assert manager._state == WindowState.IDLE

    assert manager._deviation_index is None

    assert manager._current_window is None

    assert manager._hysteresis_counter == 0


def test_reset():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    manager._state = WindowState.ACTIVE_EVENT

    manager._deviation_index = 15

    manager._current_window = [1, 2, 3]

    manager._hysteresis_counter = 5

    manager.reset()

    assert manager._state == WindowState.IDLE

    assert manager._deviation_index is None

    assert manager._current_window is None

    assert manager._hysteresis_counter == 0


def test_is_event_active():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    assert not manager.is_event_active()

    manager._state = WindowState.WAITING_WINDOW
    assert manager.is_event_active()

    manager._state = WindowState.ACTIVE_EVENT
    assert manager.is_event_active()

    manager._state = WindowState.HYSTERESIS
    assert manager.is_event_active()

    manager._state = WindowState.IDLE
    assert not manager.is_event_active()


def test_notify_deviation_from_idle():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    manager.notify_deviation(25)

    assert manager._state == WindowState.WAITING_WINDOW

    assert manager._deviation_index == 25


def test_notify_deviation_from_waiting_window():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    manager._state = WindowState.WAITING_WINDOW

    manager.notify_deviation(30)

    assert manager._state == WindowState.WAITING_WINDOW

    assert manager._deviation_index is None


def test_notify_deviation_from_active_event():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    manager._state = WindowState.ACTIVE_EVENT

    manager.notify_deviation(40)

    assert manager._state == WindowState.ACTIVE_EVENT

    assert manager._deviation_index is None


def test_notify_deviation_from_hysteresis():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    manager._state = WindowState.HYSTERESIS

    manager._hysteresis_counter = 6

    manager.notify_deviation(50)

    assert manager._state == WindowState.ACTIVE_EVENT

    assert manager._deviation_index == 50

    assert manager._hysteresis_counter == 0


def test_update_hysteresis_from_active_event():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    manager._state = WindowState.ACTIVE_EVENT

    manager.update_hysteresis(False)

    assert manager._state == WindowState.HYSTERESIS

    assert manager._hysteresis_counter == 1


def test_update_hysteresis_counter_increment():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    manager._state = WindowState.HYSTERESIS

    manager._hysteresis_counter = 3

    manager.update_hysteresis(False)

    assert manager._state == WindowState.HYSTERESIS

    assert manager._hysteresis_counter == 4


def test_update_hysteresis_returns_to_active_event():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    manager._state = WindowState.HYSTERESIS

    manager._hysteresis_counter = 5

    manager.update_hysteresis(True)

    assert manager._state == WindowState.ACTIVE_EVENT

    assert manager._hysteresis_counter == 0


def test_update_hysteresis_finishes_event():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    manager._state = WindowState.HYSTERESIS

    manager._deviation_index = 20

    manager._current_window = [1, 2, 3]

    manager._hysteresis_counter = 7

    manager.update_hysteresis(False)

    assert manager._state == WindowState.IDLE

    assert manager._deviation_index is None

    assert manager._current_window is None

    assert manager._hysteresis_counter == 0


def test_update_hysteresis_from_idle():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    manager.update_hysteresis(False)

    assert manager._state == WindowState.IDLE

    assert manager._hysteresis_counter == 0


def test_update_hysteresis_from_waiting_window():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    manager._state = WindowState.WAITING_WINDOW

    manager.update_hysteresis(False)

    assert manager._state == WindowState.WAITING_WINDOW

    assert manager._hysteresis_counter == 0
