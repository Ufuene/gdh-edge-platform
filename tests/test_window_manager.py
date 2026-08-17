from datetime import datetime

from app.data_manager.sample import Sample
from app.data_manager.sample_buffer import SampleBuffer
from app.data_manager.window_manager import (
    WindowManager,
    WindowState,
)


def create_sample(index: int) -> Sample:
    """
    Cria uma amostra de teste identificável pelo índice.
    """

    return Sample(
        timestamp=datetime.now(),
        irradiance=800 + index,
        temperature=25,
        v_pv=30,
        i_pv=7,
        v_out=14.5,
        i_out=8,
        i_bat=3,
        i_load=5,
        p_out=100 + index,
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


def test_notify_deviation_requires_previous_context():
    """
    Verifica que um desvio não pode iniciar um evento antes
    de existir o número mínimo de amostras anteriores.

    Com previous_samples=3:

        índice 0 → rejeitado
        índice 1 → rejeitado
        índice 2 → rejeitado
        índice 3 → aceito
    """

    buffer = SampleBuffer()

    manager = WindowManager(
        buffer,
        previous_samples=3,
        future_samples=4,
    )

    manager.notify_deviation(0)

    assert manager._state == WindowState.IDLE
    assert manager._deviation_index is None

    manager.notify_deviation(1)

    assert manager._state == WindowState.IDLE
    assert manager._deviation_index is None

    manager.notify_deviation(2)

    assert manager._state == WindowState.IDLE
    assert manager._deviation_index is None

    manager.notify_deviation(3)

    assert manager._state == WindowState.WAITING_WINDOW
    assert manager._deviation_index == 3


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


def test_add_sample_waits_for_four_future_samples():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    for index in range(7):
        buffer.add_sample(create_sample(index))

    manager.notify_deviation(3)

    manager.add_sample()

    assert manager._state == WindowState.WAITING_WINDOW

    assert not manager.has_complete_window()

    assert manager.get_current_window() is None


def test_add_sample_builds_eight_sample_window():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    for index in range(8):
        buffer.add_sample(create_sample(index))

    manager.notify_deviation(3)

    manager.add_sample()

    assert manager._state == WindowState.ACTIVE_EVENT

    assert manager.has_complete_window()

    window = manager.get_current_window()

    assert len(window) == 8

    assert window[0] == buffer.get_sample(0)
    assert window[1] == buffer.get_sample(1)
    assert window[2] == buffer.get_sample(2)

    assert window[3] == buffer.get_sample(3)

    assert window[4] == buffer.get_sample(4)
    assert window[5] == buffer.get_sample(5)
    assert window[6] == buffer.get_sample(6)
    assert window[7] == buffer.get_sample(7)


def test_add_sample_does_not_build_window_when_deviation_is_last_sample():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    for index in range(8):
        buffer.add_sample(create_sample(index))

    manager.notify_deviation(7)

    manager.add_sample()

    assert manager._state == WindowState.WAITING_WINDOW

    assert not manager.has_complete_window()

    assert manager.get_current_window() is None


def test_add_sample_builds_window_with_three_previous_and_four_future():

    buffer = SampleBuffer()

    manager = WindowManager(buffer)

    for index in range(12):
        buffer.add_sample(create_sample(index))

    manager.notify_deviation(7)

    manager.add_sample()

    assert manager._state == WindowState.ACTIVE_EVENT

    assert manager.has_complete_window()

    window = manager.get_current_window()

    assert len(window) == 8

    assert window[0] == buffer.get_sample(4)
    assert window[1] == buffer.get_sample(5)
    assert window[2] == buffer.get_sample(6)
    assert window[3] == buffer.get_sample(7)
    assert window[4] == buffer.get_sample(8)
    assert window[5] == buffer.get_sample(9)
    assert window[6] == buffer.get_sample(10)
    assert window[7] == buffer.get_sample(11)


def test_complete_event_lifecycle_with_hysteresis():
    """
    Verifica o ciclo completo de um evento:

        IDLE
        ↓
        WAITING_WINDOW
        ↓
        ACTIVE_EVENT
        ↓
        HYSTERESIS
        ↓
        ACTIVE_EVENT
        ↓
        HYSTERESIS
        ↓
        IDLE
    """

    buffer = SampleBuffer()

    manager = WindowManager(
        buffer,
        previous_samples=3,
        future_samples=4,
        hysteresis_samples=8,
    )

    # ==========================================================
    # Construir contexto suficiente antes do desvio.
    # ==========================================================

    for index in range(8):
        buffer.add_sample(create_sample(index))

    # ==========================================================
    # Desvio no índice 3.
    # ==========================================================

    manager.notify_deviation(3)

    assert manager._state == WindowState.WAITING_WINDOW
    assert manager._deviation_index == 3

    # ==========================================================
    # Chegada da quarta amostra futura.
    # ==========================================================

    manager.add_sample()

    assert manager._state == WindowState.ACTIVE_EVENT
    assert manager.has_complete_window()

    window = manager.get_current_window()

    assert window is not None
    assert len(window) == 8

    # ==========================================================
    # O desvio desaparece.
    # ==========================================================

    manager.update_hysteresis(False)

    assert manager._state == WindowState.HYSTERESIS
    assert manager._hysteresis_counter == 1

    # ==========================================================
    # Novo desvio durante a histerese.
    # ==========================================================

    manager.notify_deviation(7)

    assert manager._state == WindowState.ACTIVE_EVENT
    assert manager._deviation_index == 7
    assert manager._hysteresis_counter == 0

    # ==========================================================
    # Novo período sem desvio.
    # ==========================================================

    manager.update_hysteresis(False)

    assert manager._state == WindowState.HYSTERESIS
    assert manager._hysteresis_counter == 1

    # ==========================================================
    # Ainda em histerese após 7 amostras sem desvio.
    # ==========================================================

    for _ in range(6):
        manager.update_hysteresis(False)

    assert manager._state == WindowState.HYSTERESIS
    assert manager._hysteresis_counter == 7

    # ==========================================================
    # O oitavo período sem desvio encerra o evento.
    # ==========================================================

    manager.update_hysteresis(False)

    assert manager._state == WindowState.IDLE
    assert manager._deviation_index is None
    assert manager._current_window is None
    assert manager._hysteresis_counter == 0


def test_invalid_early_deviation_does_not_break_hysteresis():
    """
    Verifica que um desvio sem contexto temporal suficiente
    não reativa um evento durante a histerese.
    """

    buffer = SampleBuffer()

    manager = WindowManager(
        buffer,
        previous_samples=3,
        future_samples=4,
    )

    manager._state = WindowState.HYSTERESIS
    manager._hysteresis_counter = 3
    manager._deviation_index = 10

    manager.notify_deviation(2)

    assert manager._state == WindowState.HYSTERESIS
    assert manager._deviation_index == 10
    assert manager._hysteresis_counter == 3


def test_window_manager_registers_next_non_overlapping_window_start():
    """
    Verifica que, após construir uma janela de 8 amostras,
    o WindowManager registra corretamente o início da próxima
    janela não sobreposta.
    """

    buffer = SampleBuffer()

    manager = WindowManager(
        buffer,
        previous_samples=3,
        future_samples=4,
    )

    for index in range(8):
        buffer.add_sample(create_sample(index))

    manager.notify_deviation(3)

    manager.add_sample()

    assert manager.has_complete_window()

    assert manager.get_current_window_start_index() == 0
    assert manager.get_current_window_end_index() == 7

    assert manager.get_next_window_start_index() == 8


def test_window_manager_next_window_starts_after_current_window():
    """
    Verifica explicitamente a regra de janelas não sobrepostas.

    Para uma janela:

        [2, 3, 4, 5, 6, 7, 8, 9]

    a próxima deverá começar em:

        10
    """

    buffer = SampleBuffer()

    manager = WindowManager(
        buffer,
        previous_samples=3,
        future_samples=4,
    )

    for index in range(10):
        buffer.add_sample(create_sample(index))

    manager.notify_deviation(5)

    manager.add_sample()

    assert manager.has_complete_window()

    assert manager.get_current_window_start_index() == 2
    assert manager.get_current_window_end_index() == 9

    assert manager.get_next_window_start_index() == 10


def test_window_manager_does_not_build_next_window_before_consumption():
    """
    Verifica que uma nova janela não sobreposta não é construída
    enquanto a janela atual ainda não tiver sido consumida.
    """

    buffer = SampleBuffer()

    manager = WindowManager(
        buffer,
        previous_samples=3,
        future_samples=4,
    )

    # ----------------------------------------------------------
    # Primeira janela:
    #
    # [0, 1, 2, 3, 4, 5, 6, 7]
    # ----------------------------------------------------------

    for index in range(8):
        buffer.add_sample(create_sample(index))

    manager.notify_deviation(3)
    manager.add_sample()

    assert manager.has_complete_window()
    assert manager.get_current_window_start_index() == 0
    assert manager.get_current_window_end_index() == 7
    assert manager.get_next_window_start_index() == 8

    # ----------------------------------------------------------
    # A janela ainda não foi consumida.
    # ----------------------------------------------------------

    assert manager.is_current_window_consumed() is False

    # ----------------------------------------------------------
    # Adicionar amostras suficientes para a próxima janela
    # não deve substituí-la enquanto ela não for consumida.
    # ----------------------------------------------------------

    for index in range(8, 16):
        buffer.add_sample(create_sample(index))

    manager.add_sample()

    assert manager.get_current_window_start_index() == 0
    assert manager.get_current_window_end_index() == 7

    window = manager.get_current_window()

    assert window is not None
    assert len(window) == 8

    assert window[0] == buffer.get_sample(0)
    assert window[7] == buffer.get_sample(7)


def test_window_manager_builds_next_window_after_consumption():
    """
    Verifica que, depois de consumir a janela atual, o
    WindowManager consegue construir a próxima janela
    não sobreposta quando houver amostras suficientes.
    """

    buffer = SampleBuffer()

    manager = WindowManager(
        buffer,
        previous_samples=3,
        future_samples=4,
    )

    # ----------------------------------------------------------
    # Primeira janela:
    #
    # [0, 1, 2, 3, 4, 5, 6, 7]
    # ----------------------------------------------------------

    for index in range(8):
        buffer.add_sample(create_sample(index))

    manager.notify_deviation(3)
    manager.add_sample()

    assert manager.has_complete_window()

    assert manager.get_current_window_start_index() == 0
    assert manager.get_current_window_end_index() == 7
    assert manager.get_next_window_start_index() == 8

    # ----------------------------------------------------------
    # Consumir primeira janela.
    # ----------------------------------------------------------

    manager.consume_current_window()

    assert manager.is_current_window_consumed() is True

    # ----------------------------------------------------------
    # Adicionar as oito amostras seguintes.
    #
    # Segunda janela esperada:
    #
    # [8, 9, 10, 11, 12, 13, 14, 15]
    # ----------------------------------------------------------

    for index in range(8, 16):
        buffer.add_sample(create_sample(index))

    # ----------------------------------------------------------
    # Solicitar construção da próxima janela.
    # ----------------------------------------------------------

    manager.build_next_window()

    assert manager.has_complete_window()

    assert manager.get_current_window_start_index() == 8
    assert manager.get_current_window_end_index() == 15

    assert manager.get_next_window_start_index() == 16

    window = manager.get_current_window()

    assert window is not None
    assert len(window) == 8

    for position, expected_index in enumerate(range(8, 16)):
        assert window[position] == buffer.get_sample(expected_index)


def test_window_manager_next_window_remains_unavailable_without_enough_samples():
    """
    Verifica que o WindowManager não constrói a próxima janela
    antes de existirem as oito amostras necessárias.
    """

    buffer = SampleBuffer()

    manager = WindowManager(
        buffer,
        previous_samples=3,
        future_samples=4,
    )

    # ----------------------------------------------------------
    # Primeira janela.
    # ----------------------------------------------------------

    for index in range(8):
        buffer.add_sample(create_sample(index))

    manager.notify_deviation(3)
    manager.add_sample()

    assert manager.has_complete_window()

    manager.consume_current_window()

    # ----------------------------------------------------------
    # Apenas sete amostras da próxima janela.
    # ----------------------------------------------------------

    for index in range(8, 15):
        buffer.add_sample(create_sample(index))

    manager.build_next_window()

    # ----------------------------------------------------------
    # A janela anterior continua registrada porque ainda não
    # existe contexto suficiente para substituir a janela.
    # ----------------------------------------------------------

    assert manager.get_current_window_start_index() == 0
    assert manager.get_current_window_end_index() == 7

    window = manager.get_current_window()

    assert window is not None
    assert len(window) == 8

    assert window[0] == buffer.get_sample(0)
    assert window[7] == buffer.get_sample(7)
