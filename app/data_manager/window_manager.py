"""
window_manager.py

Implementa o gerenciador de janelas temporais utilizado pelo
Gêmeo Digital Híbrido (GDH).

O WindowManager é responsável por:

- gerenciar eventos de desvio;
- aguardar contexto temporal suficiente;
- construir janelas para o ML2;
- controlar a histerese;
- encerrar eventos ativos.

Ele NÃO realiza inferência.
Ele NÃO detecta desvios.
Ele NÃO armazena amostras.

Essas responsabilidades pertencem a outros componentes.
"""

from enum import Enum, auto

from app.data_manager.sample_buffer import SampleBuffer


class WindowState(Enum):
    """
    Estados possíveis do WindowManager.
    """

    IDLE = auto()

    WAITING_WINDOW = auto()

    ACTIVE_EVENT = auto()

    HYSTERESIS = auto()


class WindowManager:
    """
    Gerencia a construção das janelas temporais utilizadas pelo ML2.
    """

    def __init__(
        self,
        sample_buffer: SampleBuffer,
        previous_samples: int = 3,
        future_samples: int = 4,
        hysteresis_samples: int = 8,
    ):
        """
        Inicializa o WindowManager.
        """

        self._sample_buffer = sample_buffer

        self._previous_samples = previous_samples
        self._future_samples = future_samples

        self._window_size = previous_samples + 1 + future_samples

        self._hysteresis_samples = hysteresis_samples

        self.reset()

    def reset(self):
        """
        Retorna o WindowManager ao estado inicial.
        """

        self._state = WindowState.IDLE

        self._deviation_index = None

        self._current_window = None

        self._hysteresis_counter = 0

    def notify_deviation(self, deviation_index: int):
        """
        Notifica o WindowManager que um desvio foi detectado.

        Parameters
        ----------
        deviation_index : int
            Índice da amostra onde o desvio foi detectado.
        """

        if self._state == WindowState.IDLE:

            self._state = WindowState.WAITING_WINDOW

            self._deviation_index = deviation_index

        elif self._state == WindowState.HYSTERESIS:

            self._state = WindowState.ACTIVE_EVENT

            self._deviation_index = deviation_index

            self._hysteresis_counter = 0

    def add_sample(self):
        """
        Notifica a chegada de uma nova amostra.
        """
        pass

    def has_complete_window(self) -> bool:
        """
        Verifica se existe uma janela completa disponível.
        """
        pass

    def get_current_window(self):
        """
        Retorna a janela atual.
        """
        pass

    def is_event_active(self) -> bool:
        """
        Verifica se existe um evento ativo.

        Returns
        -------
        bool
            True se o WindowManager estiver tratando um evento.
        """

        return self._state in (
            WindowState.WAITING_WINDOW,
            WindowState.ACTIVE_EVENT,
            WindowState.HYSTERESIS,
        )

    def update_hysteresis(self, deviation_detected: bool):
        """
        Atualiza o estado da histerese.

        Parameters
        ----------
        deviation_detected : bool
            True se ainda existir desvio.
            False caso contrário.
        """

        # Não existe evento ativo.
        if self._state == WindowState.IDLE:
            return

        # Ainda aguardando a primeira janela.
        if self._state == WindowState.WAITING_WINDOW:
            return

        # Evento ativo.
        if self._state == WindowState.ACTIVE_EVENT:

            if not deviation_detected:

                self._state = WindowState.HYSTERESIS

                self._hysteresis_counter = 1

            return

        # Estado de histerese.
        if self._state == WindowState.HYSTERESIS:

            # Novo desvio detectado.
            if deviation_detected:

                self._state = WindowState.ACTIVE_EVENT

                self._hysteresis_counter = 0

                return

            # Continua abaixo do limiar.
            self._hysteresis_counter += 1

            if self._hysteresis_counter >= self._hysteresis_samples:
                self.reset()
