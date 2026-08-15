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

        Um desvio somente pode iniciar um novo evento quando
        já existirem amostras anteriores suficientes para formar
        a janela temporal completa.

        Com a configuração padrão:

            previous_samples = 3

        portanto:

            deviation_index < 3
                → não inicia evento

            deviation_index >= 3
                → pode iniciar evento

        Parameters
        ----------
        deviation_index : int
            Índice da amostra onde o desvio foi detectado.
        """

        # ======================================================
        # Verificar contexto temporal mínimo
        # ======================================================

        if deviation_index < self._previous_samples:
            return

        # ======================================================
        # Evento inicial
        # ======================================================

        if self._state == WindowState.IDLE:

            self._state = WindowState.WAITING_WINDOW

            self._deviation_index = deviation_index

        # ======================================================
        # Novo desvio durante a histerese
        # ======================================================

        elif self._state == WindowState.HYSTERESIS:

            self._state = WindowState.ACTIVE_EVENT

            self._deviation_index = deviation_index

            self._hysteresis_counter = 0

    def add_sample(self):
        """
        Notifica a chegada de uma nova amostra.

        Quando o WindowManager estiver aguardando uma janela,
        verifica se já existem amostras suficientes para construir
        uma janela completa.

        A janela possui exatamente:

            previous_samples
            +
            amostra do desvio
            +
            future_samples

        Com a configuração padrão:

            3 + 1 + 4 = 8 amostras.
        """

        if self._state != WindowState.WAITING_WINDOW:
            return

        if self._deviation_index is None:
            return

        latest_index = self._sample_buffer.size() - 1

        required_latest_index = self._deviation_index + self._future_samples

        if latest_index < required_latest_index:
            return

        start_index = self._deviation_index - self._previous_samples

        end_index = self._deviation_index + self._future_samples + 1

        if start_index < 0:
            return

        window = self._sample_buffer.get_samples_range(
            start_index,
            end_index,
        )

        if len(window) != self._window_size:
            return

        self._current_window = window

        self._state = WindowState.ACTIVE_EVENT

    def has_complete_window(self) -> bool:
        """
        Verifica se existe uma janela completa disponível.

        Returns
        -------
        bool
            True se existir uma janela completa.
            False caso contrário.
        """

        return (
            self._current_window is not None
            and len(self._current_window) == self._window_size
        )

    def get_current_window(self):
        """
        Retorna a janela atual.

        Returns
        -------
        list | None
            Janela atual ou None se ainda não existir uma janela.
        """

        return self._current_window

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
