"""
window_manager.py

Implementa o gerenciador de janelas temporais utilizado pelo
Gêmeo Digital Híbrido (GDH).

O WindowManager é responsável por:

- gerenciar eventos de desvio;
- aguardar contexto temporal suficiente;
- construir janelas para o ML2;
- controlar a histerese;
- registrar os limites da janela temporal;
- registrar o início da próxima janela não sobreposta;
- controlar o consumo da janela pelo pipeline de inferência;
- encerrar eventos ativos.

Ele NÃO realiza inferência.
Ele NÃO detecta desvios.
Ele NÃO armazena amostras.

Essas responsabilidades pertencem aos demais componentes.
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

    O WindowManager separa dois conceitos:

    1. ciclo de vida do evento;
    2. disponibilidade/consumo das janelas temporais.

    Um evento pode passar por:

        IDLE
          ↓
        WAITING_WINDOW
          ↓
        ACTIVE_EVENT
          ↓
        HYSTERESIS
          ↓
        IDLE

    Enquanto o evento estiver ativo, podem ser construídas
    sucessivas janelas não sobrepostas.

    Exemplo:

        [0..7]
        [8..15]
        [16..23]

    A construção de uma nova janela não depende de um novo
    desvio e não altera diretamente o estado do evento.
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
        Retorna completamente o WindowManager ao estado inicial.

        IMPORTANTE
        ----------
        reset() é um reset completo.

        Portanto, além do estado do evento, também são apagados:

        - janela atual;
        - última janela;
        - índices da janela;
        - índice da próxima janela;
        - estado de consumo;
        - contador da histerese.

        O comportamento de preservação de uma janela durante o
        encerramento normal de um evento é tratado por end_event().
        """

        self._state = WindowState.IDLE

        self._deviation_index = None

        # ------------------------------------------------------
        # Janela corrente do evento
        # ------------------------------------------------------

        self._current_window = None

        self._current_window_start_index = None
        self._current_window_end_index = None

        # ------------------------------------------------------
        # Última janela do evento encerrado
        # ------------------------------------------------------

        # Preserva a última janela construída/classificada para
        # inspeção, sem confundi-la com a janela corrente de um
        # novo evento.

        self._last_window = None

        self._last_window_start_index = None
        self._last_window_end_index = None

        # ------------------------------------------------------
        # Próxima janela
        # ------------------------------------------------------

        self._next_window_start_index = None

        # ------------------------------------------------------
        # Controle de consumo
        # ------------------------------------------------------

        self._current_window_consumed = False

        # ------------------------------------------------------
        # Controle da histerese
        # ------------------------------------------------------

        self._hysteresis_counter = 0

    def end_event(self):
        """
        Encerra o evento atual preservando a última janela construída.

        A janela corrente deixa de ser a janela ativa do evento.

        Antes de encerrar:

            _current_window
                ↓
            _last_window

        Depois:

            _current_window = None
            _last_window = última janela construída

        Isso permite preservar o artefato temporal para inspeção
        sem deixar uma janela antiga sendo interpretada como a
        janela corrente de um novo evento.

        Diferença entre reset() e end_event()
        -------------------------------------

        reset():
            - limpa completamente o componente;
            - usado para reinicialização completa.

        end_event():
            - encerra somente o evento;
            - preserva a última janela construída;
            - limpa a janela corrente;
            - coloca o componente em IDLE.
        """

        # ======================================================
        # Preservar a última janela corrente
        # ======================================================

        if self._current_window is not None:
            self._last_window = self._current_window

            self._last_window_start_index = self._current_window_start_index

            self._last_window_end_index = self._current_window_end_index

        # ======================================================
        # Encerrar o evento
        # ======================================================

        self._state = WindowState.IDLE

        self._deviation_index = None

        self._hysteresis_counter = 0

        # ======================================================
        # A janela deixa de ser "corrente".
        #
        # Isso é fundamental para que um próximo evento não
        # interprete a janela anterior como uma janela já
        # disponível.
        # ======================================================

        self._current_window = None

        self._current_window_start_index = None
        self._current_window_end_index = None

        # ======================================================
        # Não existe próxima janela associada a um evento
        # encerrado.
        # ======================================================

        self._next_window_start_index = None

        self._current_window_consumed = False

    def notify_deviation(self, deviation_index: int):
        """
        Notifica o WindowManager que um desvio foi detectado.

        Um desvio somente pode iniciar um novo evento quando
        já existirem amostras anteriores suficientes para formar
        a janela temporal completa.

        Com:

            previous_samples = 3

        temos:

            deviation_index < 3
                → não inicia evento

            deviation_index >= 3
                → pode iniciar evento
        """

        # ======================================================
        # Contexto temporal mínimo
        # ======================================================

        if deviation_index < self._previous_samples:
            return

        # ======================================================
        # Novo evento
        # ======================================================

        if self._state == WindowState.IDLE:

            self._state = WindowState.WAITING_WINDOW

            self._deviation_index = deviation_index

            self._hysteresis_counter = 0

            return

        # ======================================================
        # Novo desvio durante a histerese
        # ======================================================

        if self._state == WindowState.HYSTERESIS:

            self._state = WindowState.ACTIVE_EVENT

            self._deviation_index = deviation_index

            self._hysteresis_counter = 0

            return

        # ======================================================
        # ACTIVE_EVENT e WAITING_WINDOW
        #
        # Um novo desvio não reinicia o evento já em tratamento.
        # ======================================================

    def add_sample(self):
        """
        Notifica a chegada de uma nova amostra.

        Quando o WindowManager estiver aguardando a primeira
        janela do evento, verifica se já existem amostras
        suficientes para construí-la.

        A janela possui:

            previous_samples
            +
            amostra do desvio
            +
            future_samples

        Com a configuração padrão:

            3 + 1 + 4 = 8 amostras.
        """

        # ======================================================
        # A construção inicial somente ocorre em
        # WAITING_WINDOW.
        # ======================================================

        if self._state != WindowState.WAITING_WINDOW:
            return

        if self._deviation_index is None:
            return

        latest_index = self._sample_buffer.size() - 1

        required_latest_index = self._deviation_index + self._future_samples

        if latest_index < required_latest_index:
            return

        # ======================================================
        # Determinar limites da janela.
        # ======================================================

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

        # ======================================================
        # Registrar janela corrente.
        # ======================================================

        self._current_window = window

        self._current_window_start_index = start_index

        self._current_window_end_index = end_index - 1

        # ======================================================
        # Registrar início da próxima janela não sobreposta.
        # ======================================================

        self._next_window_start_index = end_index

        # ======================================================
        # Nova janela ainda não consumida.
        # ======================================================

        self._current_window_consumed = False

        # ======================================================
        # Evento passa a estar ativo.
        # ======================================================

        self._state = WindowState.ACTIVE_EVENT

    def build_next_window(self) -> bool:
        """
        Constrói a próxima janela temporal não sobreposta.

        A próxima janela somente pode ser construída quando:

        1. existir um evento em tratamento;
        2. existir uma janela atual completa;
        3. a janela atual já tiver sido consumida;
        4. existirem amostras suficientes para formar a próxima
           janela.

        Exemplo:

            primeira janela:
                [0..7]

            segunda janela:
                [8..15]

            terceira janela:
                [16..23]

        A construção da próxima janela:

        - não depende de um novo desvio;
        - não altera o estado do evento;
        - substitui a janela atualmente armazenada;
        - registra o início da janela subsequente.
        """

        # ======================================================
        # Só pode haver continuidade enquanto existir evento.
        # ======================================================

        if self._state not in (
            WindowState.ACTIVE_EVENT,
            WindowState.HYSTERESIS,
        ):
            return False

        # ======================================================
        # Deve existir uma janela atual.
        #
        # Aqui não usamos apenas has_complete_window(), porque
        # esse método também reconhece _last_window após o
        # encerramento de um evento.
        # ======================================================

        if (
            self._current_window is None
            or len(self._current_window) != self._window_size
        ):
            return False

        # ======================================================
        # A janela atual precisa ter sido consumida.
        # ======================================================

        if not self._current_window_consumed:
            return False

        # ======================================================
        # Deve existir cursor para a próxima janela.
        # ======================================================

        if self._next_window_start_index is None:
            return False

        start_index = self._next_window_start_index

        # ======================================================
        # Verificar quantidade de amostras disponível.
        # ======================================================

        required_end_index = start_index + self._window_size - 1

        latest_index = self._sample_buffer.size() - 1

        if latest_index < required_end_index:
            return False

        # ======================================================
        # Limite final exclusivo.
        # ======================================================

        end_index = start_index + self._window_size

        window = self._sample_buffer.get_samples_range(
            start_index,
            end_index,
        )

        if len(window) != self._window_size:
            return False

        # ======================================================
        # Registrar nova janela corrente.
        # ======================================================

        self._current_window = window

        self._current_window_start_index = start_index

        self._current_window_end_index = end_index - 1

        # ======================================================
        # Preparar cursor da próxima janela.
        # ======================================================

        self._next_window_start_index = end_index

        # ======================================================
        # Nova janela ainda não consumida.
        # ======================================================

        self._current_window_consumed = False

        # ======================================================
        # IMPORTANTE:
        #
        # Não alterar _state.
        #
        # A construção da janela e o ciclo de vida do evento
        # são mecanismos independentes.
        # ======================================================

        return True

    def has_complete_window(self) -> bool:
        """
        Verifica se existe uma janela temporal completa disponível.

        A janela pode estar em dois contextos:

        1. janela corrente de um evento ativo;
        2. última janela preservada após o encerramento de um evento.

        Portanto, o método não depende exclusivamente de
        _current_window.

        Durante um evento:

            _current_window != None
                →
            janela corrente disponível

        Após o encerramento:

            _current_window == None
            _last_window != None
                →
            última janela disponível para inspeção

        IMPORTANTE
        ----------
        A existência de uma janela completa não significa que exista
        um evento ativo.

        Para verificar o ciclo de vida do evento utilizar:

            is_event_active()
        """

        if (
            self._current_window is not None
            and len(self._current_window) == self._window_size
        ):
            return True

        if (
            self._last_window is not None
            and len(self._last_window) == self._window_size
        ):
            return True

        return False

    def has_unconsumed_window(self) -> bool:
        """
        Verifica se existe uma janela corrente completa ainda
        não consumida pelo pipeline.

        A última janela preservada após o encerramento de um evento
        não é considerada disponível para nova classificação.
        """

        return (
            self._current_window is not None
            and len(self._current_window) == self._window_size
            and not self._current_window_consumed
        )

    def consume_current_window(self) -> None:
        """
        Marca a janela corrente como consumida.

        A janela não é removida.
        """

        if (
            self._current_window is None
            or len(self._current_window) != self._window_size
        ):
            return

        self._current_window_consumed = True

    def get_current_window(self):
        """
        Retorna a janela temporal atualmente disponível para inspeção.

        Durante um evento, retorna _current_window.

        Após o encerramento de um evento, quando _current_window foi
        limpo, retorna _last_window.

        Isso não reativa o evento.

        Returns
        -------
        list | None
            Janela atual ou última janela preservada.
        """

        if self._current_window is not None:
            return self._current_window

        return self._last_window

    def get_last_window(self):
        """
        Retorna a última janela preservada após o encerramento
        do evento.

        Returns
        -------
        list | None
            Última janela construída ou None.
        """

        return self._last_window

    def get_current_window_start_index(self):
        """
        Retorna o índice inicial da janela temporal disponível.

        Durante um evento, retorna o índice da janela corrente.

        Após o encerramento, retorna o índice da última janela
        preservada para inspeção.
        """

        if self._current_window is not None:
            return self._current_window_start_index

        return self._last_window_start_index

    def get_current_window_end_index(self):
        """
        Retorna o índice final da janela temporal disponível.

        Durante um evento, retorna o índice da janela corrente.

        Após o encerramento, retorna o índice da última janela
        preservada para inspeção.
        """

        if self._current_window is not None:
            return self._current_window_end_index

        return self._last_window_end_index

    def get_last_window_start_index(self):
        """
        Retorna o índice inicial da última janela preservada.
        """

        return self._last_window_start_index

    def get_last_window_end_index(self):
        """
        Retorna o índice final da última janela preservada.
        """

        return self._last_window_end_index

    def get_next_window_start_index(self):
        """
        Retorna o índice inicial previsto para a próxima janela.
        """

        return self._next_window_start_index

    def is_current_window_consumed(self) -> bool:
        """
        Verifica se a janela corrente já foi consumida.

        Se não existir uma janela corrente, mas existir uma última
        janela preservada após o encerramento do evento, ela é
        considerada consumida, pois somente janelas já classificadas
        são preservadas nesse estado.
        """

        if self._current_window is not None:
            return self._current_window_consumed

        if self._last_window is not None:
            return True

        return False

    def is_event_active(self) -> bool:
        """
        Verifica se existe um evento em tratamento.

        A existência de _last_window não significa que exista
        um evento ativo.
        """

        return self._state in (
            WindowState.WAITING_WINDOW,
            WindowState.ACTIVE_EVENT,
            WindowState.HYSTERESIS,
        )

    def update_hysteresis(self, deviation_detected: bool):
        """
        Atualiza o estado da histerese.

        Regras:

        ACTIVE_EVENT + sem desvio
            → HYSTERESIS, contador = 1

        HYSTERESIS + desvio
            → ACTIVE_EVENT, contador = 0

        HYSTERESIS + sem desvio
            → contador += 1

        contador >= hysteresis_samples
            → IDLE + end_event()
        """

        # ======================================================
        # Sem evento.
        # ======================================================

        if self._state == WindowState.IDLE:
            return

        # ======================================================
        # Ainda aguardando a primeira janela.
        # ======================================================

        if self._state == WindowState.WAITING_WINDOW:
            return

        # ======================================================
        # Evento ativo.
        # ======================================================

        if self._state == WindowState.ACTIVE_EVENT:

            if not deviation_detected:

                self._state = WindowState.HYSTERESIS

                self._hysteresis_counter = 1

            return

        # ======================================================
        # Histerese.
        # ======================================================

        if self._state == WindowState.HYSTERESIS:

            # --------------------------------------------------
            # Novo desvio.
            # --------------------------------------------------

            if deviation_detected:

                self._state = WindowState.ACTIVE_EVENT

                self._hysteresis_counter = 0

                return

            # --------------------------------------------------
            # Continua sem desvio.
            # --------------------------------------------------

            self._hysteresis_counter += 1

            # --------------------------------------------------
            # Encerrar evento.
            #
            # end_event() preserva a última janela construída,
            # mas remove a janela do contexto "current".
            # --------------------------------------------------

            if self._hysteresis_counter >= self._hysteresis_samples:

                self.end_event()
