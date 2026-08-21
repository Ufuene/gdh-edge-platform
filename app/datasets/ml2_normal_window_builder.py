"""
ml2_normal_window_builder.py

Constrói janelas temporais normais para o dataset do ML2.

O ML2 trabalha com contexto temporal de oito amostras.

Cada amostra contém nove grandezas:

    Irradiancia
    Temperatura
    Vout
    Iout
    Ipv
    Vpv
    Iload
    Pout
    Ibat

Portanto:

    8 amostras × 9 grandezas = 72 features

As janelas normais devem representar uma sequência temporal
realmente contínua de operação normal.

Regra:

    amostras elegíveis para ML1
        ↓
    verificação de continuidade temporal
        ↓
    sequência contínua de 8 amostras
        ↓
    janela ML2 NORMAL
        ↓
    label = 0

IMPORTANTE
----------

Uma amostra normal que aparece depois de uma interrupção temporal
não deve ser simplesmente concatenada à sequência anterior.

Exemplo inválido:

    02:17:27
    02:17:37
    02:17:42

Existe um intervalo de 10 segundos entre as duas primeiras amostras.

Se o período esperado de aquisição é 5 segundos, essas amostras
não pertencem à mesma sequência temporal contínua.

Nesse caso, o buffer normal é reiniciado.

O componente NÃO realiza:

- inferência ML1;
- inferência ML2;
- detecção de desvios;
- classificação;
- persistência;
- comunicação com AWS.

Ele somente constrói janelas normais a partir de amostras
já consideradas elegíveis para ML1 e temporalmente contínuas.
"""

from collections import deque
from datetime import datetime
from typing import Deque, List

from app.data_manager.sample import Sample


class ML2NormalWindowBuilder:
    """
    Constrói janelas temporais normais e não sobrepostas para o ML2.

    Uma janela normal somente pode ser formada quando existem
    oito amostras consecutivas e temporalmente contínuas.

    O período temporal esperado entre duas amostras consecutivas
    é configurável e, por padrão, corresponde a 5 segundos.
    """

    NORMAL_LABEL = 0

    DEFAULT_WINDOW_SIZE = 8

    # O sistema Edge está configurado para adquirir uma amostra
    # aproximadamente a cada 5 segundos.
    DEFAULT_SAMPLE_INTERVAL_SECONDS = 5.0

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        expected_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    ):
        """
        Inicializa o construtor.

        Parameters
        ----------
        window_size : int
            Quantidade de amostras por janela normal.
            O valor padrão é 8.

        expected_interval_seconds : float
            Intervalo temporal esperado entre duas amostras
            consecutivas da mesma sequência.
            O valor padrão é 5 segundos.
        """

        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")

        if expected_interval_seconds <= 0:
            raise ValueError("expected_interval_seconds must be greater than zero")

        self._window_size = window_size

        self._expected_interval_seconds = expected_interval_seconds

        self._samples: Deque[Sample] = deque()

        self._ml1_sequence_counter = 0

        self._window_counter = 0

    # ==========================================================
    # CONTINUIDADE TEMPORAL
    # ==========================================================

    def _is_temporally_continuous(
        self,
        previous_sample: Sample,
        current_sample: Sample,
    ) -> bool:
        """
        Verifica se duas amostras pertencem à mesma sequência
        temporal contínua.

        A diferença entre os timestamps deve corresponder ao
        intervalo esperado de aquisição.

        Exemplo:

            02:17:27
            02:17:32

        diferença = 5 segundos
        → sequência contínua

        Já:

            02:17:27
            02:17:37

        diferença = 10 segundos
        → interrupção temporal
        """

        delta = (current_sample.timestamp - previous_sample.timestamp).total_seconds()

        return delta == self._expected_interval_seconds

    # ==========================================================
    # ADIÇÃO DE AMOSTRA
    # ==========================================================

    def add_sample(self, sample: Sample) -> bool:
        """
        Adiciona uma amostra elegível para ML1.

        A amostra somente permanece na sequência atual se for
        temporalmente consecutiva em relação à última amostra
        armazenada.

        Caso exista uma interrupção temporal, o buffer anterior
        é descartado e uma nova sequência é iniciada com a
        amostra atual.

        Returns
        -------
        bool
            True quando a adição completa uma nova janela.
            False caso contrário.
        """

        self._ml1_sequence_counter += 1

        # ------------------------------------------------------
        # Primeira amostra da sequência
        # ------------------------------------------------------

        if not self._samples:
            self._samples.append(sample)

            return len(self._samples) >= self._window_size

        # ------------------------------------------------------
        # Verificar continuidade temporal
        # ------------------------------------------------------

        previous_sample = self._samples[-1]

        if not self._is_temporally_continuous(
            previous_sample,
            sample,
        ):
            # --------------------------------------------------
            # Houve uma interrupção temporal.
            #
            # Não podemos misturar as duas sequências.
            # --------------------------------------------------

            self._samples.clear()

            self._samples.append(sample)

            return len(self._samples) >= self._window_size

        # ------------------------------------------------------
        # Continuidade confirmada
        # ------------------------------------------------------

        self._samples.append(sample)

        return len(self._samples) >= self._window_size

    # ==========================================================
    # JANELA COMPLETA
    # ==========================================================

    def has_complete_window(self) -> bool:
        """
        Verifica se existe uma janela normal completa.
        """

        return len(self._samples) >= self._window_size

    # ==========================================================
    # CONSTRUÇÃO DA JANELA
    # ==========================================================

    def build_window(self) -> List[Sample]:
        """
        Constrói e remove a próxima janela normal completa.

        A janela é formada pelas primeiras oito amostras da
        sequência temporal contínua ainda não consumida.

        Returns
        -------
        list[Sample]
            Janela normal.

        Raises
        ------
        RuntimeError
            Caso não existam amostras suficientes.
        """

        if not self.has_complete_window():
            raise RuntimeError("A complete ML2 normal window is not available.")

        window = [self._samples.popleft() for _ in range(self._window_size)]

        self._window_counter += 1

        return window

    # ==========================================================
    # ADICIONAR E CONSTRUIR
    # ==========================================================

    def add_sample_and_build_if_ready(
        self,
        sample: Sample,
    ) -> List[Sample] | None:
        """
        Adiciona uma amostra e constrói uma janela quando possível.

        A janela somente será produzida quando oito amostras
        temporalmente consecutivas estiverem disponíveis.

        Returns
        -------
        list[Sample] | None
            Nova janela normal quando oito amostras contínuas
            estiverem disponíveis; caso contrário, None.
        """

        self.add_sample(sample)

        if not self.has_complete_window():
            return None

        return self.build_window()

    # ==========================================================
    # CONSULTA
    # ==========================================================

    def pending_samples(self) -> int:
        """
        Retorna a quantidade de amostras temporalmente contínuas
        ainda aguardando completar uma janela.
        """

        return len(self._samples)

    def window_size(self) -> int:
        """
        Retorna o tamanho das janelas.
        """

        return self._window_size

    def expected_interval_seconds(self) -> float:
        """
        Retorna o intervalo temporal esperado entre amostras.
        """

        return self._expected_interval_seconds

    def ml1_sequence_count(self) -> int:
        """
        Retorna a quantidade total de amostras elegíveis para ML1
        recebidas pelo componente.

        Este contador representa a quantidade de amostras recebidas,
        e não a quantidade de amostras pertencentes a uma mesma
        sequência temporal.
        """

        return self._ml1_sequence_counter

    def window_count(self) -> int:
        """
        Retorna a quantidade de janelas normais construídas.
        """

        return self._window_counter

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self) -> None:
        """
        Limpa o estado do construtor.
        """

        self._samples.clear()

        self._ml1_sequence_counter = 0

        self._window_counter = 0
