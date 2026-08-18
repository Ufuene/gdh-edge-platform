"""
ml2_normal_window_builder.py

Constrói janelas temporais normais para o dataset do ML2.

As janelas normais são formadas exclusivamente a partir de amostras
elegíveis para o dataset ML1.

Regra:

    amostras elegíveis para ML1
        ↓
    sequência lógica ML1
        ↓
    grupos consecutivos de 8 amostras
        ↓
    janelas ML2 NORMAL
        ↓
    label = 0

As janelas são:

- sequenciais dentro do conjunto ML1;
- não sobrepostas;
- independentes dos índices originais da telemetria.

Exemplo:

    sequência ML1:

        1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

    janelas:

        [1..8]
        [9..16]

Uma interrupção na sequência original da telemetria não interrompe
a sequência lógica do dataset ML1.

Por exemplo:

    telemetria:
        1 2 3 4 5 6 17 18 19 20 ...

    ML1:
        1 2 3 4 5 6 17 18 19 20 ...

    ML2 normal:

        [1, 2, 3, 4, 5, 6, 17, 18]
        [19, 20, ...]

O componente NÃO realiza:

- inferência ML1;
- inferência ML2;
- detecção de desvios;
- classificação;
- persistência;
- comunicação com AWS.

Ele somente constrói janelas normais a partir de amostras
já consideradas elegíveis para ML1.
"""

from collections import deque
from typing import Deque, List

from app.data_manager.sample import Sample


class ML2NormalWindowBuilder:
    """
    Constrói janelas não sobrepostas de operação normal para o ML2.

    Cada oito amostras elegíveis para ML1 formam uma janela ML2
    com label normal = 0.
    """

    NORMAL_LABEL = 0

    def __init__(self, window_size: int = 8):
        """
        Inicializa o construtor.

        Parameters
        ----------
        window_size : int
            Quantidade de amostras por janela normal.
            O valor padrão é 8.
        """

        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")

        self._window_size = window_size

        self._samples: Deque[Sample] = deque()

        self._ml1_sequence_counter = 0
        self._window_counter = 0

    def add_sample(self, sample: Sample) -> bool:
        """
        Adiciona uma amostra elegível para ML1.

        A amostra recebe uma posição sequencial lógica dentro do
        fluxo ML1 através do contador interno.

        Returns
        -------
        bool
            True quando a adição completa uma nova janela.
            False caso contrário.
        """

        self._ml1_sequence_counter += 1

        self._samples.append(sample)

        return len(self._samples) >= self._window_size

    def has_complete_window(self) -> bool:
        """
        Verifica se existe uma janela normal completa.
        """

        return len(self._samples) >= self._window_size

    def build_window(self) -> List[Sample]:
        """
        Constrói e remove a próxima janela normal completa.

        A janela é formada pelas primeiras oito amostras da sequência
        ML1 ainda não consumidas.

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

    def add_sample_and_build_if_ready(
        self,
        sample: Sample,
    ) -> List[Sample] | None:
        """
        Adiciona uma amostra e constrói uma janela quando possível.

        Returns
        -------
        list[Sample] | None
            Nova janela normal quando oito amostras estiverem
            disponíveis; caso contrário, None.
        """

        self.add_sample(sample)

        if not self.has_complete_window():
            return None

        return self.build_window()

    def pending_samples(self) -> int:
        """
        Retorna a quantidade de amostras elegíveis ainda aguardando
        completar uma janela.
        """

        return len(self._samples)

    def window_size(self) -> int:
        """
        Retorna o tamanho das janelas.
        """

        return self._window_size

    def ml1_sequence_count(self) -> int:
        """
        Retorna a quantidade total de amostras elegíveis para ML1
        recebidas pelo componente.
        """

        return self._ml1_sequence_counter

    def window_count(self) -> int:
        """
        Retorna a quantidade de janelas normais construídas.
        """

        return self._window_counter

    def reset(self) -> None:
        """
        Limpa o estado do construtor.
        """

        self._samples.clear()

        self._ml1_sequence_counter = 0
        self._window_counter = 0
