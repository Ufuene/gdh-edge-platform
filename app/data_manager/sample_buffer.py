"""
sample_buffer.py

Implementa o buffer temporal utilizado pela Plataforma Edge AI.

O SampleBuffer é responsável exclusivamente por armazenar as amostras
recebidas do ESP32 em ordem cronológica.

Ele não executa inferência, não detecta desvios e não constrói janelas.
Essas responsabilidades pertencem aos demais componentes da plataforma.
"""

from collections import deque

from app.data_manager.sample import Sample


class SampleBuffer:
    """
    Buffer temporal de amostras.

    Armazena até 'max_size' amostras utilizando uma fila do tipo deque.
    """

    def __init__(self, max_size: int = 100):
        """
        Inicializa o buffer.

        Parameters
        ----------
        max_size : int
            Número máximo de amostras armazenadas.
        """

        # Armazena o tamanho máximo configurado do buffer.
        self._max_size = max_size

        # Buffer circular que mantém automaticamente apenas as
        # últimas 'max_size' amostras.
        self._buffer = deque(maxlen=max_size)

    def add_sample(self, sample: Sample):
        """
        Adiciona uma nova amostra ao buffer.

        Parameters
        ----------
        sample : Sample
            Amostra adquirida pelo sistema fotovoltaico.
        """

        self._buffer.append(sample)

    def get_latest(self):
        """
        Retorna a amostra mais recente.

        Returns
        -------
        Sample | None
            Última amostra armazenada ou None se o buffer estiver vazio.
        """
        pass

    def get_last_samples(self, n: int):
        """
        Retorna as últimas n amostras.

        Parameters
        ----------
        n : int
            Número de amostras desejadas.

        Returns
        -------
        list[Sample]
        """
        pass

    def get_sample(self, index: int):
        """
        Retorna uma amostra pelo índice.

        Parameters
        ----------
        index : int
            Índice da amostra.

        Returns
        -------
        Sample
        """
        pass

    def get_samples_range(self, start: int, end: int):
        """
        Retorna um intervalo de amostras.

        O índice 'start' é inclusivo e o índice 'end' é exclusivo.

        Parameters
        ----------
        start : int

        end : int

        Returns
        -------
        list[Sample]
        """
        pass

    def get_all(self):
        """
        Retorna todas as amostras armazenadas.

        Returns
        -------
        list[Sample]
        """
        pass

    def clear(self):
        """
        Remove todas as amostras do buffer.
        """
        pass

    def size(self) -> int:
        """
        Retorna a quantidade atual de amostras armazenadas.

        Returns
        -------
        int
            Número de amostras presentes no buffer.
        """

        return len(self._buffer)

    def is_empty(self) -> bool:
        """
        Verifica se o buffer está vazio.

        Returns
        -------
        bool
            True se não existir nenhuma amostra armazenada.
            False caso contrário.
        """

        return self.size() == 0

    def has_minimum_samples(self, n: int) -> bool:
        """
        Verifica se o buffer possui pelo menos n amostras.

        Parameters
        ----------
        n : int

        Returns
        -------
        bool
        """
        pass
