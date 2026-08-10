"""
deviation_detector.py

Implementa o detector de desvios utilizado pelo
Gêmeo Digital Híbrido (GDH).

O DeviationDetector é responsável por comparar a potência
medida do sistema fotovoltaico com a potência prevista pelo ML1.

Responsabilidades:

- armazenar o limiar de potência;
- calcular o erro absoluto entre potência medida e prevista;
- determinar se o erro ultrapassou o limiar.

O DeviationDetector NÃO executa inferência do ML1.
O DeviationDetector NÃO executa inferência do ML2.
O DeviationDetector NÃO constrói janelas.

Essas responsabilidades pertencem a outros componentes.
"""


class DeviationDetector:
    """
    Detector de desvios baseado na diferença entre a potência
    medida e a potência prevista pelo ML1.
    """

    def __init__(self, power_threshold: float = 5.0):
        """
        Inicializa o DeviationDetector.

        Parameters
        ----------
        power_threshold : float
            Limiar de potência utilizado para determinar
            se existe um desvio.
        """

        self._power_threshold = power_threshold

    def compute_error(
        self,
        measured_power: float,
        predicted_power: float,
    ) -> float:
        """
        Calcula o erro absoluto entre a potência medida e a
        potência prevista pelo ML1.

        Parameters
        ----------
        measured_power : float
            Potência real medida pelo sistema.

        predicted_power : float
            Potência prevista pelo ML1.

        Returns
        -------
        float
            Erro absoluto de potência.
        """

        return abs(measured_power - predicted_power)

    def has_deviation(
        self,
        measured_power: float,
        predicted_power: float,
    ) -> bool:
        """
        Determina se existe um desvio de potência.

        O desvio é considerado existente somente quando o erro
        ultrapassa estritamente o limiar configurado.

        Portanto:

            erro <= power_threshold  -> False
            erro >  power_threshold  -> True

        Parameters
        ----------
        measured_power : float
            Potência real medida pelo sistema.

        predicted_power : float
            Potência prevista pelo ML1.

        Returns
        -------
        bool
            True se o erro ultrapassar o limiar.
            False caso contrário.
        """

        error = self.compute_error(
            measured_power,
            predicted_power,
        )

        return error > self._power_threshold
