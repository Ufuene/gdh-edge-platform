"""
ml1_service.py

Serviço de inferência do ML1 utilizado pelo
Gêmeo Digital Híbrido (GDH).

O ML1 é um modelo de regressão ponto a ponto.

Entradas:

- irradiance (G)
- temperature (T)
- v_out (Vout)

Saída:

- predicted_power (Pref)

O ML1 NÃO trabalha com janelas temporais.
Cada chamada a predict() corresponde a uma única amostra.

O carregamento do modelo não é responsabilidade deste serviço.
O modelo é fornecido ao ML1Service por injeção de dependência.
"""


class ML1Service:
    """
    Serviço responsável pela inferência do modelo ML1.
    """

    def __init__(self, model):
        """
        Inicializa o serviço ML1.

        Parameters
        ----------
        model :
            Modelo de regressão compatível com a interface
            de predição utilizada pelo serviço.

        Notes
        -----
        O modelo deve ser carregado externamente.
        O ML1Service é responsável apenas pela execução
        da inferência.
        """

        self._model = model

    def predict(
        self,
        irradiance: float,
        temperature: float,
        v_out: float,
    ) -> float:
        """
        Executa a inferência ponto a ponto do ML1.

        Parameters
        ----------
        irradiance : float
            Irradiância solar em W/m².

        temperature : float
            Temperatura em °C.

        v_out : float
            Tensão de saída em V.

        Returns
        -------
        float
            Potência de referência prevista pelo ML1 (Pref).

        Notes
        -----
        A ordem das características é:

            [Irradiancia, Temperatura, Vout]

        Essa ordem deve permanecer alinhada com a ordem
        utilizada durante o treinamento do modelo ML1.
        """

        features = [
            [
                irradiance,
                temperature,
                v_out,
            ]
        ]

        prediction = self._model.predict(features)

        return float(prediction[0])
