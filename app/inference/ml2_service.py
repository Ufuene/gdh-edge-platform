"""
ml2_service.py

Serviço de inferência do ML2 utilizado pelo
Gêmeo Digital Híbrido (GDH).

O ML2 é um modelo de classificação baseado em
janelas temporais de 8 amostras.

Cada amostra contém 9 características:

- irradiance
- temperature
- v_out
- i_out
- i_pv
- v_pv
- i_load
- p_out
- i_bat

A janela completa contém:

    8 amostras × 9 features = 72 features

O ML2 recebe uma janela temporal já construída pelo
WindowManager e realiza a classificação da condição
operacional.

O ML2Service NÃO constrói janelas.
O ML2Service NÃO detecta desvios.
O ML2Service NÃO controla a janela temporal.

Essas responsabilidades pertencem ao WindowManager
e ao InferenceOrchestrator.
"""

import numpy as np

from app.data_manager.sample import Sample


class ML2Service:
    """
    Serviço responsável pela inferência do modelo ML2.
    """

    FEATURES_PER_SAMPLE = 9
    WINDOW_SIZE = 8
    TOTAL_FEATURES = FEATURES_PER_SAMPLE * WINDOW_SIZE

    def __init__(self, model):
        """
        Inicializa o serviço ML2.

        Parameters
        ----------
        model :
            Modelo de classificação compatível com as
            interfaces predict() e predict_proba().
        """

        self._model = model

    def _build_features(self, window: list[Sample]) -> np.ndarray:
        """
        Constrói a matriz de entrada do ML2.

        Parameters
        ----------
        window : list[Sample]
            Janela temporal contendo exatamente 8 amostras.

        Returns
        -------
        numpy.ndarray
            Matriz float32 com shape (1, 72).
        """

        if len(window) != self.WINDOW_SIZE:
            raise ValueError(
                f"ML2 requires exactly {self.WINDOW_SIZE} samples, "
                f"received {len(window)}"
            )

        features = []

        for sample in window:
            features.extend(
                [
                    sample.irradiance,
                    sample.temperature,
                    sample.v_out,
                    sample.i_out,
                    sample.i_pv,
                    sample.v_pv,
                    sample.i_load,
                    sample.p_out,
                    sample.i_bat,
                ]
            )

        if len(features) != self.TOTAL_FEATURES:
            raise ValueError(
                f"ML2 requires {self.TOTAL_FEATURES} features, "
                f"generated {len(features)}"
            )

        return np.asarray(
            features,
            dtype=np.float32,
        ).reshape(1, self.TOTAL_FEATURES)

    def predict(self, window: list[Sample]) -> int:
        """
        Executa a inferência do ML2 sobre uma janela completa.

        Parameters
        ----------
        window : list[Sample]
            Janela temporal contendo exatamente 8 amostras.

        Returns
        -------
        int
            Diagnóstico produzido pelo ML2.

            0 = condição normal
            1 = condição com partial shading
        """

        features = self._build_features(window)

        prediction = self._model.predict(features)

        return int(prediction[0])

    def predict_proba(
        self,
        window: list[Sample],
    ) -> tuple[float, float]:
        """
        Retorna as probabilidades das duas classes do ML2.

        Parameters
        ----------
        window : list[Sample]
            Janela temporal contendo exatamente 8 amostras.

        Returns
        -------
        tuple[float, float]
            Probabilidades na ordem:

                (probability_normal, probability_psc)

            correspondentes às classes:

                0 = NORMAL
                1 = PSC
        """

        features = self._build_features(window)

        probability = self._model.predict_proba(features)

        return (
            float(probability[0][0]),
            float(probability[0][1]),
        )
