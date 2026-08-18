"""
dataset_manager.py

Orquestra a seleção de dados produzidos pelo pipeline de inferência
para posterior formação dos datasets ML1 e ML2.

O DatasetManager NÃO:

- executa ML1;
- executa ML2;
- detecta desvios;
- constrói janelas;
- envia dados para AWS.

Ele somente transforma resultados já produzidos pela inferência
em registros elegíveis para os datasets.

Regras congeladas:

CASO A — saudável
-----------------

deviation_detected = False
ml2_executed = False

    ML1 -> elegível
    ML2 -> elegível como NORMAL (label 0), por meio do construtor
           sequencial de janelas normais.

CASO B — PSC
------------

ml2_executed = True
diagnosis = 1
window_samples disponíveis

    ML1 -> depende do estado da amostra atual
    ML2 -> elegível como PSC (label 1)

CASO C — falso positivo / diagnóstico normal durante desvio
------------------------------------------------------------

deviation_detected = True
ml2_executed = True
diagnosis = 0

    ML1 -> não elegível para a amostra com desvio
    ML2 -> não elegível

CASO D — desvio aguardando janela
---------------------------------

deviation_detected = True
ml2_executed = False

    ML1 -> não elegível
    ML2 -> ainda não elegível

O ponto importante é que a classificação ML2 pertence à janela
temporal efetivamente construída e executada, e não necessariamente
à amostra que fecha a janela.

Consequentemente, ML1 e ML2 são processados de forma independente.

Exemplo:

    índice 3:
        desvio
        -> ML1 não elegível

    índices 4..7:
        amostras saudáveis
        -> ML1 elegível

    índice 7:
        fecha a janela [0..7]
        ML2 executa
        diagnosis = 1
        -> janela ML2 = PSC

Nesse caso, a mesma chamada do DatasetManager pode produzir:

    ML1DatasetRecord
    +
    ML2DatasetWindow(label=1)
"""

from typing import Optional

from app.data_manager.sample import Sample
from app.datasets.ml1_dataset import ML1DatasetRecord
from app.datasets.ml2_dataset import ML2DatasetWindow
from app.datasets.ml2_normal_window_builder import ML2NormalWindowBuilder
from app.inference.inference_result import InferenceResult


class DatasetManager:
    """
    Gerencia a elegibilidade e transformação dos dados de inferência.
    """

    def __init__(self):
        """
        Inicializa o DatasetManager.
        """

        self._ml1_records: list[ML1DatasetRecord] = []

        self._ml2_records: list[ML2DatasetWindow] = []

        self._ml2_normal_window_builder = ML2NormalWindowBuilder(window_size=8)

        # ------------------------------------------------------
        # Contador independente para janelas PSC.
        # ------------------------------------------------------

        self._psc_window_count = 0

    # ==========================================================
    # Processamento de amostra
    # ==========================================================

    def process_sample(
        self,
        sample: Sample,
        result: InferenceResult,
    ) -> tuple[
        Optional[ML1DatasetRecord],
        Optional[ML2DatasetWindow],
    ]:
        """
        Processa uma amostra após a inferência.

        Existem dois caminhos independentes:

        1. dados normais:
           a amostra pode alimentar ML1 e o construtor das
           janelas normais do ML2;

        2. janela efetivamente classificada pelo ML2:
           quando diagnosis == 1, a janela temporal preservada
           no InferenceResult é registrada como PSC.

        A classificação ML2 não depende do estado isolado da
        amostra atual.

        IMPORTANTE
        ----------
        Uma mesma chamada pode produzir simultaneamente:

            ML1DatasetRecord
            +
            ML2DatasetWindow

        Isso ocorre quando a amostra atual é saudável e,
        simultaneamente, completa uma janela temporal que foi
        iniciada por um desvio anterior.
        """

        ml1_record = None
        ml2_window = None

        # ======================================================
        # 1. PROCESSAMENTO DA JANELA ML2
        # ======================================================
        #
        # A janela ML2 deve ser processada independentemente
        # da elegibilidade da amostra atual para ML1.
        #
        # Exemplo:
        #
        # desvio no índice 3
        # janela [0..7]
        # ML2 executado no índice 7
        #
        # No índice 7:
        #
        #     deviation_detected == False
        #     ml2_executed == True
        #     diagnosis == 1
        #
        # A janela [0..7] continua sendo PSC.
        # ======================================================

        if (
            result.ml2_executed
            and result.diagnosis == 1
            and result.window_samples is not None
        ):
            ml2_window = self._create_psc_window(result)

            if ml2_window is not None:
                self.register_ml2(ml2_window)

        # ======================================================
        # 2. PROCESSAMENTO ML1
        # ======================================================
        #
        # A elegibilidade ML1 depende exclusivamente do estado
        # da amostra atual.
        #
        # Portanto:
        #
        #     deviation_detected == True
        #         -> não elegível
        #
        #     deviation_detected == False
        #         -> elegível
        #
        # Isso permite que uma amostra saudável que fecha uma
        # janela PSC continue sendo registrada no dataset ML1.
        # ======================================================

        if not self.is_ml1_eligible(result):
            return ml1_record, ml2_window

        # ======================================================
        # 3. Criar registro ML1
        # ======================================================

        ml1_record = self.create_ml1_record(
            sample=sample,
            result=result,
        )

        if ml1_record is None:
            return ml1_record, ml2_window

        # ======================================================
        # 4. Registrar dado no dataset ML1
        # ======================================================

        self.register_ml1(ml1_record)

        # ======================================================
        # 5. Alimentar construtor das janelas normais ML2
        # ======================================================
        #
        # Somente amostras saudáveis alimentam o construtor
        # sequencial das janelas normais.
        # ======================================================

        window_samples = self._ml2_normal_window_builder.add_sample_and_build_if_ready(
            sample
        )

        # ======================================================
        # Ainda não existem oito amostras normais elegíveis.
        #
        # Se uma janela PSC já tiver sido produzida acima,
        # ela deve continuar sendo retornada.
        # ======================================================

        if window_samples is None:
            return ml1_record, ml2_window

        # ======================================================
        # 6. Uma nova janela normal foi formada.
        # ======================================================

        window_number = self._ml2_normal_window_builder.window_count()

        window_start = ((window_number - 1) * 8) + 1

        window_end = window_start + len(window_samples) - 1

        normal_window = ML2DatasetWindow.from_window(
            window_id=f"normal-{window_number:06d}",
            window_start=window_start,
            window_end=window_end,
            samples=window_samples,
            label=0,
        )

        # ======================================================
        # 7. Registrar janela normal no dataset ML2
        # ======================================================

        self.register_ml2(normal_window)

        # ======================================================
        # 8. Retornar o resultado da chamada
        # ======================================================
        #
        # Se uma janela PSC já tiver sido produzida nesta mesma
        # chamada, ela tem prioridade como objeto retornado.
        #
        # A janela normal, entretanto, permanece registrada em
        # _ml2_records.
        # ======================================================

        if ml2_window is not None:
            return ml1_record, ml2_window

        return ml1_record, normal_window

    # ==========================================================
    # Construção de janela PSC
    # ==========================================================

    def _create_psc_window(
        self,
        result: InferenceResult,
    ) -> Optional[ML2DatasetWindow]:
        """
        Cria uma janela PSC a partir da janela efetivamente
        utilizada pelo ML2.

        A função não constrói a janela temporal.

        A janela já foi construída pelo WindowManager e preservada
        pelo InferenceResult.
        """

        if not result.ml2_executed:
            return None

        if result.diagnosis != 1:
            return None

        if result.window_samples is None:
            return None

        if not result.window_samples:
            return None

        if result.window_start is None:
            return None

        if result.window_end is None:
            return None

        # ------------------------------------------------------
        # Incrementar contador PSC.
        # ------------------------------------------------------

        self._psc_window_count += 1

        window_id = f"psc-{self._psc_window_count:06d}"

        # ------------------------------------------------------
        # Criar registro.
        # ------------------------------------------------------

        return ML2DatasetWindow.from_window(
            window_id=window_id,
            window_start=result.window_start,
            window_end=result.window_end,
            samples=result.window_samples,
            label=1,
        )

    # ==========================================================
    # ML1
    # ==========================================================

    @staticmethod
    def is_ml1_eligible(
        result: InferenceResult,
    ) -> bool:
        """
        Determina se a amostra pode entrar no dataset ML1.

        Regra:

            deviation_detected == False
                -> True

            deviation_detected == True
                -> False

        IMPORTANTE
        ----------
        Esta função avalia somente o estado da amostra atual.

        A classificação de uma janela pelo ML2 é tratada
        separadamente.
        """

        return not result.deviation_detected

    def create_ml1_record(
        self,
        sample: Sample,
        result: InferenceResult,
    ) -> Optional[ML1DatasetRecord]:
        """
        Cria um registro ML1 se a amostra for elegível.
        """

        if not self.is_ml1_eligible(result):
            return None

        return ML1DatasetRecord.from_sample(sample)

    # ==========================================================
    # ML2
    # ==========================================================

    @staticmethod
    def determine_ml2_label(
        result: InferenceResult,
    ) -> Optional[int]:
        """
        Determina o label ML2.

        Retorna:

            0 -> operação normal elegível
            1 -> PSC diagnosticado pelo ML2
            None -> não elegível

        IMPORTANTE
        ----------
        Para uma janela efetivamente classificada pelo ML2,
        a decisão de label deve considerar a execução do ML2
        e seu diagnóstico.

        Para dados normais, o construtor sequencial de janelas
        produz explicitamente label 0.
        """

        # ------------------------------------------------------
        # CASO A — operação normal
        # ------------------------------------------------------

        if not result.deviation_detected:
            return 0

        # ------------------------------------------------------
        # CASO B — desvio + ML2 + PSC
        # ------------------------------------------------------

        if result.deviation_detected and result.ml2_executed and result.diagnosis == 1:
            return 1

        # ------------------------------------------------------
        # CASO C — desvio + diagnóstico normal
        # ------------------------------------------------------

        return None

    @classmethod
    def is_ml2_eligible(
        cls,
        result: InferenceResult,
    ) -> bool:
        """
        Verifica se o resultado é elegível para ML2.
        """

        return cls.determine_ml2_label(result) is not None

    def create_ml2_window(
        self,
        result: InferenceResult,
        samples: list[Sample],
        window_id: str,
    ) -> Optional[ML2DatasetWindow]:
        """
        Cria um registro de janela ML2 quando elegível.

        A função não constrói a janela.

        A janela deve ser fornecida pelo componente responsável
        pela construção temporal.
        """

        label = self.determine_ml2_label(result)

        if label is None:
            return None

        if not samples:
            return None

        if result.window_start is None or result.window_end is None:
            return None

        return ML2DatasetWindow.from_window(
            window_id=window_id,
            window_start=result.window_start,
            window_end=result.window_end,
            samples=samples,
            label=label,
        )

    # ==========================================================
    # Armazenamento local temporário
    # ==========================================================

    def register_ml1(
        self,
        record: ML1DatasetRecord,
    ) -> None:
        """
        Registra um novo registro ML1.
        """

        self._ml1_records.append(record)

    def register_ml2(
        self,
        record: ML2DatasetWindow,
    ) -> None:
        """
        Registra uma nova janela ML2.
        """

        self._ml2_records.append(record)

    # ==========================================================
    # Consulta
    # ==========================================================

    def get_ml1_records(self) -> list[ML1DatasetRecord]:
        """
        Retorna os registros ML1 acumulados.
        """

        return list(self._ml1_records)

    def get_ml2_records(self) -> list[ML2DatasetWindow]:
        """
        Retorna as janelas ML2 acumuladas.
        """

        return list(self._ml2_records)

    def clear(self) -> None:
        """
        Limpa os registros acumulados.
        """

        self._ml1_records.clear()
        self._ml2_records.clear()

        self._ml2_normal_window_builder.clear()

        self._psc_window_count = 0
