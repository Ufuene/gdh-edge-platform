"""
test_real_inference_pipeline.py

Testes de integração do pipeline real de inferência do GDH Edge.

Fluxos validados:

    Teste 1:
    Sample
        ↓
    SampleBuffer
        ↓
    ML1Service + modelo real
        ↓
    DeviationDetector
        ↓
    WindowManager
        ↓
    ML2Service + modelo real
        ↓
    InferenceResult


    Teste 2:
    Dataset PSC operacional — 100 amostras
        ↓
    processamento sequencial
        ↓
    ML1 real
        ↓
    desvio operacional real
        ↓
    WindowManager
        ↓
    janela 3 + 1 + 4
        ↓
    ML2 real
        ↓
    diagnóstico

Este arquivo NÃO altera os testes unitários existentes.
"""

import csv
from datetime import datetime
from pathlib import Path

import joblib

from app.data_manager.sample import Sample
from app.data_manager.sample_buffer import SampleBuffer
from app.data_manager.window_manager import WindowManager
from app.inference.deviation_detector import DeviationDetector
from app.inference.inference_orchestrator import InferenceOrchestrator
from app.inference.inference_result import InferenceResult
from app.inference.ml1_service import ML1Service
from app.inference.ml2_service import ML2Service

PROJECT_ROOT = Path("/home/ufuene/gdh-platform")

ML1_MODEL_PATH = PROJECT_ROOT / "models/ml1/RandomForest_ML1_CPU.pkl"
ML2_MODEL_PATH = PROJECT_ROOT / "models/ml2/RandomForest_ML2_cpu.pkl"

OPERATIONAL_PSC_DATASET = (
    PROJECT_ROOT / "datasets/ml2/test/" / "Dataset_PSC_Teste_20_30s_100amostras.csv"
)

POWER_THRESHOLD = 5.0


def create_sample(
    irradiance: float,
    temperature: float,
    v_pv: float,
    i_pv: float,
    v_out: float,
    i_out: float,
    i_bat: float,
    i_load: float,
    p_out: float,
) -> Sample:
    """
    Cria uma Sample para o teste do pipeline real.
    """

    return Sample(
        timestamp=datetime.now(),
        irradiance=irradiance,
        temperature=temperature,
        v_pv=v_pv,
        i_pv=i_pv,
        v_out=v_out,
        i_out=i_out,
        i_bat=i_bat,
        i_load=i_load,
        p_out=p_out,
    )


def load_real_services():
    """
    Carrega os dois modelos reais e cria os serviços ML1 e ML2.
    """

    assert ML1_MODEL_PATH.exists(), f"ML1 model not found: {ML1_MODEL_PATH}"

    assert ML2_MODEL_PATH.exists(), f"ML2 model not found: {ML2_MODEL_PATH}"

    ml1_model = joblib.load(ML1_MODEL_PATH)
    ml2_model = joblib.load(ML2_MODEL_PATH)

    assert ml1_model.n_features_in_ == 3

    assert ml2_model.n_features_in_ == 72
    assert list(ml2_model.classes_) == [0, 1]

    ml1_service = ML1Service(ml1_model)
    ml2_service = ML2Service(ml2_model)

    return ml1_service, ml2_service


def create_pipeline():
    """
    Cria o pipeline completo utilizando os modelos reais.
    """

    ml1_service, ml2_service = load_real_services()

    sample_buffer = SampleBuffer()

    window_manager = WindowManager(
        sample_buffer,
        previous_samples=3,
        future_samples=4,
    )

    deviation_detector = DeviationDetector(
        power_threshold=POWER_THRESHOLD,
    )

    orchestrator = InferenceOrchestrator(
        sample_buffer=sample_buffer,
        ml1_service=ml1_service,
        ml2_service=ml2_service,
        deviation_detector=deviation_detector,
        window_manager=window_manager,
    )

    return (
        orchestrator,
        sample_buffer,
        window_manager,
        ml1_service,
        ml2_service,
    )


def test_real_inference_pipeline():
    """
    Executa o pipeline completo usando os dois modelos reais.

    O teste:

    1. cria oito amostras;
    2. processa cada uma pelo ML1 real;
    3. cria intencionalmente um desvio na quarta amostra;
    4. aguarda as quatro amostras futuras;
    5. permite ao WindowManager construir a janela;
    6. executa o ML2 real;
    7. verifica o InferenceResult final.
    """

    (
        orchestrator,
        sample_buffer,
        window_manager,
        _,
        _,
    ) = create_pipeline()

    results = []

    # ==========================================================
    # Perfil base utilizado para as amostras
    # ==========================================================

    base = {
        "irradiance": 62.2,
        "temperature": 10.86175,
        "v_pv": 23.88,
        "i_pv": 0.2615,
        "v_out": 12.0347,
        "i_out": 1.0,
        "i_bat": -13.5,
        "i_load": 14.5,
    }

    # ==========================================================
    # Gerar as 8 amostras
    #
    # A amostra 3 (quarta posição) será utilizada como
    # ponto de desvio.
    # ==========================================================

    for index in range(8):
        p_out = 15.0

        if index == 3:
            p_out = 0.0

        sample = create_sample(
            irradiance=base["irradiance"],
            temperature=base["temperature"],
            v_pv=base["v_pv"],
            i_pv=base["i_pv"],
            v_out=base["v_out"],
            i_out=base["i_out"],
            i_bat=base["i_bat"],
            i_load=base["i_load"],
            p_out=p_out,
        )

        result = orchestrator.process_sample(
            sample=sample,
            sample_index=index,
        )

        results.append(result)

    # ==========================================================
    # Verificações do buffer
    # ==========================================================

    assert sample_buffer.size() == 8

    # ==========================================================
    # Deve existir uma janela completa
    # ==========================================================

    assert window_manager.has_complete_window()

    window = window_manager.get_current_window()

    assert window is not None
    assert len(window) == 8

    # ==========================================================
    # O evento deve ter sido identificado na quarta amostra
    # ==========================================================

    assert results[3].deviation_detected is True

    # ==========================================================
    # A primeira vez que a janela fica completa é na amostra 7
    # ==========================================================

    final_result = results[7]

    assert isinstance(final_result, InferenceResult)

    assert final_result.diagnosis in (0, 1)

    # ==========================================================
    # O ML1 deve ter produzido uma potência de referência
    # válida em todas as amostras.
    # ==========================================================

    for result in results:
        assert isinstance(result.predicted_power, float)
        assert result.predicted_power >= 0.0

    # ==========================================================
    # A janela deve conter a amostra de desvio na posição 4.
    # ==========================================================

    assert len(window) == 8
    assert window[3].p_out == 0.0


def test_real_pipeline_with_operational_psc_dataset():
    """
    Executa o pipeline real sobre o dataset operacional PSC
    de 100 amostras.

    O dataset contém somente as variáveis de aquisição:

        Irradiancia
        Temperatura
        Vout
        Iout
        Ipv
        Vpv
        Iload
        Pout
        Ibat

    O Pref é produzido pelo ML1 em tempo de execução.

    O primeiro desvio previamente identificado no Raspberry
    ocorre no índice 5.

    A janela operacional esperada é:

        2, 3, 4, 5, 6, 7, 8, 9

    correspondendo a:

        3 anteriores
        +
        amostra do desvio
        +
        4 posteriores

    A validação da janela é realizada no instante em que
    ela é construída. O teste não depende de a janela
    permanecer armazenada após o encerramento do evento.
    """

    # ==========================================================
    # Validar existência do dataset
    # ==========================================================

    assert OPERATIONAL_PSC_DATASET.exists(), (
        "Operational PSC dataset not found: " f"{OPERATIONAL_PSC_DATASET}"
    )

    # ==========================================================
    # Criar pipeline real
    # ==========================================================

    (
        orchestrator,
        sample_buffer,
        window_manager,
        ml1_service,
        ml2_service,
    ) = create_pipeline()

    # ==========================================================
    # Ler CSV
    # ==========================================================

    with OPERATIONAL_PSC_DATASET.open(
        "r",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        expected_columns = [
            "Irradiancia",
            "Temperatura",
            "Vout",
            "Iout",
            "Ipv",
            "Vpv",
            "Iload",
            "Pout",
            "Ibat",
        ]

        assert reader.fieldnames == expected_columns

        rows = list(reader)

    # ==========================================================
    # Validar dataset
    # ==========================================================

    assert len(rows) == 100

    # ==========================================================
    # Variáveis para capturar o evento operacional no instante
    # em que a janela for construída.
    # ==========================================================

    operational_window = None
    operational_diagnosis = None

    # ==========================================================
    # Processar as 100 amostras sequencialmente
    # ==========================================================

    results = []

    for index, row in enumerate(rows):

        sample = create_sample(
            irradiance=float(row["Irradiancia"]),
            temperature=float(row["Temperatura"]),
            v_pv=float(row["Vpv"]),
            i_pv=float(row["Ipv"]),
            v_out=float(row["Vout"]),
            i_out=float(row["Iout"]),
            i_bat=float(row["Ibat"]),
            i_load=float(row["Iload"]),
            p_out=float(row["Pout"]),
        )

        result = orchestrator.process_sample(
            sample=sample,
            sample_index=index,
        )

        results.append(result)

        # ======================================================
        # Capturar a janela exatamente no momento em que
        # o evento operacional esperado é classificado.
        #
        # O desvio está no índice 5, portanto a janela esperada
        # contém os índices:
        #
        # 2, 3, 4, 5, 6, 7, 8, 9
        #
        # A posição central [3] corresponde ao índice 5.
        # ======================================================

        if window_manager.has_complete_window():

            current_window = window_manager.get_current_window()

            if (
                current_window is not None
                and len(current_window) == 8
                and current_window[3].p_out == float(rows[5]["Pout"])
            ):

                operational_window = list(current_window)
                operational_diagnosis = result.diagnosis

    # ==========================================================
    # O buffer deve conter as 100 amostras processadas.
    # ==========================================================

    assert sample_buffer.size() == 100

    # ==========================================================
    # O primeiro desvio identificado pelo ML1 deve ser o
    # índice 0 ou 2 antes do evento operacional escolhido.
    #
    # O primeiro candidato com contexto suficiente é o índice 5.
    # ==========================================================

    assert results[5].deviation_detected is True

    assert results[5].predicted_power == (
        ml1_service.predict(
            irradiance=float(rows[5]["Irradiancia"]),
            temperature=float(rows[5]["Temperatura"]),
            v_out=float(rows[5]["Vout"]),
        )
    )

    # ==========================================================
    # Deve ter sido capturada a janela operacional.
    #
    # Importante:
    #
    # não verificamos o estado final do WindowManager,
    # porque o evento pode ter sido encerrado posteriormente
    # pela histerese.
    # ==========================================================

    assert operational_window is not None
    assert len(operational_window) == 8

    # ==========================================================
    # O diagnóstico produzido no instante da classificação
    # deve ser PSC.
    # ==========================================================

    assert operational_diagnosis == 1

    # ==========================================================
    # A janela deve corresponder exatamente aos índices:
    #
    # 2, 3, 4, 5, 6, 7, 8, 9
    # ==========================================================

    expected_window_indices = list(range(2, 10))

    for sample, expected_index in zip(
        operational_window,
        expected_window_indices,
    ):

        expected_row = rows[expected_index]

        assert sample.irradiance == float(expected_row["Irradiancia"])

        assert sample.temperature == float(expected_row["Temperatura"])

        assert sample.v_out == float(expected_row["Vout"])

        assert sample.i_out == float(expected_row["Iout"])

        assert sample.i_pv == float(expected_row["Ipv"])

        assert sample.v_pv == float(expected_row["Vpv"])

        assert sample.i_load == float(expected_row["Iload"])

        assert sample.p_out == float(expected_row["Pout"])

        assert sample.i_bat == float(expected_row["Ibat"])

    # ==========================================================
    # A amostra de desvio deve estar na posição central:
    #
    # t4 da janela = índice 5 do dataset.
    # ==========================================================

    deviation_sample = operational_window[3]

    assert deviation_sample.p_out == float(rows[5]["Pout"])

    # ==========================================================
    # O resultado da amostra de desvio deve indicar desvio.
    # ==========================================================

    assert results[5].deviation_detected is True
    assert results[5].diagnosis is None

    # ==========================================================
    # O ML2 deve ter sido executado na amostra 9.
    #
    # O índice 5 é o desvio:
    #
    # 2, 3, 4, [5], 6, 7, 8, 9
    #
    # Portanto, são necessárias quatro amostras futuras:
    # 6, 7, 8 e 9.
    # ==========================================================

    assert results[9].diagnosis == 1

    assert isinstance(
        results[9],
        InferenceResult,
    )

    # ==========================================================
    # O diagnóstico capturado da janela deve coincidir com
    # o diagnóstico produzido no momento da classificação.
    # ==========================================================

    assert operational_diagnosis == results[9].diagnosis

    # ==========================================================
    # Verificação independente do ML2 usando exatamente
    # a janela operacional capturada.
    #
    # Não altera o fluxo do Orchestrator; apenas confirma
    # que o mesmo ML2 reconhece essa mesma janela.
    # ==========================================================

    direct_diagnosis = ml2_service.predict(operational_window)

    assert direct_diagnosis == operational_diagnosis

    # ==========================================================
    # Informações úteis no output do teste.
    # ==========================================================

    print()
    print("=" * 80)
    print("REAL OPERATIONAL PSC PIPELINE")
    print("=" * 80)
    print(f"Dataset samples      : {len(rows)}")
    print("First operational candidate index : 5")
    print("Operational window  : [2, 3, 4, 5, 6, 7, 8, 9]")
    print(f"ML2 diagnosis       : {operational_diagnosis}")
    print("=" * 80)
