"""
main.py

Ponto de entrada da plataforma GDH Edge.

Arquitetura:

    ESP32
       |
       | MQTT
       v
    Mosquitto
       |
       v
    MQTTSubscriber
       |
       v
    PVDataMessage
       |
       v
    InferenceRuntime
       |
       v
    InferenceOrchestrator
       |
       +---- ML1
       |
       +---- DeviationDetector
       |
       +---- WindowManager
       |
       +---- ML2
       |
       v
    InferenceResult

O main.py é responsável somente pela composição
dos componentes da aplicação.
"""

from pathlib import Path

import joblib

from app.communication.mqtt_subscriber import (
    MQTTSubscriber,
    PVDataMessage,
)

from app.data_manager.sample_buffer import SampleBuffer
from app.data_manager.window_manager import WindowManager

from app.inference.deviation_detector import DeviationDetector
from app.inference.inference_orchestrator import InferenceOrchestrator
from app.inference.inference_runtime import InferenceRuntime
from app.inference.ml1_service import ML1Service
from app.inference.ml2_service import ML2Service

# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path("/home/ufuene/gdh-platform")

ML1_MODEL_PATH = PROJECT_ROOT / "models/ml1/RandomForest_ML1_CPU.pkl"

ML2_MODEL_PATH = PROJECT_ROOT / "models/ml2/RandomForest_ML2_cpu.pkl"

POWER_THRESHOLD = 5.0


# ============================================================
# CARREGAMENTO DOS MODELOS
# ============================================================


def load_models():
    """
    Carrega os modelos reais de ML1 e ML2.

    Os caminhos são os mesmos utilizados
    pelo teste do pipeline real.
    """

    print()
    print("=" * 60)
    print("CARREGANDO MODELOS")
    print("=" * 60)

    print(f"ML1 model   : {ML1_MODEL_PATH}")
    print(f"ML2 model   : {ML2_MODEL_PATH}")

    if not ML1_MODEL_PATH.exists():

        raise FileNotFoundError(f"ML1 model not found: {ML1_MODEL_PATH}")

    if not ML2_MODEL_PATH.exists():

        raise FileNotFoundError(f"ML2 model not found: {ML2_MODEL_PATH}")

    print()
    print("Carregando ML1...")

    ml1_model = joblib.load(ML1_MODEL_PATH)

    print("ML1 carregado.")

    print()
    print("Carregando ML2...")

    ml2_model = joblib.load(ML2_MODEL_PATH)

    print("ML2 carregado.")

    # --------------------------------------------------------
    # Validar identidade dos modelos
    # --------------------------------------------------------

    if ml1_model.n_features_in_ != 3:

        raise ValueError("ML1 deve possuir exatamente 3 features.")

    if ml2_model.n_features_in_ != 72:

        raise ValueError("ML2 deve possuir exatamente 72 features.")

    if list(ml2_model.classes_) != [0, 1]:

        raise ValueError("ML2 deve possuir classes [0, 1].")

    print()
    print("IDENTIDADE DOS MODELOS VALIDADA")

    print(f"ML1 features : " f"{ml1_model.n_features_in_}")

    print(f"ML2 features : " f"{ml2_model.n_features_in_}")

    print(f"ML2 classes  : " f"{list(ml2_model.classes_)}")

    return ml1_model, ml2_model


# ============================================================
# CONSTRUÇÃO DO PIPELINE
# ============================================================


def create_pipeline():
    """
    Cria todos os componentes do pipeline de inferência.
    """

    # ========================================================
    # MODELOS
    # ========================================================

    ml1_model, ml2_model = load_models()

    # ========================================================
    # SERVICES
    # ========================================================

    ml1_service = ML1Service(ml1_model)

    ml2_service = ML2Service(ml2_model)

    # ========================================================
    # SAMPLE BUFFER
    # ========================================================

    sample_buffer = SampleBuffer()

    # ========================================================
    # WINDOW MANAGER
    # ========================================================

    window_manager = WindowManager(
        sample_buffer,
        previous_samples=3,
        future_samples=4,
    )

    # ========================================================
    # DEVIATION DETECTOR
    # ========================================================

    deviation_detector = DeviationDetector(
        power_threshold=POWER_THRESHOLD,
    )

    # ========================================================
    # INFERENCE ORCHESTRATOR
    # ========================================================

    orchestrator = InferenceOrchestrator(
        sample_buffer=sample_buffer,
        ml1_service=ml1_service,
        ml2_service=ml2_service,
        deviation_detector=deviation_detector,
        window_manager=window_manager,
    )

    # ========================================================
    # INFERENCE RUNTIME
    # ========================================================

    runtime = InferenceRuntime(orchestrator)

    return runtime


# ============================================================
# HANDLER MQTT → INFERENCE
# ============================================================


def handle_message(
    message: PVDataMessage,
    subscriber_index: int,
):
    """
    Recebe uma mensagem validada pelo MQTTSubscriber
    e encaminha a Sample para o InferenceRuntime.

    O subscriber_index é mantido apenas como informação
    da camada de comunicação.
    """

    print()
    print("=" * 60)
    print("INTEGRACAO MQTT -> INFERENCE")
    print("=" * 60)

    print(f"Device ID       : " f"{message.device_id}")

    print(f"Sequence        : " f"{message.sequence}")

    print(f"Subscriber index: " f"{subscriber_index}")

    result = runtime.process_message(message)

    print()
    print("RESULTADO DA INFERENCIA")
    print("-" * 60)

    print(f"Pref            : " f"{result.predicted_power:.4f}")

    print(f"Deviation       : " f"{result.deviation:.4f}")

    print(f"Deviation       : " f"{result.deviation_detected}")

    print(f"Diagnosis       : " f"{result.diagnosis}")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================


def main():

    global runtime

    print()
    print("=" * 60)
    print("GDH EDGE PLATFORM")
    print("=" * 60)

    print()
    print("Inicializando pipeline de inferência...")

    runtime = create_pipeline()

    print()
    print("Pipeline de inferência inicializado.")

    print()
    print("Inicializando MQTT Subscriber...")

    subscriber = MQTTSubscriber(message_handler=handle_message)

    subscriber.connect()

    print()
    print("Sistema pronto.")
    print()
    print("Fluxo ativo:")
    print()
    print("ESP32")
    print("  ↓")
    print("Mosquitto")
    print("  ↓")
    print("MQTTSubscriber")
    print("  ↓")
    print("InferenceRuntime")
    print("  ↓")
    print("InferenceOrchestrator")
    print("  ↓")
    print("ML1 / WindowManager / ML2")
    print()
    print("Aguardando dados...")
    print()

    subscriber.loop_forever()


# ============================================================
# EXECUÇÃO
# ============================================================


if __name__ == "__main__":
    main()
