"""
Wake Word Trainer
Entrena modelos de wake word personalizados usando OpenWakeWord.
"""

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WakeWordTrainer:
    """
    Entrena modelos de wake word personalizados.

    Usa OpenWakeWord para entrenar un modelo ONNX ligero
    que puede detectar una palabra de activación personalizada.

    Requisitos:
        - openwakeword instalado
        - Muestras positivas (diciendo el wake word) - mínimo 30
        - Muestras negativas (otras frases) - mínimo 30
    """

    def __init__(
        self,
        data_dir: str = "./data/wakeword_training",
        models_dir: str = "./models/wakeword",
    ):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        wake_word_name: str,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        augment_data: bool = True,
        validation_split: float = 0.2,
    ) -> Optional[str]:
        """
        Entrenar modelo de wake word.

        Args:
            wake_word_name: Nombre del wake word (debe coincidir con el directorio de datos)
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
            learning_rate: Tasa de aprendizaje
            augment_data: Aplicar data augmentation
            validation_split: Porcentaje de datos para validación

        Returns:
            Path al modelo entrenado (.onnx) o None si falla
        """
        wake_word_name = wake_word_name.lower().replace(" ", "_")
        data_path = self.data_dir / wake_word_name

        if not data_path.exists():
            logger.error(f"No se encontraron datos de entrenamiento en: {data_path}")
            return None

        positive_dir = data_path / "positive"
        negative_dir = data_path / "negative"

        # Verificar muestras
        positive_count = len(list(positive_dir.glob("*.wav")))
        negative_count = len(list(negative_dir.glob("*.wav")))

        logger.info(f"Muestras positivas: {positive_count}")
        logger.info(f"Muestras negativas: {negative_count}")

        if positive_count < 30:
            logger.error(f"Se necesitan al menos 30 muestras positivas (tienes {positive_count})")
            return None

        if negative_count < 30:
            logger.error(f"Se necesitan al menos 30 muestras negativas (tienes {negative_count})")
            return None

        # Preparar configuración de entrenamiento
        output_path = self.models_dir / f"{wake_word_name}.onnx"

        logger.info(f"Iniciando entrenamiento de '{wake_word_name}'...")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")

        try:
            # Intentar usar el entrenamiento de OpenWakeWord
            model_path = self._train_openwakeword(
                wake_word_name=wake_word_name,
                positive_dir=positive_dir,
                negative_dir=negative_dir,
                output_path=output_path,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                augment_data=augment_data,
                validation_split=validation_split
            )

            if model_path and Path(model_path).exists():
                logger.info(f"✅ Modelo entrenado: {model_path}")

                # Guardar metadata
                self._save_metadata(
                    wake_word_name=wake_word_name,
                    model_path=model_path,
                    positive_samples=positive_count,
                    negative_samples=negative_count,
                    epochs=epochs
                )

                return str(model_path)

        except Exception as e:
            logger.error(f"Error en entrenamiento: {e}")

            # Fallback: crear modelo simple con método alternativo
            logger.info("Intentando método alternativo...")
            return self._train_simple(
                wake_word_name=wake_word_name,
                positive_dir=positive_dir,
                negative_dir=negative_dir,
                output_path=output_path
            )

        return None

    def _train_openwakeword(
        self,
        wake_word_name: str,
        positive_dir: Path,
        negative_dir: Path,
        output_path: Path,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        augment_data: bool,
        validation_split: float
    ) -> Optional[str]:
        """Entrenar usando la API de OpenWakeWord"""
        try:
            # Crear archivo de configuración temporal
            config = {
                "model_name": wake_word_name,
                "positive_clips": str(positive_dir),
                "negative_clips": str(negative_dir),
                "output_dir": str(output_path.parent),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "augment": augment_data,
                "validation_split": validation_split,
            }

            # OpenWakeWord training (si está disponible el script de entrenamiento)
            # Nota: OpenWakeWord puede requerir un proceso más complejo
            # Por ahora usamos un approach simplificado

            from openwakeword.utils import train_custom_model

            model_path = train_custom_model(
                model_name=wake_word_name,
                positive_clips_dir=str(positive_dir),
                negative_clips_dir=str(negative_dir),
                output_dir=str(self.models_dir),
                epochs=epochs
            )

            return model_path

        except ImportError:
            logger.warning("Función de entrenamiento de OpenWakeWord no disponible")
            return None
        except Exception as e:
            logger.error(f"Error en OpenWakeWord training: {e}")
            return None

    def _train_simple(
        self,
        wake_word_name: str,
        positive_dir: Path,
        negative_dir: Path,
        output_path: Path
    ) -> Optional[str]:
        """
        Entrenamiento simplificado usando embeddings de audio.
        Este es un fallback si OpenWakeWord training no está disponible.
        """
        try:
            import torch
            import torch.nn as nn
            import torchaudio
            from torch.utils.data import DataLoader, Dataset

            logger.info("Usando entrenamiento simplificado con PyTorch")

            # Dataset simple
            class WakeWordDataset(Dataset):
                def __init__(self, positive_dir, negative_dir, sample_rate=16000):
                    self.samples = []
                    self.labels = []
                    self.sample_rate = sample_rate
                    self.target_length = sample_rate * 2  # 2 segundos

                    # Cargar positivos
                    for wav_path in positive_dir.glob("*.wav"):
                        self.samples.append(str(wav_path))
                        self.labels.append(1)

                    # Cargar negativos
                    for wav_path in negative_dir.glob("*.wav"):
                        self.samples.append(str(wav_path))
                        self.labels.append(0)

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    waveform, sr = torchaudio.load(self.samples[idx])

                    # Resample si es necesario
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)

                    # Ajustar longitud
                    if waveform.shape[1] < self.target_length:
                        padding = self.target_length - waveform.shape[1]
                        waveform = torch.nn.functional.pad(waveform, (0, padding))
                    else:
                        waveform = waveform[:, :self.target_length]

                    # Extraer mel spectrogram
                    mel_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=self.sample_rate,
                        n_mels=40,
                        n_fft=400,
                        hop_length=160
                    )
                    mel = mel_transform(waveform)

                    return mel.squeeze(0), self.labels[idx]

            # Modelo simple
            class SimpleWakeWordModel(nn.Module):
                def __init__(self, input_size=40, hidden_size=64):
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(2),
                        nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool1d(1)
                    )
                    self.fc = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    x = self.conv(x)
                    x = x.squeeze(-1)
                    return torch.sigmoid(self.fc(x))

            # Entrenar
            dataset = WakeWordDataset(positive_dir, negative_dir)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

            model = SimpleWakeWordModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCELoss()

            logger.info("Entrenando modelo simple...")
            for epoch in range(50):
                total_loss = 0
                for mel, label in dataloader:
                    optimizer.zero_grad()
                    output = model(mel).squeeze()
                    loss = criterion(output, label.float())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/50, Loss: {total_loss/len(dataloader):.4f}")

            # Exportar a ONNX
            dummy_input = torch.randn(1, 40, 200)
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch', 2: 'time'}}
            )

            logger.info(f"Modelo exportado a: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error en entrenamiento simple: {e}")
            return None

    def _save_metadata(
        self,
        wake_word_name: str,
        model_path: str,
        positive_samples: int,
        negative_samples: int,
        epochs: int
    ):
        """Guardar metadata del modelo entrenado"""
        metadata = {
            "wake_word": wake_word_name,
            "model_path": model_path,
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "epochs": epochs,
            "threshold_recommended": 0.5,
        }

        metadata_path = Path(model_path).with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def list_trained_models(self) -> list[dict]:
        """Listar modelos entrenados disponibles"""
        models = []

        for onnx_file in self.models_dir.glob("*.onnx"):
            metadata_file = onnx_file.with_suffix(".json")

            model_info = {
                "name": onnx_file.stem,
                "path": str(onnx_file),
                "size_kb": onnx_file.stat().st_size / 1024
            }

            if metadata_file.exists():
                with open(metadata_file) as f:
                    model_info.update(json.load(f))

            models.append(model_info)

        return models
