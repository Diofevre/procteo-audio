"""
Processeur VAD utilisant Pyannote avec post-processing et gestion d'erreurs robuste.
"""

import os
import hashlib
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import time

import torch
import torchaudio
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import yaml
from loguru import logger

from ..utils.token_manager import TokenManager


class VADProcessor:
    """
    Processeur VAD utilisant Pyannote avec gestion robuste des erreurs.
    """
    
    def __init__(self, config_path: str = "config/vad_config.yaml"):
        """
        Initialise le processeur VAD.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
        self.model_id = self.config.get("model", "pyannote/voice-activity-detection")
        
        # Gestion des tokens
        self.token_manager = TokenManager()
        self.token = self.token_manager.get_token()
        
        # Pipeline et état
        self.pipeline = None
        self.device = TokenManager.get_torch_device()
        self.device_str = str(self.device)
        self.last_processed_hash = None
        self.last_result = None
        self.initialization_error = None
        
        logger.info(f"VADProcessor initialisé sur device: {self.device_str}")

    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration depuis un fichier YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration chargée depuis {config_path}")
            return config
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            # Configuration par défaut
            return {
                "model": "pyannote/voice-activity-detection",
                "confidence_threshold": 0.3,
                "min_duration_s": 0.3,
                "merge_gap_s": 0.2,
                "margin_s": 0.25,
                "onset": 0.30,
                "offset": 0.25,
                "min_duration_on": 0.20,
                "min_duration_off": 0.10
            }

    def initialize_pipeline(self, hf_token: Union[str, None] = None) -> dict:
        """
        Initialise la pipeline VAD avec gestion d'erreurs détaillée.
        
        Args:
            hf_token: Token HuggingFace (optionnel, sinon utilise TokenManager)
            
        Returns:
            Dict avec statut et détails de l'initialisation
        """
        try:
            # Utiliser le token fourni ou celui du TokenManager
            token = hf_token or self.token
            if not token:
                return {
                    "success": False,
                    "message": "Token HuggingFace manquant",
                    "details": "Aucun token trouvé dans .streamlit/secrets.toml ou variables d'environnement"
                }

            # Vérifier l'accès au modèle
            access_status = self.token_manager.check_model_access(self.model_id, token)
            if not access_status["success"]:
                return {
                    "success": False,
                    "message": f"Échec de la vérification d'accès au modèle: {access_status['message']}",
                    "details": access_status["details"],
                    "access_status": access_status
                }

            logger.info(f"Initialisation de la pipeline {self.model_id} sur {self.device_str}")
            
            # Initialisation avec paramètres de binarisation
            self.pipeline = Pipeline.from_pretrained(
                self.model_id,
                use_auth_token=token,
                
                
                
                
            )
            
            # Déplacement sur le device
            self.pipeline = self.pipeline.to(self.device)
            
            logger.info(f"Pipeline VAD initialisée avec succès sur {self.device_str}")
            self.initialization_error = None
            
            return {
                "success": True,
                "message": "Pipeline VAD initialisée avec succès.",
                "access_status": access_status
            }

        except Exception as e:
            error_msg = f"Erreur initialisation pipeline: {str(e)}"
            logger.error(error_msg)
            self.initialization_error = error_msg
            return {
                "success": False,
                "message": "Échec de l'initialisation de la pipeline",
                "details": str(e),
                "error_type": type(e).__name__
            }

    def _extract_audio_from_video(self, video_path: str) -> str:
        """Extrait l'audio d'une vidéo en WAV mono 16kHz."""
        try:
            # Créer le dossier temporaire
            os.makedirs(".tmp", exist_ok=True)
            
            # Nom du fichier de sortie
            video_name = Path(video_path).stem
            audio_path = f".tmp/{video_name}.wav"
            
            # Extraction avec ffmpeg
            import subprocess
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vn",  # Pas de vidéo
                "-ac", "1",  # Mono
                "-ar", "16000",  # 16kHz
                "-acodec", "pcm_s16le",  # WAV 16-bit
                "-y",  # Écraser si existe
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"ffmpeg error: {result.stderr}")
            
            logger.info(f"Audio extrait: {video_path} → {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Erreur extraction audio: {e}")
            raise

    def _get_file_hash(self, file_path: str) -> str:
        """Calcule le hash SHA-256 d'un fichier."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _post_process_segments(self, segments: List, audio_duration: float) -> List:
        """
        Post-traitement des segments pyannote.
        
        Args:
            segments: Liste d'objets pyannote.core.Segment
            audio_duration: Durée totale de l'audio
            
        Returns:
            Liste de segments post-traités (objets pyannote)
        """
        try:
            processed_segments = []
            
            for segment in segments:
                start = segment.start
                end = segment.end
                
                # Filtrage par durée minimale
                duration = end - start
                if duration < self.config["min_duration_s"]:
                    continue
                
                # Application des marges
                start = max(0, start - self.config["margin_s"])
                end = min(audio_duration, end + self.config["margin_s"])
                
                # Créer un nouveau segment pyannote avec les marges appliquées
                from pyannote.core import Segment
                processed_segments.append(Segment(start, end))
            
            # Fusion des segments proches
            if processed_segments and self.config["merge_gap_s"] > 0:
                merged = []
                current = processed_segments[0]
                
                for next_seg in processed_segments[1:]:
                    if next_seg.start - current.end <= self.config["merge_gap_s"]:
                        # Fusion en créant un nouveau segment étendu
                        current = Segment(current.start, next_seg.end)
                    else:
                        # Nouveau segment
                        merged.append(current)
                        current = next_seg
                
                merged.append(current)
                processed_segments = merged
            
            logger.info(f"Post-traitement: {len(segments)} segments → {len(processed_segments)} segments")
            
            return processed_segments
            
        except Exception as e:
            logger.error(f"Erreur post-traitement: {e}")
            return []

    def process_audio(self, file_path: str) -> Dict[str, Any]:
        """
        Traite un fichier audio/vidéo avec VAD.
        
        Args:
            file_path: Chemin vers le fichier audio/vidéo
            
        Returns:
            Dict avec résultats et métadonnées
        """
        start_time = time.time()
        
        try:
            # Vérification de l'initialisation
            if not self.pipeline:
                if not self.initialize_pipeline():
                    return {
                        "success": False,
                        "error": self.initialization_error or "Pipeline non initialisée",
                        "health": self.check_prerequisites()
                    }
            
            # Détermination du type de fichier
            file_ext = Path(file_path).suffix.lower()
            is_video = file_ext in ['.mp4', '.mkv', '.avi', '.mov']
            
            # Extraction audio si vidéo
            if is_video:
                audio_path = self._extract_audio_from_video(file_path)
                input_type = "video"
            else:
                audio_path = file_path
                input_type = "audio"
            
            # Calcul du hash du fichier
            file_hash = self._get_file_hash(file_path)
            
            # Lecture de l'audio
            waveform, sample_rate = sf.read(audio_path)
            audio_duration = len(waveform) / sample_rate
            
            logger.info(f"Traitement {input_type}: {file_path} ({audio_duration:.2f}s)")
            
            # Inférence VAD avec paramètres de binarisation
            vad_result = self.pipeline(audio_path)
            
            # Log de la sortie brute pyannote pour debug
            logger.info(f"Sortie pyannote: {type(vad_result)}")
            
            # Extraction des segments (garder les objets pyannote)
            segments = []
            if hasattr(vad_result, "get_timeline"):
                # Format Annotation standard
                timeline = vad_result.get_timeline()
                logger.info(f"Timeline pyannote: {len(timeline)} segments")
                for i, segment in enumerate(timeline[:5]):  # Log des 5 premiers
                    logger.info(f"Segment {i}: {segment.start:.2f}s - {segment.end:.2f}s")
                segments = list(timeline)  # Garder les objets pyannote
            elif isinstance(vad_result, dict) and "speech" in vad_result:
                # Format dict avec clé "speech"
                timeline = vad_result["speech"].get_timeline()
                logger.info(f"Timeline speech: {len(timeline)} segments")
                segments = list(timeline)
            else:
                logger.warning(f"Format de sortie pyannote inattendu: {type(vad_result)}")
                segments = []
            
            # Post-traitement sur objets pyannote
            processed_segments = self._post_process_segments(segments, audio_duration)
            
            # Conversion en dict seulement à la fin
            final_segments = []
            for segment in processed_segments:
                final_segments.append({
                    "type": "speech",
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "confidence": None,
                    "duration": round(segment.end - segment.start, 3)
                })
            
            # Calcul des métriques
            processing_time = time.time() - start_time
            rtf = processing_time / audio_duration if audio_duration > 0 else 0
            
            logger.info(f"VAD terminé: {len(final_segments)} segments, RTF: {rtf:.3f}")
            
            # Nettoyage des fichiers temporaires
            if is_video and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Fichier temporaire supprimé: {audio_path}")
            
            # Construction du résultat final
            results = {
                "success": True,
                "metadata": {
                    "source": file_path,
                    "input_type": input_type,
                    "duration_s": round(audio_duration, 3),
                    "profile": "vad",
                    "model": self.model_id,
                    "device": self.device_str,
                    "processing_time_s": round(processing_time, 3),
                    "rtf": round(rtf, 3),
                    "segments_count": len(final_segments),
                    "config": self.config
                },
                "events": final_segments
            }
            
            # Sauvegarde des résultats
            self.save_results(results, file_path)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur traitement VAD: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    def check_prerequisites(self) -> Dict[str, Any]:
        """Vérifie les prérequis du système."""
        try:
            # Vérification des dépendances
            import torch
            import torchaudio
            import soundfile
            import yaml
            
            # Vérification de ffmpeg pour l'extraction vidéo
            import subprocess
            try:
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
                ffmpeg_available = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                ffmpeg_available = False
            
            return {
                "torch_version": torch.__version__,
                "torchaudio_version": torchaudio.__version__,
                "device": self.device_str,
                "ffmpeg_available": ffmpeg_available,
                "config_loaded": bool(self.config),
                "token_available": bool(self.token)
            }
            
        except ImportError as e:
            return {"error": f"Import manquant: {e}"}
        except Exception as e:
            return {"error": f"Erreur vérification: {e}"}

    def save_results(self, results: Dict[str, Any], source_file: str) -> str:
        """
        Sauvegarde les résultats dans un fichier JSON.
        
        Args:
            results: Résultats à sauvegarder
            source_file: Fichier source analysé
            
        Returns:
            Chemin du fichier de résultats
        """
        try:
            # Créer le dossier reports s'il n'existe pas
            os.makedirs("reports", exist_ok=True)
            
            # Nom du fichier de sortie
            source_name = Path(source_file).stem
            timestamp = int(time.time())
            output_file = f"reports/{source_name}_vad_{timestamp}.json"
            
            # Sauvegarde
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Résultats sauvegardés: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
            return ""

    def get_health_status(self) -> Dict[str, Any]:
        """Retourne le statut de santé complet du processeur VAD."""
        return {
            "initialized": bool(self.pipeline),
            "device": self.device_str,
            "model": self.model_id,
            "config": self.config,
            "token_status": self.token_manager.get_token_status(),
            "last_error": self.initialization_error,
            "prerequisites": self.check_prerequisites()
        }
