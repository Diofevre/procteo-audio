"""
Gestionnaire de tokens HuggingFace pour l'authentification Pyannote.
"""

import os
import streamlit as st
from typing import Dict, Any, Optional
from loguru import logger


class TokenManager:
    """
    Gestionnaire centralisé des tokens HuggingFace.
    """
    
    def __init__(self):
        """Initialise le gestionnaire de tokens."""
        self.token = None
        self.token_source = None
        self._load_token()
    
    def _load_token(self) -> None:
        """Charge le token depuis les sources disponibles."""
        # Ordre de priorité: 1. Streamlit secrets, 2. Variables d'environnement
        if hasattr(st, 'secrets') and st.secrets.get("HF_TOKEN"):
            self.token = st.secrets["HF_TOKEN"]
            self.token_source = "streamlit_secrets"
            logger.info("Token HF chargé depuis Streamlit secrets")
        elif os.environ.get("HUGGINGFACE_HUB_TOKEN"):
            self.token = os.environ["HUGGINGFACE_HUB_TOKEN"]
            self.token_source = "environment"
            logger.info("Token HF chargé depuis variables d'environnement")
        else:
            self.token = None
            self.token_source = None
            logger.warning("Aucun token HF trouvé")
    
    def get_token(self) -> Optional[str]:
        """Retourne le token HF chargé."""
        return self.token
    
    def get_token_status(self) -> Dict[str, Any]:
        """Retourne le statut du token."""
        return {
            "available": bool(self.token),
            "source": self.token_source,
            "length": len(self.token) if self.token else 0
        }
    
    def check_model_access(self, model_id: str, token: str) -> Dict[str, Any]:
        """
        Vérifie l'accès au modèle en tentant de télécharger sa configuration.
        
        Args:
            model_id: ID du modèle Pyannote
            token: Token HF à vérifier
            
        Returns:
            Dict avec statut et détails de l'accès
        """
        try:
            from huggingface_hub import hf_hub_download
            
            # Tentative de téléchargement de la config
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.yaml",
                token=token,
                local_files_only=False
            )
            
            return {
                "success": True,
                "message": "Accès au modèle confirmé",
                "config_path": config_path
            }
            
        except Exception as e:
            error_msg = str(e)
            
            if "401" in error_msg or "Unauthorized" in error_msg:
                return {
                    "success": False,
                    "message": "Token invalide ou expiré",
                    "details": "Vérifiez que votre token HF est valide et non expiré"
                }
            elif "403" in error_msg or "Forbidden" in error_msg:
                return {
                    "success": False,
                    "message": "Conditions d'utilisation non acceptées",
                    "details": f"Vous devez accepter les conditions d'utilisation pour {model_id} sur HuggingFace"
                }
            elif "404" in error_msg or "Not Found" in error_msg:
                return {
                    "success": False,
                    "message": "Modèle non trouvé",
                    "details": f"Le modèle {model_id} n'existe pas ou n'est pas accessible"
                }
            else:
                return {
                    "success": False,
                    "message": f"Erreur d'accès: {error_msg}",
                    "details": "Vérifiez votre connexion réseau et les permissions"
                }
    
    @staticmethod
    def get_torch_device():
        """
        Détermine le meilleur device Torch disponible.
        
        Returns:
            torch.device object
        """
        try:
            import torch
            
            # Priorité: MPS (Mac M1/M2), CUDA, CPU
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
                
        except ImportError:
            return torch.device("cpu")
        except Exception as e:
            logger.warning(f"Erreur détection device: {e}, fallback CPU")
            return torch.device("cpu")
    
    def refresh_token(self) -> bool:
        """Rafraîchit le token depuis les sources."""
        old_token = self.token
        self._load_token()
        return self.token != old_token
