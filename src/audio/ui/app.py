"""
Application Streamlit VAD pour Procteo-Audio.
Interface pour l'analyse Voice Activity Detection avec Pyannote.
Interface complète avec diagnostic de santé et gestion d'erreurs robuste.
"""

import streamlit as st
import os
import platform
from pathlib import Path
import tempfile
import json
from loguru import logger

# Import des modules VAD
from src.audio.pipeline.vad_processor import VADProcessor
from src.audio.utils.token_manager import TokenManager

st.set_page_config(
    page_title="Procteo-Audio - VAD",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Procteo-Audio - Voice Activity Detection")
st.markdown("---")

# Initialisation de la session
if 'vad_processor' not in st.session_state:
    st.session_state.vad_processor = None

if 'health_status' not in st.session_state:
    st.session_state.health_status = None

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Vérification de la santé du système
    st.subheader("🔍 Santé du système")
    
    # Vérification complète de la santé
    if st.button("🔍 Vérifier la santé du système", type="primary"):
        # Créer un statut de santé local
        token_manager = TokenManager()
        token_status = token_manager.get_token_status()
        
        # Vérifier l'accès au modèle si on a un token
        model_access = None
        if token_status["available"]:
            model_access = token_manager.check_model_access(
                "pyannote/voice-activity-detection", 
                token_status.get("token", "")
            )
        
        st.session_state.health_status = {
            "token_available": token_status["available"],
            "token_source": token_status["source"],
            "model_accessible": model_access["success"] if model_access else False,
            "model_details": model_access.get("details", "") if model_access else "",
            "device": TokenManager.get_torch_device(),
            "timestamp": "maintenant"
        }
        st.rerun()
    
    if st.session_state.health_status:
        health = st.session_state.health_status
        
        # Statut du token
        if health["token_available"]:
            st.success(f"✅ Token HF détecté ({health['token_source']})")
        else:
            st.error("❌ Token HF manquant")
            st.info("Configurez HF_TOKEN dans .streamlit/secrets.toml")
        
        # Statut du modèle
        if health["model_accessible"]:
            st.success("✅ Modèle accessible")
        else:
            st.warning("⚠️ Modèle non accessible")
            if health["model_details"]:
                st.info(f"Détails: {health['model_details']}")
        
        # Device
        st.info(f"🖥️ Device: {health['device']}")
        
        st.markdown("---")
    
    # Initialisation du processeur VAD
    st.subheader("🚀 Initialisation VAD")
    
    if st.button("🔧 Initialiser le processeur VAD", type="secondary"):
        try:
            with st.spinner("Initialisation en cours..."):
                vad = VADProcessor()
                init_result = vad.initialize_pipeline()
                
                if init_result["success"]:
                    st.session_state.vad_processor = vad
                    st.success("✅ Processeur VAD initialisé avec succès!")
                    st.rerun()
                else:
                    st.error(f"❌ Échec de l'initialisation: {init_result['message']}")
                    if "details" in init_result:
                        st.error(f"Détails: {init_result['details']}")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
    
    # Statut du processeur
    if st.session_state.vad_processor:
        st.success("✅ Processeur VAD prêt")
        
        # Afficher les informations du processeur
        health_info = st.session_state.vad_processor.get_health_status()
        st.info(f"Device: {health_info['device']}")
        st.info(f"Modèle: {health_info['model']}")
        
        if st.button("🔄 Réinitialiser", type="secondary"):
            st.session_state.vad_processor = None
            st.rerun()

# Interface principale
st.header("🎯 Analyse VAD")

# Section d'upload
st.subheader("📁 Upload de fichier")
uploaded_file = st.file_uploader(
    "Choisissez un fichier audio ou vidéo",
    type=['wav', 'mp3', 'mp4', 'mkv', 'avi', 'mov'],
    help="Formats supportés: WAV, MP3, MP4, MKV, AVI, MOV"
)

# Section d'analyse
if uploaded_file is not None:
    st.subheader("🔍 Analyse")
    
    # Vérifier que le processeur est initialisé
    if not st.session_state.vad_processor:
        st.warning("⚠️ Le processeur VAD n'est pas initialisé. Utilisez le bouton dans la sidebar.")
    else:
        # Afficher les informations du fichier
        file_details = {
            "Nom": uploaded_file.name,
            "Type": uploaded_file.type,
            "Taille": f"{uploaded_file.size / 1024 / 1024:.2f} MB"
        }
        
        col1, col2, col3 = st.columns(3)
        for key, value in file_details.items():
            with col1 if key == "Nom" else col2 if key == "Type" else col3:
                st.metric(key, value)
        
        # Bouton d'analyse
        if st.button("🚀 Lancer l'analyse VAD", type="primary"):
            try:
                with st.spinner("Analyse en cours..."):
                    # Sauvegarder temporairement le fichier
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Lancer l'analyse
                    results = st.session_state.vad_processor.process_audio(tmp_path)
                    
                    # Nettoyer le fichier temporaire
                    os.unlink(tmp_path)
                    
                    if results["success"]:
                        st.success("✅ Analyse terminée avec succès!")
                        
                        # Afficher les métadonnées
                        metadata = results["metadata"]
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Durée", f"{metadata['duration_s']:.1f}s")
                        with col2:
                            st.metric("Segments", metadata["segments_count"])
                        with col3:
                            st.metric("Temps traitement", f"{metadata['processing_time_s']:.1f}s")
                        with col4:
                            st.metric("RTF", f"{metadata['rtf']:.3f}")
                        
                        # Afficher les segments
                        if results["events"]:
                            st.subheader("📊 Segments détectés")
                            segments_data = []
                            for event in results["events"]:
                                segments_data.append({
                                    "Début": f"{event['start']:.2f}s",
                                    "Fin": f"{event['end']:.2f}s",
                                    "Durée": f"{event['duration']:.2f}s",
                                    "Confiance": event['confidence'] or "N/A"
                                })
                            
                            st.dataframe(segments_data, use_container_width=True)
                            
                            # Téléchargement des résultats
                            json_str = json.dumps(results, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="📥 Télécharger les résultats JSON",
                                data=json_str,
                                file_name=f"{uploaded_file.name.split('.')[0]}_vad_results.json",
                                mime="application/json"
                            )
                        else:
                            st.warning("⚠️ Aucun segment de parole détecté")
                            st.info("Essayez d'ajuster les seuils VAD ou vérifiez que le fichier contient de la parole")
                    else:
                        st.error(f"❌ Erreur lors de l'analyse: {results.get('error', 'Erreur inconnue')}")
                        
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                logger.error(f"Erreur analyse: {e}")

# Section de diagnostic
st.header("🔧 Diagnostic")
st.subheader("📋 Informations système")

# Informations de base
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Python", platform.python_version())
with col2:
    st.metric("OS", platform.system())
with col3:
    st.metric("Architecture", platform.machine())

# Vérification des prérequis
if st.session_state.vad_processor:
    st.subheader("🔍 Prérequis VAD")
    prerequisites = st.session_state.vad_processor.check_prerequisites()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PyTorch", prerequisites.get("torch_version", "N/A"))
        st.metric("Device", prerequisites.get("device", "N/A"))
    with col2:
        st.metric("TorchAudio", prerequisites.get("torchaudio_version", "N/A"))
        st.metric("FFmpeg", "✅" if prerequisites.get("ffmpeg_available") else "❌")

# Footer
st.markdown("---")
st.markdown("*Procteo-Audio - Voice Activity Detection avec Pyannote*")
