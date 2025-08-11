# Procteo-Audio

## 🎯 Objectifs

Projet dédié exclusivement au traitement audio avec VAD (Voice Activity Detection) et transcription.
- **VAD:** Pyannote (à venir)
- **Transcription:** Whisper (optionnel, à venir)
- **Aucune dépendance vidéo:** Pas de YOLO, OpenCV, ni traitement vidéo

## 🏗️ Structure du projet

```
procteo-audio/
├─ src/
│  ├─ audio/
│  │  ├─ pipeline/         # Pipeline VAD (à venir)
│  │  └─ ui/
│  │     └─ app.py         # Interface Streamlit
├─ reports/                # Sorties JSON (git-ignorées)
├─ .streamlit/
│  └─ secrets.toml         # Token HuggingFace (non commité)
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## 🚀 Installation et lancement

### 1. Cloner le projet
```bash
git clone <url-du-repo>
cd procteo-audio
```

### 2. Créer l'environnement virtuel
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configurer le token HuggingFace
Éditer `.streamlit/secrets.toml` et remplacer `A_REMPLIR` par votre token HF :
```toml
HF_TOKEN = "votre_token_ici"
```

### 5. Lancer l'application
```bash
python3 -m streamlit run src/audio/ui/app.py
```

## 📋 Roadmap

- [x] **Étape 1:** Scaffold du projet (smoke test)
- [ ] **Étape 2:** Intégration Pyannote VAD
- [ ] **Étape 3:** Pipeline VAD fonctionnel
- [ ] **Étape 4:** Intégration optionnelle Whisper

## 🔒 Sécurité

- Le fichier `.streamlit/secrets.toml` est dans `.gitignore`
- Ne jamais commiter de tokens ou secrets
- Utiliser des variables d'environnement en production

## 📝 Notes

- Projet 100% audio, aucune dépendance vidéo
- Code propre, sans héritage d'anciens projets
- Structure modulaire pour faciliter l'évolution
