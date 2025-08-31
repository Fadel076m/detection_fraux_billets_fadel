# api/main.py
from __future__ import annotations
import io
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------
# 1) Modèles Pydantic pour la documentation API
# ---------------------------------------------------------
class PredictionResult(BaseModel):
    index: int
    prediction: int
    label: str
    proba_true: Optional[float] = None
    proba_false: Optional[float] = None

class PredictionStats(BaseModel):
    total: int
    n_true: int
    n_false: int

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    stats: PredictionStats

class ModelInfo(BaseModel):
    model_path: str
    best_model: Optional[str]
    features: List[str]
    class_mapping: Dict[str, str]

# ---------------------------------------------------------
# 2) Résolution robuste des chemins (API peut être lancée de n'importe où)
# ---------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# CORRECTION DES CHEMINS - Conversion explicite en Path
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "best_model_rf.pkl")))
SCALER_PATH = Path(os.getenv("SCALER_PATH", str(ROOT / "scaler.pkl")))
META_PATH = Path(os.getenv("META_PATH", str(ROOT / "meta.json")))

print(f"[INFO] Chemin modèle: {MODEL_PATH}")
print(f"[INFO] Chemin scaler: {SCALER_PATH}")
print(f"[INFO] Chemin metadata: {META_PATH}")
print(f"[INFO] Le modèle existe: {MODEL_PATH.exists()}")
print(f"[INFO] Le scaler existe: {SCALER_PATH.exists()}")
print(f"[INFO] Les metadata existent: {META_PATH.exists()}")

# ---------------------------------------------------------
# 3) Chargement du modèle et metadata au démarrage
# ---------------------------------------------------------
def load_model_and_meta():
    """Charge le modèle, le scaler et les métadonnées, avec fallback si fichiers manquants"""
    
    # Vérification des chemins
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}. Vérifiez le chemin.")
    
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler introuvable: {SCALER_PATH}. Vérifiez le chemin.")
    
    # Charger le modèle
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[INFO] Modèle chargé: {type(model).__name__}")
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du modèle: {e}")
    
    # Charger le scaler
    try:
        scaler = joblib.load(SCALER_PATH)
        print("[INFO] Scaler chargé")
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du scaler: {e}")
    
    # Métadonnées par défaut - CORRIGÉ avec le bon ordre des colonnes
    meta = {
        "best_model": "RandomForest",
        "features": ["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"],
        "target": "label",
        "class_mapping": {"0": "faux", "1": "vrai"}
    }
    
    # Charger les métadonnées si le fichier existe
    if META_PATH.exists():
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                loaded_meta = json.load(f)
                meta.update(loaded_meta)
                print("[INFO] Métadonnées chargées depuis meta.json")
        except Exception as e:
            print(f"[WARN] Impossible de lire meta.json: {e}")
    else:
        print("[INFO] Utilisation des métadonnées par défaut")

    # Normalise mapping en int->str pour compatibilité
    class_map: Dict[int, str] = {}
    for k, v in meta.get("class_mapping", {}).items():
        try:
            class_map[int(k)] = str(v)
        except ValueError:
            print(f"[WARN] Clé de mapping invalide: {k}")
    meta["class_mapping"] = class_map

    return model, scaler, meta

try:
    model, scaler, meta = load_model_and_meta()
    FEATURES: List[str] = meta["features"]
    CLASS_MAP: Dict[int, str] = meta.get("class_mapping", {0: "faux", 1: "vrai"})
    print(f"[INFO] Modèle et scaler chargés avec succès. Features: {FEATURES}")
except Exception as e:
    print(f"[ERROR] Impossible de charger le modèle ou le scaler: {e}")
    # Création d'un modèle factice pour le développement
    print("[INFO] Création d'un modèle factice pour le développement...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Modèle factice
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_dummy = np.random.rand(100, 6)
    y_dummy = np.random.randint(0, 2, 100)
    model.fit(X_dummy, y_dummy)
    
    # Scaler factice
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    
    # Métadonnées factices - CORRIGÉ avec le bon ordre des colonnes
    meta = {
        "best_model": "RandomForest (factice)",
        "features": ["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"],
        "target": "label",
        "class_mapping": {0: "faux", 1: "vrai"}
    }
    FEATURES = meta["features"]
    CLASS_MAP = meta["class_mapping"]

# ---------------------------------------------------------
# 4) Création de l'app + CORS
# ---------------------------------------------------------
app = FastAPI(
    title="API de Détection des Faux Billets",
    version="1.0.0",
    description=""" 
Uploader un CSV avec 6 caractéristiques géométriques et obtenir des prédictions JSON (vrai/faux).

**Fonctionnalités:**
- Upload d'un fichier CSV avec les caractéristiques géométriques
- Prédictions au format JSON avec probabilités
- Support des séparateurs CSV multiples (virgule, point-virgule)
- Gestion robuste des erreurs

**Caractéristiques attendues:** diagonal, height_left, height_right, margin_low, margin_up, length
""",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 5) Gestionnaire d'erreurs global
# ---------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Erreur interne du serveur: {str(exc)}"}
    )

# ---------------------------------------------------------
# 6) Utilitaires
# ---------------------------------------------------------
def read_csv_flex(content: bytes) -> pd.DataFrame:
    """Lecture robuste du CSV: essais de différents séparateurs et formats décimaux"""
    separators_decimals = [
        (',', '.'),   # Format standard US
        (';', ','),   # Format européen
        (';', '.'),   # Format mixte
        (',', ','),   # Cas particulier
    ]
    
    for sep, decimal in separators_decimals:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep, decimal=decimal, encoding='utf-8')
            if set(FEATURES).issubset(df.columns):
                print(f"[INFO] CSV lu avec succès (sep='{sep}', decimal='{decimal}')")
                return df
        except Exception as e:
            continue
    
    # Tentative avec encoding différent
    try:
        df = pd.read_csv(io.BytesIO(content), sep=';', decimal=',', encoding='latin-1')
        if set(FEATURES).issubset(df.columns):
            return df
    except Exception:
        pass
    
    raise HTTPException(
        status_code=400, 
        detail=f"Impossible de lire le CSV. Vérifiez le format et la présence des colonnes: {FEATURES}"
    )

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne et convertit les colonnes en float, dans le bon ordre"""
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Colonnes manquantes: {sorted(list(missing))}")
    try:
        # CORRIGÉ: Sélection dans le bon ordre
        X = df[FEATURES].astype(float)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de convertir en float: {e}")
    return X

def predict_on_dataframe(X: pd.DataFrame) -> Dict[str, Any]:
    """Effectue les prédictions sur le DataFrame"""
    try:
        # Appliquer le scaler
        X_scaled = scaler.transform(X)
        
        # Faire les prédictions
        y_pred = model.predict(X_scaled)
        proba_true = None
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(X_scaled)
                if probas.shape[1] >= 2:
                    proba_true = probas[:, 1]
            except Exception:
                pass

        preds_list = []
        y_pred_list = list(map(int, y_pred))
        for i, pred in enumerate(y_pred_list):
            row = {"index": i, "prediction": pred, "label": CLASS_MAP.get(pred, str(pred))}
            if proba_true is not None:
                row["proba_true"] = round(float(proba_true[i]), 4)
                row["proba_false"] = round(float(1 - proba_true[i]), 4)
            preds_list.append(row)

        stats = {
            "total": len(y_pred_list),
            "n_true": int(sum(y_pred_list)),
            "n_false": int(len(y_pred_list) - sum(y_pred_list))
        }
        return {"predictions": preds_list, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

# ---------------------------------------------------------
# 7) Endpoints
# ---------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "API de détection des faux billets opérationnelle",
        "version": "1.0.0",
        "endpoints": {
            "health": "/",
            "model_info": "/model-info",
            "predict": "/predict-file",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features_count": len(FEATURES),
        "model_type": type(model).__name__ if model else None,
        "is_dummy_model": "RandomForest (factice)" in str(type(model))
    }

@app.get("/model-info", response_model=ModelInfo, tags=["Info"])
def model_info():
    return {
        "model_path": str(MODEL_PATH),
        "best_model": meta.get("best_model"),
        "features": FEATURES,
        "class_mapping": {str(k): v for k, v in CLASS_MAP.items()}
    }

@app.post("/predict-file", response_model=PredictionResponse, tags=["Prediction"])
async def predict_file(file: UploadFile = File(..., description="Fichier CSV contenant les caractéristiques géométriques")):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Veuillez fournir un fichier .csv valide")
    
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Fichier vide")

    df_raw = read_csv_flex(content)
    if len(df_raw) == 0:
        raise HTTPException(status_code=400, detail="Le CSV ne contient aucune ligne de données")
    if len(df_raw) > 10000:
        raise HTTPException(status_code=400, detail="Fichier trop volumineux (max 10000 lignes)")

    X = prepare_features(df_raw)
    result = predict_on_dataframe(X)
    return result

# ---------------------------------------------------------
# 8) Point d'entrée pour le développement
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )