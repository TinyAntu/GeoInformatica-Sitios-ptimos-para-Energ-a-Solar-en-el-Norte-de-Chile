import joblib
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def entrenar_modelo_rf(positivas: gpd.GeoDataFrame, pool_negativos: gpd.GeoDataFrame, 
                       ratio: int, out_shp: str, random_state: int = 42,
                       out_model_path: str | None = None, n_estimators: int = 500) -> dict:
    """Entrena el Random Forest balanceando el dataset según el ratio dado."""
    print(f"Entrenando modelo Random Forest con Ratio 1:{ratio}...")
    
    features = ['ghi', 'slope', 'dist_transmision', 'elev']
    n_sample = min(int(len(positivas) * ratio), len(pool_negativos))
    
    if n_sample == 0:
        raise ValueError("No hay suficientes datos negativos para entrenar.")

    # Balanceo y ensamble
    neg_sample = pool_negativos.sample(n=n_sample, random_state=random_state)
    
    dataset = pd.concat([positivas, neg_sample], ignore_index=True)
    dataset = gpd.GeoDataFrame(dataset, geometry='geometry', crs='EPSG:32718').dropna(subset=features)
    
    X = dataset[features]
    y = dataset['clase']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)
    
    # Modelo
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Métricas
    y_score = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    
    importances = pd.DataFrame({'Criterio': features, 'Peso_ML': clf.feature_importances_}).sort_values('Peso_ML', ascending=False)
    
    print(f"Resultado -> AUC: {auc:.4f}")
    print(importances.to_string(index=False))
    
    # Guardar shapefile de entrenamiento
    dataset.to_file(out_shp)
    print(f"Dataset de entrenamiento guardado en: {out_shp}")

    if out_model_path:
        joblib.dump(clf, out_model_path)
        print(f"Modelo Random Forest guardado en: {out_model_path}")
    
    return {'auc': auc, 'importances': importances, 'modelo': clf}