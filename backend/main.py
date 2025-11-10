from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import pandas as pd
import io

app = FastAPI(
    title="Gaming Behavior Classification API",
    version="1.0.0",
    description="API для класифікації рівня залученості гравців на основі їхньої поведінки в грі."
)

import os

# Визначаємо шлях до ml моделей (працює як локально, так і в Docker)
ML_PATH = "/app/ml" if os.path.exists("/app/ml") else "../ml"

try:
    model = joblib.load(f"{ML_PATH}/model.pkl")
    scaler = joblib.load(f"{ML_PATH}/scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Не вдалося завантажити модель або scaler: {e}")

class GamerBehaviorRequest(BaseModel):
    Age: int
    Gender: int  # 0 = Male, 1 = Female
    Location: int  # 0 = USA, 1 = Europe, 2 = Other
    GameGenre: int  # 0 = Action, 1 = Strategy, 2 = Sports, 3 = RPG, 4 = Puzzle
    PlayTimeHours: float
    InGamePurchases: int
    GameDifficulty: int  # 0 = Easy, 1 = Medium, 2 = Hard
    SessionsPerWeek: int
    AvgSessionDurationMinutes: float
    PlayerLevel: int
    AchievementsUnlocked: int


@app.get("/")
def root():
    return {"message": "� Gaming Behavior Classification API", "docs": "/docs"}

@app.post("/predict", status_code=status.HTTP_200_OK)
def predict_engagement(data: GamerBehaviorRequest):
    try:
        # Збір даних у словник
        features_data = {
            'Age': data.Age,
            'Gender': data.Gender,
            'Location': data.Location,
            'GameGenre': data.GameGenre,
            'PlayTimeHours': data.PlayTimeHours,
            'InGamePurchases': data.InGamePurchases,
            'GameDifficulty': data.GameDifficulty,
            'SessionsPerWeek': data.SessionsPerWeek,
            'AvgSessionDurationMinutes': data.AvgSessionDurationMinutes,
            'PlayerLevel': data.PlayerLevel,
            'AchievementsUnlocked': data.AchievementsUnlocked
        }

        features = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 
                   'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
                   'Gender', 'Location', 'GameGenre', 'GameDifficulty']
        
        X = np.array([[features_data[feature] for feature in features]])
        X_scaled = scaler.transform(X)
        
        prediction = model.predict(X_scaled)[0]

        engagement_classes = {0: "Low", 1: "Medium", 2: "High"}

        return {
            "predicted_class": int(prediction),
            "engagement_level": engagement_classes.get(int(prediction), "Unknown"),
            "description": {
                0: "Низький рівень залученості. Гравець проводить мало часу в грі та має обмежену активність.",
                1: "Середній рівень залученості. Гравець регулярно грає та помірно взаємодіє з грою.",
                2: "Високий рівень залученості. Гравець дуже активний, багато часу проводить у грі та має високі досягнення."
            }.get(int(prediction), "Опис недоступний"),
            "features_importance": {
                "play_time": "високий" if data.PlayTimeHours > 10 else "середній" if data.PlayTimeHours > 5 else "низький",
                "achievements": "високий" if data.AchievementsUnlocked > 30 else "середній" if data.AchievementsUnlocked > 15 else "низький",
                "session_frequency": "високий" if data.SessionsPerWeek > 10 else "середній" if data.SessionsPerWeek > 5 else "низький"
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Помилка під час прогнозу: {e}"
        )


@app.post("/cluster")
async def cluster_data(file: UploadFile = File(...), n_clusters: int = 3):
    try:
        # Зчитування даних
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Видаляємо ідентифікаційні колонки, які не повинні брати участь в кластеризації
        columns_to_exclude = ['PlayerID']
        feature_columns = [col for col in df.columns if col not in columns_to_exclude]
        df_features = df[feature_columns]

        # Відбираємо тільки числові та закодовані категоріальні ознаки для кластеризації
        features_for_clustering = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 
                                'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
                                'Gender', 'Location', 'GameGenre', 'GameDifficulty']
        
        # Перевіряємо, чи всі необхідні ознаки присутні в даних
        available_features = [f for f in features_for_clustering if f in df_features.columns]
        numeric_df = df_features[available_features]

        # Обробка пропущених значень
        for col in numeric_df.columns:
            if numeric_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())
            else:
                numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mode()[0] if not numeric_df[col].mode().empty else 0)

        # Переконуємося, що всі значення є числовими
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce').fillna(0)

        # Стандартизація даних
        scaler = StandardScaler()
        numeric_df_scaled = scaler.fit_transform(numeric_df)
        
        # Кластеризація
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(numeric_df_scaled)
        
        # Підготовка центрів кластерів
        centers = kmeans.cluster_centers_.tolist()
        centers = [[float(val) for val in center] for center in centers]
        
        # Підготовка аналізу кластерів
        def safe_mean(series):
            try:
                return float(series.mean()) if not series.empty else 0.0
            except:
                return 0.0

        def safe_mode(series):
            try:
                return int(series.mode()[0]) if not series.empty and len(series.mode()) > 0 else 0
            except:
                return 0

        cluster_analysis = []
        # Define helper functions outside the loop
        def safe_float(value):
            if pd.isna(value):
                return 0.0
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
            
        def safe_mean(series):
            if series.empty:
                return 0.0
            mean_val = series.mean()
            return safe_float(mean_val)
            
        def safe_mode(series):
            if series.empty:
                return 0
            mode_result = series.mode()
            if len(mode_result) == 0:
                return 0
            return int(mode_result[0])
            
        for i in range(n_clusters):
            cluster_data = df[df['Cluster'] == i]
            
            cluster_analysis.append({
                "cluster_id": i,
                "size": int(len(cluster_data)),
                "avg_play_time": safe_mean(cluster_data['PlayTimeHours']),
                "avg_sessions": safe_mean(cluster_data['SessionsPerWeek']),
                "avg_achievements": safe_mean(cluster_data['AchievementsUnlocked']),
                "common_genre": safe_mode(cluster_data['GameGenre']),
                "avg_player_level": safe_mean(cluster_data['PlayerLevel'])
            })

        # Prepare final data for JSON serialization
        cleaned_df = df.copy()
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['float64', 'float32']:
                cleaned_df[column] = cleaned_df[column].fillna(0.0).astype(float)
            elif cleaned_df[column].dtype in ['int64', 'int32']:
                cleaned_df[column] = cleaned_df[column].fillna(0).astype(int)
            else:
                cleaned_df[column] = cleaned_df[column].fillna("unknown")

        result = {
            "n_clusters": int(n_clusters),
            "centers": centers,
            "cluster_analysis": cluster_analysis,
            "clustered_data": cleaned_df.to_dict(orient="records")
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка при кластеризації: {e}")


@app.post("/cluster/compare")
async def cluster_compare_with_classification(file: UploadFile = File(...), n_clusters: int = 3):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Видаляємо ідентифікаційні колонки
        columns_to_exclude = ['PlayerID']
        feature_columns = [col for col in df.columns if col not in columns_to_exclude]
        df_features = df[feature_columns]
        
        required_features = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 
                           'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
                           'Gender', 'Location', 'GameGenre', 'GameDifficulty']
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Відсутні колонки: {missing_features}"
            )
        
        X = df[required_features]
        
        # Обробка пропущених значень
        numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = X.select_dtypes(exclude=['float64', 'int64']).columns
        
        # Заповнюємо пропущені значення медіаною для числових колонок
        for col in numeric_columns:
            X[col].fillna(X[col].median(), inplace=True)
        
        # Заповнюємо пропущені значення модою для категоріальних колонок
        for col in categorical_columns:
            X[col].fillna(X[col].mode()[0], inplace=True)
        
        # Конвертація всіх значень у числові та заміна NaN на 0
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Кластеризація
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        df['Cluster'] = clusters
        
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        df['Predicted_Class'] = predictions

        engagement_classes = {0: "Low", 1: "Medium", 2: "High"}
        df['Predicted_Label'] = df['Predicted_Class'].map(engagement_classes)
        

        ari = adjusted_rand_score(predictions, clusters)
        nmi = normalized_mutual_info_score(predictions, clusters)
        silhouette = silhouette_score(X, clusters)
        
        # Аналіз кластерів
        cluster_analysis = []
        for i in range(n_clusters):
            cluster_data = df[df['Cluster'] == i]
            
            engagement_dist = cluster_data['Predicted_Label'].value_counts().to_dict()
            total = len(cluster_data)
            
            if engagement_dist:
                dominant_level = max(engagement_dist, key=engagement_dist.get)
                dominant_percentage = (engagement_dist[dominant_level] / total) * 100
            else:
                dominant_level = "N/A"
                dominant_percentage = 0
            
            cluster_analysis.append({
                "cluster_id": int(i),
                "size": int(total),
                "avg_play_time": float(cluster_data['PlayTimeHours'].mean() or 0),
                "avg_sessions": float(cluster_data['SessionsPerWeek'].mean() or 0),
                "avg_achievements": float(cluster_data['AchievementsUnlocked'].mean() or 0),
                "engagement_distribution": engagement_dist,
                "dominant_engagement": str(dominant_level),
                "engagement_purity": float(round(dominant_percentage, 2))
            })

        return {
            "n_clusters": int(n_clusters),
            "total_samples": int(len(df)),
            "metrics": {
                "adjusted_rand_index": float(round(ari, 3)),
                "normalized_mutual_info": float(round(nmi, 3)),
                "silhouette_score": float(round(silhouette, 3))
            },
            "cluster_analysis": cluster_analysis,
            "detailed_data": df.to_dict(orient="records")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка при порівнянні: {str(e)}")
