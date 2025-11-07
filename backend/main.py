from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import joblib
import numpy as np
import json
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

try:
    model = joblib.load("../ml/model.pkl")
    scaler = joblib.load("../ml/scaler.pkl")
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

        features = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 
                   'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
                   'Gender', 'Location', 'GameGenre', 'GameDifficulty']
        
        numeric_df = df[features]

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(numeric_df)

        centers = kmeans.cluster_centers_.tolist()

        cluster_analysis = []
        for i in range(n_clusters):
            cluster_data = df[df['Cluster'] == i]
            cluster_analysis.append({
                "cluster_id": i,
                "size": len(cluster_data),
                "avg_play_time": float(cluster_data['PlayTimeHours'].mean()),
                "avg_sessions": float(cluster_data['SessionsPerWeek'].mean()),
                "avg_achievements": float(cluster_data['AchievementsUnlocked'].mean()),
                "common_genre": cluster_data['GameGenre'].mode()[0],
                "avg_player_level": float(cluster_data['PlayerLevel'].mean())
            })

        result = {
            "n_clusters": n_clusters,
            "centers": centers,
            "cluster_analysis": cluster_analysis,
            "clustered_data": df.to_dict(orient="records")
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка при кластеризації: {e}")


@app.post("/cluster/compare")
async def cluster_compare_with_classification(file: UploadFile = File(...), n_clusters: int = 3):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        required_features = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 
                           'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
                           'Gender', 'Location', 'GameGenre', 'GameDifficulty']
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Відсутні колонки: {missing_features}"
            )
        
        X = df[required_features].fillna(0)
        
        # Кластеризація
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
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
                "avg_play_time": float(cluster_data['PlayTimeHours'].mean()),
                "avg_sessions": float(cluster_data['SessionsPerWeek'].mean()),
                "avg_achievements": float(cluster_data['AchievementsUnlocked'].mean()),
                "engagement_distribution": engagement_dist,
                "dominant_engagement": dominant_level,
                "engagement_purity": round(dominant_percentage, 2)
            })

        return {
            "n_clusters": n_clusters,
            "total_samples": len(df),
            "metrics": {
                "adjusted_rand_index": round(ari, 3),
                "normalized_mutual_info": round(nmi, 3),
                "silhouette_score": round(silhouette, 3)
            },
            "cluster_analysis": cluster_analysis,
            "detailed_data": df.to_dict(orient="records")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка при порівнянні: {str(e)}")
