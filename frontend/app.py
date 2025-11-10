import streamlit as st
import requests
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from typing import Dict, Any

# Initialize session state if needed
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Gaming Behavior Classifier", layout="centered")

menu = st.sidebar.radio("📂 Меню", ["Прогноз", "Візуалізація даних", "Кластеризація", "Порівняння методів"])

if menu == "Прогноз":
    st.title("Gaming Behavior Classifier")
    st.markdown("Введіть дані про поведінку гравця для визначення рівня залученості.")

    with st.form(key="gamer_behavior_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            Age = st.number_input("Вік", value=25, step=1)
            Gender = st.selectbox("Стать", options=[0, 1], format_func=lambda x: "Чоловік" if x == 0 else "Жінка")
            Location = st.selectbox("Локація", options=[0, 1, 2], 
                                  format_func=lambda x: "USA" if x == 0 else "Europe" if x == 1 else "Other")
            GameGenre = st.selectbox("Жанр гри", options=[0, 1, 2, 3, 4],
                                   format_func=lambda x: ["Action", "Strategy", "Sports", "RPG", "Puzzle"][x])

        with col2:
            PlayTimeHours = st.number_input("Час гри (години)", value=5.0, step=0.5)
            InGamePurchases = st.number_input("Кількість внутрішньоігрових покупок", value=0, step=1)
            GameDifficulty = st.selectbox("Складність гри", options=[0, 1, 2],
                                        format_func=lambda x: ["Easy", "Medium", "Hard"][x])
            SessionsPerWeek = st.number_input("Сесій на тиждень", value=5, step=1)

        with col3:
            AvgSessionDurationMinutes = st.number_input("Середня тривалість сесії (хвилини)", value=60.0, step=5.0)
            PlayerLevel = st.number_input("Рівень гравця", value=1, step=1)
            AchievementsUnlocked = st.number_input("Розблоковані досягнення", value=0, step=1)

        submit = st.form_submit_button("Отримати прогноз")

    if submit:
        payload = {
            "Age": Age,
            "Gender": Gender,
            "Location": Location,
            "GameGenre": GameGenre,
            "PlayTimeHours": PlayTimeHours,
            "InGamePurchases": InGamePurchases,
            "GameDifficulty": GameDifficulty,
            "SessionsPerWeek": SessionsPerWeek,
            "AvgSessionDurationMinutes": AvgSessionDurationMinutes,
            "PlayerLevel": PlayerLevel,
            "AchievementsUnlocked": AchievementsUnlocked
        }

        with st.spinner("Отримуємо прогноз..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if response.status_code == 200:
                    result = response.json()

                    st.success("✅ Прогноз отримано!")
                    st.subheader("Результат прогнозу")
                    st.write(f"**Рівень залученості:** {result['engagement_level']} (клас {result['predicted_class']})")
                    st.write(f"**Опис:** {result['description']}")

                    # Візуалізація важливих показників
                    st.subheader("Аналіз показників")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Ігровий час", 
                                 f"{PlayTimeHours:.1f} год",
                                 result['features_importance']['play_time'])
                    
                    with col2:
                        st.metric("Досягнення", 
                                 str(AchievementsUnlocked),
                                 result['features_importance']['achievements'])
                    
                    with col3:
                        st.metric("Частота сесій", 
                                 f"{SessionsPerWeek} на тиждень",
                                 result['features_importance']['session_frequency'])

                    # Візуалізація прогнозу
                    df = pd.DataFrame({
                        "Клас": [0, 1, 2],
                        "Рівень залученості": ["Low", "Medium", "High"],
                        "Активний": [1 if i == result['predicted_class'] else 0 for i in range(3)]
                    })
                    st.bar_chart(df.set_index("Рівень залученості")["Активний"])

                else:
                    st.error(f"❌ Помилка API: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"🚫 Помилка з'єднання з API: {e}")

    st.markdown("---")
    st.caption(f"**API URL:** {API_URL}")

elif menu == "Візуалізація даних":
    st.title("📊 Візуалізація даних поведінки гравців")
    st.markdown("Завантажте CSV-файл із даними гравців для аналізу.")

    uploaded_file = st.file_uploader("Завантажте CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Визначаємо колонки для аналізу (виключаємо ідентифікатори)
        columns_to_exclude = ['PlayerID']
        analysis_columns = [col for col in df.columns if col not in columns_to_exclude]
        
        st.subheader("🔍 Перші рядки даних")
        st.dataframe(df.head())

        st.markdown("### 📈 Статистичний опис")
        st.write(df.describe())

        # Виключаємо ідентифікатори з числового аналізу
        numeric_df = df[analysis_columns].select_dtypes(include=["float64", "int64"])

        st.markdown("### 🔗 Матриця кореляції")
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, ax=ax)
        plt.title("Кореляція між показниками поведінки гравців")
        st.pyplot(fig)
        plt.close()

        st.markdown("### 📊 Розподіл показників")
        selected_col = st.selectbox("Оберіть показник для аналізу:", numeric_df.columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_col], kde=True, bins=20, ax=ax)
        plt.title(f"Розподіл {selected_col}")
        st.pyplot(fig)
        plt.close()

        st.markdown("### ⚙️ Взаємозв'язок між показниками")
        col_x = st.selectbox("Вісь X:", numeric_df.columns, index=0, key="viz_scatter_x")
        col_y = st.selectbox("Вісь Y:", numeric_df.columns, index=min(1, len(numeric_df.columns)-1), key="viz_scatter_y")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax)
        plt.title(f"{col_x} vs {col_y}")
        st.pyplot(fig)
        plt.close()

        if "EngagementLevel" in df.columns:
            st.markdown("### 🎮 Залежність показників від рівня залученості")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x="EngagementLevel", y=selected_col, ax=ax)
            plt.title(f"Розподіл {selected_col} за рівнями залученості")
            st.pyplot(fig)
            plt.close()
            
        # Додаємо візуалізацію важливості ознак
        st.markdown("### 🎯 Важливість ознак у класифікації")
        try:
            # Визначаємо шлях до моделі (працює як локально, так і в Docker)
            ML_PATH = "/app/ml" if os.path.exists("/app/ml") else "../ml"
            model = joblib.load(f"{ML_PATH}/model.pkl")
            if hasattr(model, 'feature_importances_'):
                # Отримуємо список ознак
                feature_names = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 
                               'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
                               'Gender', 'Location', 'GameGenre', 'GameDifficulty']
                
                # Створюємо DataFrame з важливістю ознак
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Візуалізація
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
                plt.title('Важливість ознак у визначенні рівня залученості')
                ax.set_xlabel('Важливість')
                ax.set_ylabel('Ознака')
                st.pyplot(fig)
                plt.close()
                
                # Таблиця з важливістю ознак
                st.markdown("#### 📊 Деталізація важливості ознак")
                importance_df['Importance'] = importance_df['Importance'].apply(lambda x: f"{x:.4f}")
                st.table(importance_df)
                
                # Додаткова інформація
                st.info("""
                💡 **Як читати важливість ознак:**
                - Більше значення означає більший вплив ознаки на визначення рівня залученості
                - Значення показують відносний внесок кожної ознаки у прийняття рішення моделлю
                - Сума всіх значень важливості дорівнює 1
                """)
            else:
                st.warning("Модель не підтримує визначення важливості ознак")
        except Exception as e:
            st.error(f"Не вдалося завантажити модель або отримати важливість ознак: {str(e)}")
        
        # Додаткові аналітичні графіки
        st.markdown("### 🎮 Аналіз по регіонах та демографії")
        
        # Перевіряємо наявність необхідних колонок
        if 'Location' in df.columns and 'GameGenre' in df.columns:
            # Маппінг для Location та GameGenre
            location_map = {0: 'USA', 1: 'Europe', 2: 'Other'}
            genre_map = {0: 'Action', 1: 'Strategy', 2: 'Sports', 3: 'RPG', 4: 'Puzzle'}
            
            # Створюємо копію для роботи
            df_viz = df.copy()
            if df_viz['Location'].dtype in ['int64', 'int32']:
                df_viz['Location'] = df_viz['Location'].map(location_map)
            if df_viz['GameGenre'].dtype in ['int64', 'int32']:
                df_viz['GameGenre'] = df_viz['GameGenre'].map(genre_map)
            
            # 1. Популярні жанри у USA
            st.markdown("#### 🇺🇸 Популярні жанри у USA")
            usa_data = df_viz[df_viz['Location'] == 'USA']
            if len(usa_data) > 0:
                genre_counts = usa_data['GameGenre'].value_counts()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = sns.color_palette("Set2", len(genre_counts))
                bars = ax.bar(genre_counts.index, genre_counts.values, color=colors)
                ax.set_xlabel('Жанр гри')
                ax.set_ylabel('Кількість гравців')
                ax.set_title('Популярність жанрів серед гравців у USA')
                plt.xticks(rotation=45)
                
                # Додаємо значення на стовпчики
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom')
                
                st.pyplot(fig)
                plt.close()
                
                # Показуємо топ-3
                st.write("**Топ-3 найпопулярніших жанри у USA:**")
                for idx, (genre, count) in enumerate(genre_counts.head(3).items(), 1):
                    percentage = (count / len(usa_data)) * 100
                    st.write(f"{idx}. {genre}: {count} гравців ({percentage:.1f}%)")
            else:
                st.warning("Немає даних для USA")
        
        # 2. Колова діаграма жанрів за віковими групами
        if 'Age' in df.columns and 'GameGenre' in df.columns:
            st.markdown("#### 🎂 Розподіл жанрів за віковими групами")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Гравці до 20 років**")
                young_players = df_viz[df_viz['Age'] < 20]
                if len(young_players) > 0:
                    genre_young = young_players['GameGenre'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    colors = sns.color_palette("pastel", len(genre_young))
                    wedges, texts, autotexts = ax.pie(
                        genre_young.values, 
                        labels=genre_young.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=colors
                    )
                    ax.set_title('Жанри: вік < 20 років')
                    
                    # Покращуємо читабельність
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Немає даних для вікової групи < 20 років")
            
            with col2:
                st.write("**Гравці після 30 років**")
                mature_players = df_viz[df_viz['Age'] > 30]
                if len(mature_players) > 0:
                    genre_mature = mature_players['GameGenre'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    colors = sns.color_palette("muted", len(genre_mature))
                    wedges, texts, autotexts = ax.pie(
                        genre_mature.values,
                        labels=genre_mature.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=colors
                    )
                    ax.set_title('Жанри: вік > 30 років')
                    
                    # Покращуємо читабельність
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Немає даних для вікової групи > 30 років")
        
        # 3. Найбільш платоспроможна аудиторія за регіонами
        if 'Location' in df.columns and 'InGamePurchases' in df.columns:
            st.markdown("#### 💰 Найбільш платоспроможна аудиторія за регіонами")
            
            # Обчислюємо середню кількість покупок по регіонах
            purchases_by_region = df_viz.groupby('Location')['InGamePurchases'].agg([
                ('Середня к-сть покупок', 'mean'),
                ('Загальна к-сть покупок', 'sum'),
                ('К-сть гравців', 'count')
            ]).round(2)
            
            # Сортуємо за середньою кількістю покупок
            purchases_by_region = purchases_by_region.sort_values('Середня к-сть покупок', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Графік
                fig, ax = plt.subplots(figsize=(10, 6))
                x_pos = range(len(purchases_by_region))
                bars = ax.bar(x_pos, purchases_by_region['Середня к-сть покупок'], 
                             color=sns.color_palette("coolwarm", len(purchases_by_region)))
                ax.set_xlabel('Регіон')
                ax.set_ylabel('Середня кількість покупок')
                ax.set_title('Платоспроможність аудиторії за регіонами')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(purchases_by_region.index, rotation=0)
                
                # Додаємо значення на стовпчики
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
                plt.close()
            
            with col2:
                # Таблиця з детальною статистикою
                st.write("**Детальна статистика:**")
                st.dataframe(purchases_by_region.style.highlight_max(axis=0, color='lightgreen'))
                
                # Визначаємо найбільш платоспроможний регіон
                top_region = purchases_by_region.index[0]
                top_avg = purchases_by_region.iloc[0]['Середня к-сть покупок']
                st.success(f"🏆 **Найбільш платоспроможний регіон:**\n\n{top_region}\n\n{top_avg:.2f} покупок на гравця")
            
            # Додатковий аналіз: розподіл покупок
            st.markdown("#### 📊 Розподіл покупок за регіонами")
            fig, ax = plt.subplots(figsize=(12, 6))
            df_viz.boxplot(column='InGamePurchases', by='Location', ax=ax)
            ax.set_xlabel('Регіон')
            ax.set_ylabel('Кількість покупок')
            ax.set_title('Розподіл внутрішньоігрових покупок за регіонами')
            plt.suptitle('')  # Видаляємо автоматичний заголовок pandas
            st.pyplot(fig)
            plt.close()

elif menu == "Кластеризація":
    st.title("🎯 Кластеризація гравців")
    st.markdown("Завантажте CSV-файл для автоматичної кластеризації гравців за їхньою поведінкою.")
    
    if 'cluster_result' not in st.session_state:
        st.session_state.cluster_result = None
    if 'cluster_n_clusters' not in st.session_state:
        st.session_state.cluster_n_clusters = 3
    
    uploaded_file = st.file_uploader("Завантажте CSV файл", type="csv", key="cluster_file")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        missing_info = df.isnull().sum()
        if missing_info.any():
            st.warning("⚠️ Виявлено пропущені значення в даних:")
            st.write(missing_info[missing_info > 0])

            handling_method = st.radio(
                "Оберіть метод обробки пропущених значень:",
                ["Видалити рядки з пропущеними значеннями", 
                 "Заповнити середніми значеннями",
                 "Заповнити медіанними значеннями"]
            )
            
            if handling_method == "Видалити рядки з пропущеними значеннями":
                df = df.dropna()
                st.info(f"Видалено {len(df) - len(df.dropna())} рядків з пропущеними значеннями")
            elif handling_method == "Заповнити середніми значеннями":
                # Заповнюємо числові колонки середніми значеннями
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
                # Заповнюємо категоріальні колонки найчастішим значенням
                categorical_columns = df.select_dtypes(include=['object']).columns
                df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
            else:  # Заповнити медіанними значеннями
                # Заповнюємо числові колонки медіанними значеннями
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
                # Заповнюємо категоріальні колонки найчастішим значенням
                categorical_columns = df.select_dtypes(include=['object']).columns
                df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        
        # Конвертуємо категоріальні змінні в числові
        gender_mapping = {'Male': 0, 'Female': 1}
        location_mapping = {'USA': 0, 'Europe': 1, 'Other': 2}
        genre_mapping = {'Action': 0, 'Strategy': 1, 'Sports': 2, 'RPG': 3, 'Puzzle': 4}
        difficulty_mapping = {'Easy': 0, 'Medium': 1, 'Hard': 2}
        
        # Застосовуємо маппінг
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map(gender_mapping)
        if 'Location' in df.columns:
            df['Location'] = df['Location'].map(location_mapping)
        if 'GameGenre' in df.columns:
            df['GameGenre'] = df['GameGenre'].map(genre_mapping)
        if 'GameDifficulty' in df.columns:
            df['GameDifficulty'] = df['GameDifficulty'].map(difficulty_mapping)
            
        st.subheader("🔍 Перші рядки даних")
        st.dataframe(df.head())
        
        n_clusters = st.slider("Оберіть кількість кластерів", min_value=2, max_value=10, value=3)
        
        if st.button("🚀 Виконати кластеризацію", key="cluster_button"):
            with st.spinner("Виконується кластеризація..."):
                try:
                    # Конвертуємо DataFrame в CSV
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    
                    # Відправляємо оброблені дані
                    files = {"file": ("processed_data.csv", csv_data, "text/csv")}
                    params = {"n_clusters": n_clusters}
                    
                    response = requests.post(
                        f"{API_URL}/cluster", 
                        files=files, 
                        params=params,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        st.session_state.cluster_result = response.json()
                        st.session_state.cluster_n_clusters = n_clusters
                        st.success(f"✅ Кластеризацію виконано успішно!")
                    else:
                        st.error(f"❌ Помилка API: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    st.error(f"🚫 Помилка: {e}")
        
        if st.session_state.cluster_result is not None:
            result = st.session_state.cluster_result
            n_clusters = st.session_state.cluster_n_clusters
            
            st.write(f"**Кількість кластерів:** {result['n_clusters']}")
            

            df_clustered = pd.DataFrame(result['clustered_data'])

            st.subheader("📋 Дані з присвоєними кластерами")
            st.dataframe(df_clustered.head(20))
            
            st.subheader("📊 Розподіл даних по кластерах")
            cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Кількість точок у кожному кластері:**")
                st.bar_chart(cluster_counts)
            
            with col2:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(cluster_counts, labels=[f"Кластер {i}" for i in cluster_counts.index], 
                       autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
                ax.set_title("Розподіл по кластерах")
                st.pyplot(fig)
                plt.close()

            st.subheader("🎯 Центри кластерів")
            
            # Отримуємо тільки числові колонки, які використовувались для кластеризації
            clustering_features = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 
                                'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
                                'Gender', 'Location', 'GameGenre', 'GameDifficulty']
            available_features = [col for col in clustering_features if col in df_clustered.columns]
            
            # Створюємо DataFrame з центрами кластерів
            centers_df = pd.DataFrame(
                result['centers'],
                columns=available_features,
                index=[f"Кластер {i}" for i in range(n_clusters)]
            )
            
            st.dataframe(centers_df.style.highlight_max(axis=0, color='lightgreen'))
            
            st.subheader("🗺️ Візуалізація кластерів")
            
            numeric_cols = df_clustered.select_dtypes(include=['float64', 'int64']).columns.drop('Cluster', errors='ignore').tolist()
            
            if len(numeric_cols) >= 2:
                col_x = st.selectbox("Вісь X:", numeric_cols, index=0, key="cluster_x")
                col_y = st.selectbox("Вісь Y:", numeric_cols, index=min(1, len(numeric_cols)-1), key="cluster_y")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = sns.scatterplot(
                    data=df_clustered, 
                    x=col_x, 
                    y=col_y, 
                    hue='Cluster', 
                    palette='Set2',
                    s=100,
                    alpha=0.6,
                    ax=ax
                )

                if col_x in centers_df.columns and col_y in centers_df.columns:
                    ax.scatter(
                        centers_df[col_x], 
                        centers_df[col_y], 
                        c='red', 
                        s=300, 
                        alpha=0.8, 
                        marker='X',
                        edgecolors='black',
                        linewidths=2,
                        label='Центри кластерів'
                    )
                
                ax.set_title(f"Кластеризація: {col_x} vs {col_y}")
                ax.legend()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Недостатньо числових стовпців для 2D візуалізації")

            st.subheader("📦 Розподіл параметрів по кластерах")
            selected_param = st.selectbox("Оберіть параметр:", numeric_cols, key="cluster_boxplot_param")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df_clustered, x='Cluster', y=selected_param, palette='Set2', ax=ax)
            ax.set_title(f"Розподіл {selected_param} по кластерах")
            ax.set_xlabel("Кластер")
            st.pyplot(fig)
            plt.close()

            st.subheader("💾 Завантажити результати")
            csv_result = df_clustered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Завантажити CSV з кластерами",
                data=csv_result,
                file_name="clustered_data.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👆 Завантажте CSV файл, щоб розпочати кластеризацію")

        st.markdown("### 📝 Приклад формату CSV файлу")
        st.code("""Temperature,Humidity,PM2.5,PM10,NO2,SO2,CO
29.8,59.1,2.3,12.2,30.8,9.7,1.64
28.3,75.6,2.3,12.2,30.8,9.7,1.64
23.1,74.7,4.5,16.8,30.2,7.0,1.30""", language="csv")


elif menu == "Порівняння методів":
    st.title("🔬 Порівняння методів класифікації")
    st.markdown("""
    Цей розділ порівнює:
    - **Supervised Learning** (класифікація) - передбачення рівня залученості на основі навченої моделі
    - **Unsupervised Learning** (кластеризація) - автоматичне групування гравців за поведінкою
    
    **Мета:** Перевірити, як природні кластери співвідносяться з рівнями залученості гравців.
    """)
    st.markdown("### 🔍 Завантажте дані для порівняння")

    if 'comparison_result' not in st.session_state:
        st.session_state.comparison_result = None
    
    uploaded_file = st.file_uploader("Завантажте CSV файл з даними гравців", type="csv", key="compare_file")
    
    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.subheader("🔍 Перші рядки даних")
        st.dataframe(df_preview.head())

        required_cols = ['Age', 'Gender', 'Location', 'GameGenre', 'PlayTimeHours',
                        'InGamePurchases', 'GameDifficulty', 'SessionsPerWeek',
                        'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
        missing_cols = [col for col in required_cols if col not in df_preview.columns]
        
        if missing_cols:
            st.error(f"❌ Відсутні обов'язкові колонки: {missing_cols}")
            st.info("Файл повинен містити всі 9 параметрів для роботи моделі")
        else:
            n_clusters = st.slider("Кількість кластерів", min_value=2, max_value=10, value=3, step=1, key="compare_n_clusters")
            st.info(f"💡 Рекомендовано 3 кластери (відповідає 3 рівням залученості: Low, Medium, High)")
            
            if st.button("🚀 Виконати порівняння", key="compare_button"):
                with st.spinner("Аналізуємо..."):
                    try:
                        # Переміщуємо вказівник на початок файлу
                        uploaded_file.seek(0)
                        
                        # Правильно формуємо файл для відправки
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                        params = {"n_clusters": n_clusters}
                        
                        response = requests.post(
                            f"{API_URL}/cluster/compare",
                            files=files,
                            params=params,
                            timeout=30
                        )
                        
                        if response.status_code == 200:

                            st.session_state.comparison_result = response.json()
                            st.success("✅ Аналіз завершено!")
                        else:
                            st.error(f"❌ Помилка API: {response.status_code} - {response.text}")
                            
                    except Exception as e:
                        st.error(f"🚫 Помилка: {e}")

            if st.session_state.comparison_result is not None:
                result = st.session_state.comparison_result

                st.markdown("## 📊 Метрики порівняння")
                if 'metrics' in result:
                    metrics = result['metrics']
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Adjusted Rand Index",
                            f"{metrics['adjusted_rand_index']:.3f}",
                            help="Від -1 до 1. Чим вище, тим краще кластери відповідають класам"
                        )
                    
                    with col2:
                        st.metric(
                            "Normalized Mutual Info",
                            f"{metrics['normalized_mutual_info']:.3f}",
                            help="Від 0 до 1. Вимірює взаємну інформацію"
                        )
                    
                    with col3:
                        st.metric(
                            "Silhouette Score",
                            f"{metrics['silhouette_score']:.3f}",
                            help="Від -1 до 1. Якість кластеризації"
                        )
                else:
                    st.warning("Метрики порівняння недоступні")

                st.markdown("## 🎯 Відповідність кластерів та класів")
                
                mapping_data = result['cluster_analysis']
                
                for cluster_info in mapping_data:
                    with st.expander(f"Кластер {cluster_info['cluster_id']} ({cluster_info['size']} зразків) - Чистота: {cluster_info['engagement_purity']}%"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Основні характеристики:**")
                            st.write(f"- Середній час гри: {cluster_info['avg_play_time']:.1f} год")
                            st.write(f"- Середня к-сть сесій: {cluster_info['avg_sessions']:.1f}")
                            st.write(f"- Середня к-сть досягнень: {cluster_info['avg_achievements']:.1f}")
                        
                        with col2:
                            st.write("**Розподіл рівнів залученості:**")
                            engagement_dist = cluster_info['engagement_distribution']
                            for level, count in engagement_dist.items():
                                percentage = (count / cluster_info['size']) * 100
                                st.write(f"- {level}: {count} ({percentage:.1f}%)")
                            st.write(f"\n**Домінуючий рівень:** {cluster_info['dominant_engagement']}")

                        dist_df = pd.DataFrame(
                            list(cluster_info['engagement_distribution'].items()),
                            columns=['Рівень залученості', 'Кількість']
                        )
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.barplot(data=dist_df, x='Рівень залученості', y='Кількість', palette='viridis', ax=ax)
                        ax.set_title(f"Розподіл рівнів залученості у кластері {cluster_info['cluster_id']}")
                        st.pyplot(fig)
                        plt.close()
                

                st.markdown("## 📋 Детальні профілі кластерів")
                
                # Створюємо таблицю з профілями кластерів
                profile_data = []
                for cluster_info in result['cluster_analysis']:
                    profile_data.append({
                        'Кластер': f"Кластер {cluster_info['cluster_id']}",
                        'Розмір': cluster_info['size'],
                        'Середній час гри (год)': f"{cluster_info['avg_play_time']:.1f}",
                        'Середня к-сть сесій': f"{cluster_info['avg_sessions']:.1f}",
                        'Середні досягнення': f"{cluster_info['avg_achievements']:.1f}",
                        'Домінуючий рівень': cluster_info['dominant_engagement'],
                        'Чистота кластера (%)': f"{cluster_info['engagement_purity']:.1f}"
                    })
                
                profile_df = pd.DataFrame(profile_data)
                st.dataframe(profile_df, use_container_width=True)
                

                st.markdown("## 🗺️ Візуальне порівняння")
                
                df_result = pd.DataFrame(result['detailed_data'])
                
                # Числові колонки для візуалізації
                numeric_cols = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 
                               'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
                available_cols = [col for col in numeric_cols if col in df_result.columns]
                
                if len(available_cols) >= 2:
                    col_x = st.selectbox("Вісь X:", available_cols, index=0, key="compare_x")
                    col_y = st.selectbox("Вісь Y:", available_cols, index=min(1, len(available_cols)-1), key="compare_y")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                    sns.scatterplot(
                        data=df_result,
                        x=col_x,
                        y=col_y,
                        hue='Cluster',
                        palette='Set2',
                        s=100,
                        alpha=0.7,
                        ax=ax1
                    )
                    ax1.set_title("Unsupervised: Кластери")
                    ax1.legend(title='Cluster')

                    sns.scatterplot(
                        data=df_result,
                        x=col_x,
                        y=col_y,
                        hue='Predicted_Label',
                        palette='coolwarm',
                        s=100,
                        alpha=0.7,
                        ax=ax2
                    )
                    ax2.set_title("Supervised: Передбачені класи")
                    ax2.legend(title='Class')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.markdown("""
                    **Що шукати:**
                    - ✅ Якщо кольори схожі на обох графіках → кластери відповідають класам
                    - ⚠️ Якщо кольори різні → кластери не збігаються з класами
                    """)
                
                st.markdown("## 📊 Матриця збігу (Confusion-style)")

                confusion_pivot = pd.crosstab(
                    df_result['Cluster'],
                    df_result['Predicted_Label'],
                    margins=True
                )
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(
                    confusion_pivot.iloc[:-1, :-1],  # Без margins
                    annot=True,
                    fmt='d',
                    cmap='YlOrRd',
                    ax=ax,
                    cbar_kws={'label': 'Кількість зразків'}
                )
                ax.set_title("Матриця: Кластери vs Передбачені класи")
                ax.set_xlabel("Predicted Class")
                ax.set_ylabel("Cluster")
                st.pyplot(fig)
                plt.close()
                
                st.markdown("## 💾 Завантажити результати")
                csv_data = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                                label="📥 Завантажити CSV з кластерами та класами",
                                data=csv_data,
                                file_name="comparison_results.csv",
                                mime="text/csv"
                            )
    
    else:
        st.info("👆 Завантажте CSV файл для початку аналізу")
        st.markdown("""
        ### 📝 Вимоги до файлу:
        Файл повинен містити **всі 11 параметрів**:
        - Age (Вік)
        - Gender (Стать: Male/Female або 0/1)
        - Location (Локація: USA/Europe/Other або 0/1/2)
        - GameGenre (Жанр гри: Action/Strategy/Sports/RPG/Puzzle або 0/1/2/3/4)
        - PlayTimeHours (Час гри в годинах)
        - InGamePurchases (Кількість внутрішньоігрових покупок)
        - GameDifficulty (Складність гри: Easy/Medium/Hard або 0/1/2)
        - SessionsPerWeek (Сесій на тиждень)
        - AvgSessionDurationMinutes (Середня тривалість сесії в хвилинах)
        - PlayerLevel (Рівень гравця)
        - AchievementsUnlocked (Розблоковані досягнення)
        
        💡 Можете використати файл `online_gaming_behavior_dataset.csv` з папки `ml/`
        """)
