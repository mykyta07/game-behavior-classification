import streamlit as st
import requests
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Gaming Behavior Classifier �", page_icon="�", layout="centered")

menu = st.sidebar.radio("📂 Меню", ["Прогноз", "Візуалізація даних", "Кластеризація", "Порівняння методів"])

if menu == "Прогноз":
    st.title("� Gaming Behavior Classifier")
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
        st.subheader("🔍 Перші рядки даних")
        st.dataframe(df.head())

        st.markdown("### 📈 Статистичний опис")
        st.write(df.describe())

        numeric_df = df.select_dtypes(include=["float64", "int64"])

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
            model = joblib.load("../ml/model.pkl")
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

elif menu == "Кластеризація":
    st.title("🎯 Кластеризація гравців")
    st.markdown("Завантажте CSV-файл для автоматичної кластеризації гравців за їхньою поведінкою.")
    
    if 'cluster_result' not in st.session_state:
        st.session_state.cluster_result = None
    if 'cluster_n_clusters' not in st.session_state:
        st.session_state.cluster_n_clusters = 3
    
    uploaded_file = st.file_uploader("Завантажте CSV файл", type="csv", key="cluster_file")
    
    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.subheader("🔍 Перші рядки даних")
        st.dataframe(df_preview.head())
        
        n_clusters = st.slider("Оберіть кількість кластерів", min_value=2, max_value=10, value=3, step=1, key="n_clusters_slider")
        
        if st.button("🚀 Виконати кластеризацію", key="cluster_button"):
            with st.spinner("Виконується кластеризація..."):
                try:
                    uploaded_file.seek(0)
                    
                    # Відправляємо файл на API
                    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
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
            centers_df = pd.DataFrame(
                result['centers'],
                columns=df_clustered.select_dtypes(include=['float64', 'int64']).columns.drop('Cluster', errors='ignore')
            )
            centers_df.index = [f"Кластер {i}" for i in range(n_clusters)]
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
            n_clusters = st.slider("Кількість кластерів", min_value=2, max_value=10, value=4, step=1, key="compare_n_clusters")
            st.info(f"💡 Рекомендовано 4 кластери (відповідає 4 класам: Good, Moderate, Poor, Hazardous)")
            
            if st.button("🚀 Виконати порівняння", key="compare_button"):
                with st.spinner("Аналізуємо..."):
                    try:
                        uploaded_file.seek(0)
                        
                        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
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
                metrics = result['comparison_metrics']
                
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

                st.markdown("## 🎯 Відповідність кластерів та класів")
                
                mapping_data = result['cluster_class_mapping']
                
                for cluster_info in mapping_data:
                    with st.expander(f"Кластер {cluster_info['cluster_id']} ({cluster_info['size']} зразків) - Чистота: {cluster_info['purity']}%"):
                        st.write(f"**Домінуючий клас:** {cluster_info['dominant_class']}")
                        

                        dist_df = pd.DataFrame(
                            list(cluster_info['class_distribution'].items()),
                            columns=['Клас', 'Кількість']
                        )
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.barplot(data=dist_df, x='Клас', y='Кількість', palette='viridis', ax=ax)
                        ax.set_title(f"Розподіл класів у кластері {cluster_info['cluster_id']}")
                        st.pyplot(fig)
                        plt.close()
                

                st.markdown("## 📋 Детальні профілі кластерів")
                
                for profile in result['cluster_profiles']:
                    st.markdown(f"### Кластер {profile['cluster_id']}")
                    st.write(f"📊 Розмір: {profile['size']} зразків")

                    params_df = pd.DataFrame([profile['average_parameters']])
                    st.dataframe(params_df.style.background_gradient(cmap='RdYlGn_r'))

                    st.write("**Розподіл predicted класів:**")
                    class_dist = profile['class_distribution']
                    cols = st.columns(len(class_dist))
                    for idx, (cls, count) in enumerate(class_dist.items()):
                        with cols[idx]:
                            st.metric(cls, count)
                

                st.markdown("## 🗺️ Візуальне порівняння")
                
                df_result = pd.DataFrame(result['detailed_data'])
                
                numeric_cols = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
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
        Файл повинен містити **всі 9 параметрів**:
        - Temperature
        - Humidity
        - PM2_5
        - PM10
        - NO2
        - SO2
        - CO
        - Proximity_to_Industrial_Areas
        - Population_Density
        
        💡 Можете використати файл `updated_pollution_dataset.csv` з папки `ml/`
        """)
