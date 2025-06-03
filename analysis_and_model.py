import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Основная страница анализа данных и построения модели
def analysis_and_model_page():
    # Заголовок страницы
    st.title("Анализ данных и модель")

    # Загрузка датасета через интерфейс Streamlit
    uploaded_file = st.file_uploader("Загрузите датасет (CSV) из папки data", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # Функция для очистки и стандартизации имён столбцов
        def clean_col(col):
            col = col.replace('[', '').replace(']', '')
            col = col.replace('<', '').replace('>', '')
            col = col.replace(' ', '_')
            col = col.replace('(', '').replace(')', '')
            col = col.replace('/', '_').replace('-', '_')
            return col
        data.columns = [clean_col(col) for col in data.columns]
        # Удаление неинформативных столбцов
        data = data.drop(columns=[col for col in ['UDI', 'Product_ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'] if col in data.columns])
        # Преобразование категориального признака Type в числа
        if 'Type' in data.columns:
            data['Type'] = LabelEncoder().fit_transform(data['Type'])
        # Проверка пропусков
        st.write("Пропущенные значения:")
        st.write(data.isnull().sum())
        # Масштабирование числовых признаков
        num_features = [col for col in ['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min'] if col in data.columns]
        scaler = StandardScaler()
        data[num_features] = scaler.fit_transform(data[num_features])
        st.write("Первые строки данных:")
        st.write(data.head())
        # Разделение на признаки и целевую переменную
        X = data.drop(columns=['Machine_failure'])
        y = data['Machine_failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Сохраняем имена признаков
        feature_names = X_train.columns.tolist()
        # Словарь моделей для сравнения
        models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'SVM': SVC(kernel='linear', random_state=42, probability=True)
        }
        results = {}
        plt.figure(figsize=(8, 6))
        # Обучение и оценка моделей
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
            acc = accuracy_score(y_test, y_pred)
            conf = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=False)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
            results[name] = {'accuracy': acc, 'conf_matrix': conf, 'report': report, 'roc_auc': roc_auc}
        # Визуализация ROC-кривых
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривые моделей')
        plt.legend()
        st.subheader("ROC-кривые моделей")
        st.pyplot(plt)
        # Вывод метрик для каждой модели
        for name, res in results.items():
            st.subheader(f"{name}")
            st.write(f"Accuracy: {res['accuracy']:.2f}")
            st.write("Confusion Matrix:")
            st.write(res['conf_matrix'])
            st.write("Classification Report:")
            st.text(res['report'])
            st.write(f"ROC-AUC: {res['roc_auc']:.2f}")
        # Форма для предсказания по новым данным
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            type_val = st.selectbox("Type", [0, 1, 2], format_func=lambda x: ['L', 'M', 'H'][x])
            air_temp = st.number_input("Air temperature [K]", value=300.0)
            process_temp = st.number_input("Process temperature [K]", value=310.0)
            rot_speed = st.number_input("Rotational speed [rpm]", value=1500.0)
            torque = st.number_input("Torque [Nm]", value=40.0)
            tool_wear = st.number_input("Tool wear [min]", value=0.0)
            model_name = st.selectbox("Выберите модель для предсказания", list(models.keys()))
            submit_button = st.form_submit_button("Предсказать")
            if submit_button:
                # Создаем DataFrame для новых данных
                input_df = pd.DataFrame({
                    'Type': [type_val],
                    'Air_temperature_K': [air_temp],
                    'Process_temperature_K': [process_temp],
                    'Rotational_speed_rpm': [rot_speed],
                    'Torque_Nm': [torque],
                    'Tool_wear_min': [tool_wear],
                })
                input_df[num_features] = scaler.transform(input_df[num_features])
                # Приводим к нужному формату и порядку признаков
                input_df = input_df.reindex(columns=feature_names, fill_value=0)
                model = models[model_name]
                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else model.decision_function(input_df)[0]
                st.write(f"Предсказание: {'Отказ' if pred == 1 else 'Нет отказа'}")
                st.write(f"Вероятность отказа: {proba:.2f}")

# Точка входа для запуска страницы
if __name__ == "__main__":
    analysis_and_model_page()
