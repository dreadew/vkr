import streamlit as st
import reveal_slides as rs

# Функция для отображения страницы презентации
def presentation_page():
    # Заголовок страницы
    st.title("Презентация проекта")

    # Слайды презентации в формате Markdown с HTML для стилизации
    presentation_markdown = """
        ## <span style='font-size:1.2em;'>Прогнозирование отказов оборудования</span>

        ---

        ## Введение

        <div style='text-align:left;'>
            <span style='font-size:0.65em;'>Задача:</span><br>
            <span style='font-size:0.65em;'>Разработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования (Target = 1) или нет (Target = 0).</span><br>
            <span style='font-size:0.65em;'>Описание датасета:</span><br>
            <span style='font-size:0.65em;'>Используется синтетический датасет "AI4I 2020 Predictive Maintenance Dataset" (10 000 записей, 14 признаков), моделирующий работу оборудования.</span><br>
            <span style='font-size:0.65em;'>Основные признаки: температура, скорость вращения, крутящий момент, износ инструмента, тип продукта и др.</span><br>
            <span style='font-size:0.65em;'>Целевая переменная: <b>Machine failure</b> (0 — нет отказа, 1 — произошёл отказ).</span><br>
            <span style='font-size:0.65em;'>Подробнее: <a href='https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset' target='_blank'>UCI Predictive Maintenance Dataset</a></span>
        </div>

        ---

        ## Этапы работы

        1. загрузка и предобработка данных;
        2. обучение моделей (Logistic Regression, Random Forest, XGBoost, SVM);
        3. оценка качества моделей (Accuracy, Confusion Matrix, ROC-AUC);
        4. визуализация результатов (ROC-кривые);
        5. предсказание по новым данным.

        ---

        ## Streamlit-приложение

        <span style='font-size:0.95em;'><b>Основная страница:</b></span><br>
        - загрузка и анализ данных;<br>
        - обучение и сравнение моделей;<br>
        - визуализация метрик и ROC-кривых;<br>
        - предсказание по новым данным.<br>
        <br>
        <span style='font-size:0.95em;'>Страница с описанием проекта:</span><br>
        - краткое описание задачи, этапов работы и используемых методов;<br>
        - презентация проекта в виде слайдов.<br>

        ---

        ## Заключение

        <div style='text-align:left;'>
            <span style='font-size:0.75em;'>Итоги:</span><br>
            <span style='font-size:0.75em;'>- реализовано интерактивное приложение для анализа и предсказания отказов оборудования;</span><br>
            <span style='font-size:0.75em;'>- проведено сравнение нескольких моделей, визуализированы метрики качества;</span><br>
            <span style='font-size:0.75em;'>- пользователь может загрузить свои данные и получить предсказание.</span><br>
            <br>
            <span style='font-size:0.75em;'>Возможные улучшения:</span><br>
            <span style='font-size:0.75em;'>- добавить больше моделей и расширить анализ признаков;</span><br>
            <span style='font-size:0.75em;'>- реализовать автоматический подбор гиперпараметров;</span><br>
            <span style='font-size:0.75em;'>- улучшить визуализацию и добавить отчёты по результатам.</span><br>
        </div>
    """

    # Боковая понель с настройками презентации
    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=600)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )

# Точка входа для запуска страницы
if __name__ == "__main__":
    presentation_page()
