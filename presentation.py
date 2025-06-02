import streamlit as st
import reveal_slides as rs

# Функция для отображения страницы презентации
def presentation_page():
    # Заголовок страницы
    st.title("Презентация проекта")

    # Слайды презентации в формате Markdown с HTML для стилизации
    presentation_markdown = """
        # <span style='font-size:2.5em;'>Прогнозирование отказов оборудования</span>

        ---

        ## Введение

        - <span style='font-size:1.2em;'>Описание задачи и датасета</span>
        - <span style='font-size:1.2em;'>Цель: предсказать отказ оборудования (Target = 1) или его отсутствие (Target = 0)</span>

        ---

        ## Этапы работы

        1. Загрузка данных
        2. Предобработка данных
        3. Обучение модели
        4. Оценка модели
        5. Визуализация результатов

        ---

        ## Streamlit-приложение

        - Основная страница: анализ данных и предсказания
        - Страница с презентацией: описание проекта

        ---

        ## Заключение

        - Итоги и возможные улучшения
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
