import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def show_column_data(df: pd.DataFrame, null_df_under1: pd.DataFrame, column: str):

    print('\n')

    # Показати кількість пропущених даних з колонки
    print("Відсоток нульових значень для колонки", column)

    display(null_df_under1[null_df_under1.column_name == column])

    # Показати реальні дані з датафрейму
    print('\n\n', 'Перші 10 значень з колонки', column)

    print(df[column].value_counts().head(10))

    print('\n\n')

    # Побудувати boxplot для колонки
    print('Boxplot для колонки:', column)

    sns.boxplot(df[column])
    plt.show()

    print('\n\n')

    print('Розподілення значень для колонки', column)
    print(df[column].value_counts().to_string(max_rows=10))

    print('\n\n')

    # Розрахунок перцентилів для колонки
    print('Перцентилі для колонки:', column)

    print(df[column].quantile(q = [0.25,0.5,0.75,1]))

    print('\n\n')

    # Медіана
    print('Медіана для колонки:', column)

    print(df[column].median())

    print('\n\n')

    # Найбільш повторюване значення (мода)
    print('Найбільш повторюване значення (мода) для колонки:', column)

    print(df[column].mode()[0])

    print('\n\n')

    # Середнє значення
    print('Середнє значення для колонки:', column)
    print(df[column].mean())

    print('\n\n')