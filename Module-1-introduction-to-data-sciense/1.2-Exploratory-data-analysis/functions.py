import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import warnings

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


def age_cat(years):
    if years <= 20:
        return '0-20'
    elif years > 20 and years <= 30:
        return '20-30'
    elif years > 30 and years <= 40:
        return '30-40'
    elif years > 40 and years <= 50:
        return '40-50'
    elif years > 50 and years <= 60:
        return '50-60'
    elif years > 60 and years <= 70:
        return '60-70'
    elif years > 70:
        return '70+'


def bi_cat_count_plot(df: pd.DataFrame, column: str, hue_column: str):
    unique_hue_values = df[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14, 6)

    pltname = f'Нормалізований розподіл значень за категорією: {column}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions * 100).round(2)
    ax = proportions.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=axes[0], title=pltname)

    # анотація значень в барплоті
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.1f}%')

    pltname = f'Кількість даних за категорією: {column}'
    counts = df.groupby(hue_column)[column].value_counts()
    ax = counts.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=axes[1], title=pltname)

    for container in ax.containers:
        ax.bar_label(container)


def uni_cat_target_compare(df, column):
    bi_cat_count_plot(df, column, hue_column='TARGET')


def bi_count_plot_target(df0, df1, column, hue_column):
    pltname = 'Клієнт зі складнощами щодо платності'
    print(pltname.upper())
    bi_cat_count_plot(df1, column, hue_column)
    plt.show()

    pltname = 'Клієнти зі своєчасними платежами'
    print(pltname.upper())
    bi_cat_count_plot(df0, column, hue_column)
    plt.show()

def outlier_range(dataset,column):
    Q1 = dataset[column].quantile(0.25)
    Q3 = dataset[column].quantile(0.75)
    IQR = Q3 - Q1
    Min_value = (Q1 - 1.5 * IQR)
    Max_value = (Q3 + 1.5 * IQR)
    return Max_value

def dist_box(dataset, column):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")

      plt.figure(figsize=(16,6))

      plt.subplot(1,2,1)
      sns.distplot(dataset[column], color = 'purple')
      pltname = 'Графік розподілу для ' + column
      plt.ticklabel_format(style='plain', axis='x')
      plt.title(pltname)

      plt.subplot(1,2,2)
      red_diamond = dict(markerfacecolor='r', marker='D')
      sns.boxplot(y = column, data = dataset, flierprops = red_diamond)
      pltname = 'Боксплот для ' + column
      plt.title(pltname)

      plt.show()

def kde_no_outliers(df0, df1, Max_value0, Max_value1, column):
  plt.figure(figsize = (14,6))
  sns.kdeplot(df1[df1[column] <= Max_value1][column],label = 'Payment difficulties')
  sns.kdeplot(df0[df0[column] <= Max_value0][column],label = 'On-Time Payments')
  plt.ticklabel_format(style='plain', axis='x')
  plt.xticks(rotation = 45)
  plt.legend()
  plt.show()