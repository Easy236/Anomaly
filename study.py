import pandas


def plot_time_series_class(data, class_name, ax, step=10):
    """"""
    time_series_df = pandas.DataFrame(data)  # Конвертируем данные
    smooth_path = time_series_df.rolling(step).mean()  # Применяем скользящее среднее
    path_deviation = 2 * time_series_df.rolling(step).std()  # Стандартное отклонение
    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]
    ax.plot(smooth_path, linewidth=2)
    ax.fill_between(path_deviation.index, under_line, over_line, alpha=.125)
    ax.set_title(class_name)
