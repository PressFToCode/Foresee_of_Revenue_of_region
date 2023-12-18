import math
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import shapiro, chi2_contingency,pearsonr, spearmanr,linregress
from sklearn.linear_model import LinearRegression
import mplcursors
import seaborn as sns  # Импорт библиотеки seaborn

# Запись из файла .csv в переменную data
data = pd.read_csv('input.csv')

# Изменяем название колонн
data.columns = [
    "DATE",
    "GERMANY",
    "ALBANIA",
    "AUSTRIA",
    "BELGIUM",
    "BOSNIA AND HERZEGOVINA",
    "BULGARIA",
    "CZECH REPUBLIC",
    "DENMARK",
    "ESTONIA",
    "FINLAND",
    "FRANCE",
    "CYPRUS",
    "CROATIA",
    "HOLLAND",
    "ENGLAND",
    "IRLAND",
    "SPAIN",
    "SWEDEN",
    "SWITZERLAND",
    "ITALY",
    "ICELAND",
    "MONTENEGRO",
    "KOSOVO",
    "LATVIA",
    "LITHUANIA",
    "LUXEMBURG",
    "HUNGARY",
    "MAKEDONIA",
    "MALTA",
    "NORWAY",
    "POLAND",
    "PORTUGAL",
    "ROMANIA",
    "SERBIA",
    "SLOVAKIA",
    "SLOVENIA",
    "GREECE",
    "OTHER EUROPE",
    "EUROPEAN",
    "AZERBAIJAN",
    "BELARUS",
    "ARMENIA",
    "GEORGIA",
    "KAZAKHISTAN",
    "Kyrgyzstan",
    "MOLDOVACUM",
    "OZBEKISTAN",
    "RUSSIA",
    "TACHIKISTAN",
    "TURKMENISTAN",
    "UKRAINE",
    "SUM OF CIS",
    "USA",
    "ARGENTINE",
    "BRAZIL",
    "CANADA",
    "COLOMBIA",
    "MEXICO",
    "CHILE",
    "VENEZUELA",
    "OTHER AMERICA",
    "AMERICA",
    "ALGERIA",
    "MOROCCO",
    "AFRICA",
    "LIBYA",
    "EGYPT",
    "SUDAN",
    "TUNISIA",
    "OTHER AFRICA",
    "SUM OF AFRICA",
    "UAE",
    "BAHRAIN",
    "BANGLADESH",
    "CHINA",
    "INDONESIA",
    "PHILIPPINES",
    "SOUTH KOREA",
    "INDIA",
    "IRAQ",
    "IRAN",
    "ISRAEL",
    "JAPAN",
    "NORTHERN CYPRUS",
    "QATAR",
    "KUWAIT",
    "LEBANON",
    "MALAYSIA",
    "PAKISTAN",
    "SINGAPORE",
    "SYRIA",
    "SUUDARABIA",
    "THAILAND",
    "URDUN",
    "YEMEN",
    "OTHER ASIA",
    "ASIA",
    "AUSTRALIA",
    "NEW ZEALAND",
    "OCEANIA",
    "TOTAL REVENUE",
    "TOURISTS"]

data['DATE'] = pd.to_datetime(data['DATE'])

# Создание столбчатой диаграммы
plt.figure(figsize=(10, 6))
plt.bar(data['DATE'], data['TOURISTS'], color='blue', alpha=0.7, label='Tourists',width=12)

# Добавление подписей, заголовка и легенды
plt.xlabel('Season')
plt.ylabel('Number of Tourists')
plt.title('Tourists by Season')
plt.legend()

# Показать график
plt.show()

# Удаляем все ненужные данные для исследования из датафрейма
columns_to_keep = ['DATE', 'TOTAL REVENUE', 'TOURISTS']
data = data[columns_to_keep]

#Группируем данные по годам
data['YEAR'] = pd.DatetimeIndex(data['DATE']).year

# Задаем новый дата сет, который хранит данные по годам
data['YEAR'] = pd.DatetimeIndex(data['DATE']).year
data_by_years = data.groupby('YEAR')['TOTAL REVENUE'].sum().reset_index()
data_by_years['TOTAL REVENUE'] = [25415067000, 25064482000,24930997000,28115692000,29007003000,32308991000,34305903000,31464777000,22107440000,26283656000,29512926000,34520332000,19059320000,30520332000 ]

data_by_years['TOTAL_TOURISTS'] = data.groupby(data['DATE'].dt.year)['TOURISTS'].sum().values

# Упрощаем данные для дальнейшей работы приводя их к миллионам
data_by_years.iloc[:, 1:] = (data_by_years.iloc[:, 1:]) / 1000000

# Теперь data_by_years содержит сумму доходов по годам
print(data_by_years)

### Начало Главы 1 Анализ объекта исследования с помощью статистических показателей.

# Вычисление среднего геометрического, из-за того, что значения слишком большие переходим к логарифмическим значениям и затем их экспонируем
lenTotal = len(data_by_years['TOTAL REVENUE'])
log_values = np.log(data_by_years['TOTAL REVENUE'])
geom_mean = np.exp(log_values.mean())

print(f"Среднее геометрическое: {geom_mean}")

# Вычисление абсолютной динамики
# Вычисление темпа роста
# Вычисление темпа прироста
for i in range(1,len(data_by_years['TOTAL REVENUE'])):
    data_by_years.loc[i, 'ABSOLUTE_DYNAMIC'] = data_by_years.loc[i, 'TOTAL REVENUE'] - data_by_years.loc[i - 1, 'TOTAL REVENUE']
    data_by_years.loc[i, 'GROWTH_RATE'] = (data_by_years.loc[i, 'TOTAL REVENUE']/data_by_years.loc[i - 1, 'TOTAL REVENUE'])*100
    data_by_years.loc[i, 'GROWTH_INCREASE'] = data_by_years.loc[i, 'GROWTH_RATE']-100

print(data_by_years[['YEAR', 'TOTAL REVENUE', 'ABSOLUTE_DYNAMIC', 'GROWTH_RATE', 'GROWTH_INCREASE']])

yn = data_by_years['TOTAL REVENUE'][len(data_by_years['TOTAL REVENUE'])-1]
y0 = data_by_years['TOTAL REVENUE'][0]


print('Средний абсолютный прирост:', (yn-y0)/len(data_by_years['TOTAL REVENUE'])-1)
print('Средний темп роста:', math.pow(yn/y0,1/len(data_by_years['TOTAL REVENUE'])))
print('Средний темп прироста:', math.pow(yn/y0,1/len(data_by_years['TOTAL REVENUE']))*100 - 100,'\n')

columns_to_save = ['YEAR', 'TOTAL REVENUE', 'ABSOLUTE_DYNAMIC', 'GROWTH_RATE', 'GROWTH_INCREASE']

# Создайте новый DataFrame, содержащий только выбранные столбцы
data_to_save = data_by_years[columns_to_save]

# Запись названия файла
excel_file_path = 'stat_indicators.xlsx'

# Сохраните DataFrame в файл Excel
data_to_save.to_excel(excel_file_path, index=False)

# Создаем объект рисунка и оси
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Строим столбчатые диаграммы для каждого параметра на отдельных осях
axs[0].bar(data_by_years['YEAR'], data_by_years['ABSOLUTE_DYNAMIC'], width=0.5, color='green', label='Абсолютная динамика')
axs[0].set_ylabel('Значения')
axs[0].legend()

axs[1].bar(data_by_years['YEAR'], data_by_years['GROWTH_INCREASE'], width=0.5, color='salmon', label='Темп прироста')
axs[1].set_ylabel('Значения')
axs[1].legend()

axs[2].bar(data_by_years['YEAR'], data_by_years['GROWTH_RATE'], width=0.5, color='blue', label='Темп роста')
axs[2].set_ylabel('Значения')
axs[2].set_xlabel('Год')
axs[2].legend()

# Устанавливаем общий заголовок для всех графиков
fig.suptitle('Столбчатая диаграмма: Зависимость различных параметров от года')

# Добавляем горизонтальную черную линию на нуле по оси x
for ax in axs:
    ax.axhline(0, color='black', linewidth=0.5)

# Отображение графиков
plt.tight_layout()
plt.show()

# Рассчитываем относительные показатели
maxy = data_by_years['TOTAL REVENUE'].max()
compares = [(b / maxy) * 100 for b in data_by_years['TOTAL REVENUE']]

# Создаем столбчатую диаграмму
plt.figure(figsize=(10, 6))
bars = plt.bar(data_by_years['YEAR'], data_by_years['TOTAL REVENUE'], label='Общие доходы по годам')

# Добавляем текст с относительными показателями на столбцах
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{compares[i]:.2f}%', ha='center', va='bottom')

# Добавляем заголовок, подписи осей и легенду
plt.xlabel('Год')
plt.ylabel('Общие доходы')
plt.title('Относительные показатели сравнения по годам')
plt.legend()

# Отображаем график
plt.tight_layout()
plt.show()

### Конец главы 1

### Начало главы 2

# Проведение теста Шапиро-Уилка
total_revenue = data_by_years['TOTAL REVENUE']
stat, p = shapiro(total_revenue)

# Вывод результатов теста
print('Статистика=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print(f'Распределение можно считать нормальным по критерию Шапиро со значимостью {alpha}\n')
else:
    print(f'Распределение нельзя считать нормальным по критерию Шапиро со значимостью {alpha}\n')

# Создание таблицы сопряженности для chi2_contingency
crosstab = pd.crosstab(index=data_by_years['TOTAL REVENUE'], columns='count')

# Проведение критерия Пирсона
chi2, p_val, _, _ = chi2_contingency(crosstab)

# Вывод результатов
print('Хи-квадрат статистика=%.3f, p-value=%.3f' % (chi2, p_val))
alpha = 0.05
if p_val > alpha:
    print(f'Распределение можно считать нормальным по критерию Пирсона со значимостью {alpha}\n')
else:
    print(f'Распределение нельзя считать нормальным по критерию Пирсона со значимостью {alpha}\n')

# Вычисление коэффициентов корреляции Пирсона
pearson_corr, _ = pearsonr(data_by_years['TOTAL_TOURISTS'], data_by_years['TOTAL REVENUE'])

# Вычисление коэффициентов ранговой корреляции Спирмена
spearman_corr, _ = spearmanr(data_by_years['TOTAL_TOURISTS'], data_by_years['TOTAL REVENUE'])

print(f"Коэффициент корреляции Пирсона: {pearson_corr}")
print(f"Коэффициент корреляции Спирмена: {spearman_corr}\n")

# Графическое представление
plt.scatter(data_by_years['TOTAL_TOURISTS'], data_by_years['TOTAL REVENUE'], label='Исходные данные')

# Добавление заголовка и легенды
plt.title('Корреляционное поле')
plt.xlabel('Туристы')
plt.ylabel('Доход')
plt.legend()

# Линейная регрессия
slope, intercept, r_value, p_value, std_err = linregress(data_by_years['TOTAL_TOURISTS'], data_by_years['TOTAL REVENUE'])

# Коэффициент детерминации R2
r_squared = r_value ** 2

print(f"Уравнение линейной регрессии: Y = {slope} * X + {intercept}")
print(f"Коэффициент детерминации R^2: {r_squared}\n")

# Линеаризация модели: создание новых переменных U и V
data_by_years['U'] = np.log10(data_by_years['TOTAL_TOURISTS'])
data_by_years['V'] = np.log10(data_by_years['TOTAL REVENUE'])

# Рассчет коэффициента корреляции Пирсона между U и V
pearson_corr_UV = data_by_years[['U', 'V']].corr().iloc[0, 1]
print(f'Коэффициент корреляции Пирсона между U и V = {pearson_corr_UV}')

n = len(data_by_years['TOTAL_TOURISTS'])
sum_u = data_by_years['U'].sum()
sum_v = data_by_years['V'].sum()
sum_uv = sum(u*v for u, v in zip(data_by_years['U'], data_by_years['V']))
sum_uu = sum(u**2 for u in data_by_years['U'])
sum_vv = sum(v**2 for v in data_by_years['V'])

# Вычисление коэффициентов нелинейной регрессии
slope_UV1 = (sum_uv/n - sum_u*sum_v/n**2) / (sum_uu/n - (sum_u/n)**2)
intercept_UV1 = (sum_v/n)-slope_UV1*(sum_u/n)

slope_UV = slope_UV1
intercept_UV = math.pow(10,intercept_UV1)

print(f'Уравнение нелинейной регрессии: y = {intercept_UV} * x^{slope_UV}')

# Вычисление коэффициента детерминации R^2
V_pred = [slope_UV1*u + intercept_UV1 for u in data_by_years['U']]
ssr = sum((v - v_pred)**2 for v, v_pred in zip(data_by_years['V'], V_pred))
sst = sum((v - sum_v/n)**2 for v in data_by_years['V'])
r_sq_UV = 1 - ssr/sst

print(f'Коэффициент детерминации R^2: {r_sq_UV}')

Y_line_linear = [intercept+x *slope for x in data_by_years['TOTAL_TOURISTS']]\

print('Y:', Y_line_linear)

# Сортируем данные по оси X (TOTAL_TOURISTS) в порядке возрастания
sorted_data = data_by_years.sort_values('TOTAL_TOURISTS')

Y_line_nonlinear = [intercept_UV* x ** slope_UV for x in data_by_years['TOTAL_TOURISTS']]

plt.plot(data_by_years['TOTAL_TOURISTS'], intercept_UV * data_by_years['TOTAL_TOURISTS']**slope_UV, color='green', label='Нелинейная аппроксимация')
plt.plot(data_by_years['TOTAL_TOURISTS'], intercept + slope * data_by_years['TOTAL_TOURISTS'], color='red', label='Линейная аппроксимация')

# Создаем график с точками для данных TOTAL_TOURISTS и TOTAL_REVENUE
plt.scatter(data_by_years['TOTAL_TOURISTS'], data_by_years['TOTAL REVENUE'])

# Создаем линию, соединяющую точки по порядку от наименьшего к наибольшему значению TOTAL_TOURISTS
plt.plot(sorted_data['TOTAL_TOURISTS'], sorted_data['TOTAL REVENUE'], linestyle='-',color='blue',marker='o')

plt.title('Линейная аппроксимация')
plt.xlabel('Туристы')
plt.ylabel('Доход')
plt.legend()

mplcursors.cursor(hover=True)
# Отображение графика
plt.show()

### Конец главы 2

### Начало главы 3

#Критерий Фишера фактический
F = r_squared/(1-r_squared)*(14-2)
print(F)

Ft = 4.75

if(F>Ft):
    print('Полученное уравнение является статистически значимым при a = 0.05')
else:
    print('Полученное уравненеи не является статистически значимым при a = 0.05')

#Расчет дисперсии и СКО коэфиициентов a и b

SKOy = data_by_years['TOTAL REVENUE'].std()
SKOx = data_by_years['TOTAL_TOURISTS'].std()

Soct = 0

for i in range (0,len(Y_line_linear)):
    Soct += (data_by_years['TOTAL REVENUE'][i]-Y_line_linear[i])**2

Soct = math.sqrt(Soct) / 12

Sb = 0

for i in range (0,len(data_by_years['TOTAL_TOURISTS'])):
    Sb += data_by_years['TOTAL_TOURISTS'][i]**2

Sb = Soct * math.sqrt(Sb)/(14*SKOy)

Sa = Soct/(math.sqrt(14)*SKOx)

ta = slope/Sa
tb = intercept/Sb

### Конец главы 3

### Начало главы 4


# Делаем прогноз на 2021 год

print('Прогноз на 2020 год: ',data_by_years['TOTAL REVENUE'][len(data_by_years['TOTAL REVENUE'])-3] + (yn-y0)/len(data_by_years['TOTAL REVENUE'])-1)

print('Прогноз на 2020 год: ',data_by_years['TOTAL REVENUE'][len(data_by_years['TOTAL REVENUE'])-3] * math.pow(yn/y0,1/len(data_by_years['TOTAL REVENUE'])))

print('Прогноз на 2020 год: ',slope * 13 + intercept)

print('Прогноз на 2021 год: ',data_by_years['TOTAL REVENUE'][len(data_by_years['TOTAL REVENUE'])-2] + (yn-y0)/len(data_by_years['TOTAL REVENUE'])-1)

print('Прогноз на 2021 год: ',data_by_years['TOTAL REVENUE'][len(data_by_years['TOTAL REVENUE'])-2] * math.pow(yn/y0,1/len(data_by_years['TOTAL REVENUE'])))

print('Прогноз на 2021 год: ',slope * 14 + intercept)

print('Прогноз на 2021 год, если бы не было пандемии: ',34883.993 + (yn-y0)/len(data_by_years['TOTAL REVENUE'])-1)

print('Прогноз на 2021 год, если бы не было пандемии: ',34883.993 * math.pow(yn/y0,1/len(data_by_years['TOTAL REVENUE'])))

print('Прогноз на 2021 год, если бы не было пандемии: ',slope * 14 + intercept)

### Конец главы 4
