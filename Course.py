import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Запись из файла .csv в переменную data
data = pd.read_csv('input.csv')
print(data.head())

print(data.describe())

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
    "WITHOUT NATION",
    "GENERAL TOTAL"]

print(data.head())

data.iloc[:, 1:].type = ((data.iloc[:, 1:]) // 1000).astype(int)  # Упрощаем данные для дальнейшей работы

print(data.describe())

data['DATE'] = pd.to_datetime(data['DATE'])

print(data['GENERAL TOTAL'].astype(int))
print(data['GENERAL TOTAL'].mean().astype(int))
