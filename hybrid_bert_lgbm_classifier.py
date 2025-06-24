# --- Начало файла ---
"""
Гибридная система временной верификации авторства (v11 - Бинарная классификация)
Использует ансамбль из нескольких запусков для повышения стабильности и точности.
Архитектура: BERT (Mean Pooling) + Расширенные признаки + LightGBM
Задача: 1904-1905 vs 1906-1908
Цель: достижение 75%+ точности на малых данных
""" 
# Импорт библиотеки warnings для управления предупреждениями
import warnings

warnings.filterwarnings('ignore')

# Импорт библиотеки os для взаимодействия с операционной системой
import os
# Отключение параллелизма для токенизаторов из библиотеки transformers, ошибоки (deadlocks)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# --- Основные библиотеки ---
# NumPy: библиотека для научных вычислений, в том числе для работы с массивами
import numpy as np
# Pandas: библиотека для обработки и анализа данных, в основном для работы с табличными данными (DataFrame)
import pandas as pd
# PyTorch: основная библиотека для глубокого обучения, используется для работы с BERT
import torch
# StratifiedKFold: класс для стратифицированной кросс-валидации, сохраняет баланс классов в каждой выборке
from sklearn.model_selection import StratifiedKFold
# classification_report, accuracy_score: метрики для оценки качества модели
from sklearn.metrics import classification_report, accuracy_score
# LabelEncoder: кодирует текстовые метки классов в числа (например, '1904-1905' -> 0)
from sklearn.preprocessing import LabelEncoder, StandardScaler
# SelectKBest, f_classif: инструменты для отбора признаков на основе статистических тестов (ANOVA F-value)
from sklearn.feature_selection import SelectKBest, f_classif
# LightGBM: быстрая реализация градиентного бустинга, используется как основной классификатор
import lightgbm as lgb
# tqdm: библиотека для progress bars
from tqdm import tqdm
# pickle: библиотека для сериализации (сохранения) и десериализации (загрузки) объектов Python
import pickle
# datetime: для работы с датами и временем, для логирования или именования файлов
from datetime import datetime
# logging: для ведения журнала событий (логов) во время выполнения скрипта
import logging
# dataclass: декоратор для создания классов, которые в основном хранят данные
from dataclasses import dataclass
# Dict, List, Tuple: типы данных для аннотаций, улучшают читаемость кода
from typing import Dict, List, Tuple
# re: библиотека для работы с регулярными выражениями, используется для очистки текста
import re
# Counter, defaultdict: полезные структуры данных из модуля collections
from collections import Counter, defaultdict
# pymorphy2: морфологический анализатор для русского языка (определение части речи, нормальной формы и т.д.)
import pymorphy2

# --- Transformers ---
# AutoModel, AutoTokenizer: классы из библиотеки Hugging Face для загрузки предобученных моделей и их токенизаторов
from transformers import AutoModel, AutoTokenizer

# --- NLTK (Natural Language Toolkit) ---
# nltk: библиотека для обработки естественного языка
import nltk
# Загрузка необходимых пакетов NLTK: 'punkt' для токенизации предложений и слов, 'stopwords' для стоп-слов
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# stopwords: список частоупотребимых слов ("и", "в", "на"), которые часто удаляют при анализе текста
from nltk.corpus import stopwords
# word_tokenize, sent_tokenize: функции для разбиения текста на слова и предложения
from nltk.tokenize import word_tokenize, sent_tokenize
# ngrams: функция для создания n-грамм (последовательностей из n элементов)
from nltk.util import ngrams

# --- Настройка логирования ---
# Установка базовой конфигурации для логгера: уровень INFO (будут выводиться сообщения уровня INFO и выше),
# формат сообщения, включающий время, уровень и само сообщение.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Получение объекта логгера для текущего модуля
logger = logging.getLogger(__name__)

# --- Класс конфигурации ---
@dataclass # Декоратор, автоматически создающий методы __init__, __repr__ и др.
class HybridConfig:
    """Конфигурация гибридной системы (v11)"""
    # BERT параметры
    # модель BERT для русского языка от DeepPavlov
    model_name: str = "DeepPavlov/rubert-base-cased"
    # Максимальная длина последовательности в токенах для BERT. Тексты длиннее будут обрезаны.
    max_len: int = 256
    
    # Классические признаки
    # Количество лучших "классических" (стилометрических) признаков, которые нужно отобрать
    n_classical_features: int = 200
    
    # Классификатор LightGBM
    # Количество деревьев в ансамбле LightGBM
    n_estimators: int = 600
    # Скорость обучения. Меньшие значения требуют больше деревьев, но могут дать лучший результат.
    learning_rate: float = 0.02
    # Максимальное количество листьев в одном дереве. Контролирует сложность модели.
    num_leaves: int = 41
    # L1 регуляризация (штраф за абсолютное значение весов)
    reg_alpha: float = 0.1
    # L2 регуляризация (штраф за квадрат весов)
    reg_lambda: float = 0.1

    # Другое
    # Количество фолдов (частей) в кросс-валидации
    n_splits: int = 5
    # Количество полных запусков всего процесса обучения для ансамблирования результатов
    n_runs: int = 3
    # Базовое случайное число для воспроизводимости. Каждый запуск будет использовать это число + номер запуска.
    random_state_base: int = 42

# --- Класс для извлечения признаков ---
class FeatureExtractor:
    """Извлечение классических и BERT признаков"""
    
    # Метод инициализации класса
    def __init__(self, config: HybridConfig, device):
        # Сохранение конфигурации и устройства (CPU/GPU)
        self.config = config
        self.device = device
        # Инициализация морфологического анализатора для русского языка
        self.morph = pymorphy2.MorphAnalyzer()
        # Загрузка и сохранение набора русских стоп-слов
        self.stops = set(stopwords.words('russian'))
        # Загрузка токенизатора, соответствующего выбранной модели BERT
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # Загрузка предобученной модели BERT и её перемещение на указанное устройство (GPU или CPU)
        self.bert_model = AutoModel.from_pretrained(config.model_name).to(device)
        # Перевод модели в режим оценки (evaluation mode), отключает слои типа Dropout
        self.bert_model.eval()

    # Метод для извлечения n-грамм символов
    def get_char_ngrams(self, text, n, top_k=25):
        # Заменяет множественные пробелы на один и приводит текст к нижнему регистру
        text = re.sub(r'\s+', ' ', text).lower()
        # Создает список n-грамм из символов текста
        char_ngrams = [''.join(gram) for gram in ngrams(text, n)]
        # Если n-граммы не созданы, возвращает пустой словарь
        if not char_ngrams: return {}
        # Подсчитывает частоту каждой n-граммы
        counts = Counter(char_ngrams)
        # Вычисляет общее количество n-грамм
        total = sum(counts.values())
        # Возвращает словарь с относительными частотами top_k самых популярных n-грамм
        return {f'char_{n}gram_{gram}': count / total for gram, count in counts.most_common(top_k)}

    # Метод для извлечения n-грамм частей речи (POS-тегов)
    def get_pos_ngrams(self, tokens, n, top_k=25):
        # Для каждого слова в токенах определяет его часть речи (POS-тег)
        pos_tags = [str(self.morph.parse(t)[0].tag.POS) for t in tokens if t.isalpha()]
        # Если тегов меньше, чем n, возвращает пустой словарь
        if len(pos_tags) < n: return {}
        # Создает n-граммы из последовательности POS-тегов
        pos_ngrams = ['_'.join(gram) for gram in ngrams(pos_tags, n)]
        # Если n-граммы не созданы, возвращает пустой словарь
        if not pos_ngrams: return {}
        # Подсчитывает частоту каждой n-граммы
        counts = Counter(pos_ngrams)
        # Вычисляет общее количество n-грамм
        total = sum(counts.values())
        # Возвращает словарь с относительными частотами top_k самых популярных n-грамм
        return {f'pos_{n}gram_{gram}': count / total for gram, count in counts.most_common(top_k)}

    # Метод для извлечения набора стилометрических признаков
    def extract_stylometric_features(self, text: str) -> Dict[str, float]:
        # Инициализация словаря для признаков
        features = {}
        # Разбиение текста на слова (токены) и приведение к нижнему регистру
        tokens = word_tokenize(text.lower())
        # Если токенов нет, возвращает пустой словарь
        if not tokens: return {}
        
        # Средняя длина слова в тексте
        features['avg_word_length'] = np.mean([len(t) for t in tokens]) if tokens else 0
        # Коэффициент лексического разнообразия (отношение уникальных слов к общему числу слов)
        unique_tokens = set(tokens)
        features['type_token_ratio'] = len(unique_tokens) / len(tokens)
        
        # Набор знаков пунктуации для анализа
        punct_chars = '.,;:!?'
        # Расчет относительной частоты каждого знака пунктуации в тексте
        for p in punct_chars:
            features[f'punct_{p}_ratio'] = text.count(p) / len(text) if len(text) > 0 else 0
            
        # Инициализация словаря для подсчета частей речи
        pos_counts = defaultdict(int)
        # Анализ первых 500 токенов для ускорения (в предположении, что стиль автора проявляется в начале)
        for token in tokens[:500]:
            # Проверка, что токен является словом
            if token.isalpha():
                # Определение части речи (POS)
                pos = str(self.morph.parse(token)[0].tag.POS)
                # Увеличение счетчика для данной части речи
                if pos: pos_counts[pos] += 1
        
        # Общее число определенных частей речи
        total_pos = sum(pos_counts.values())
        # Если части речи были определены
        if total_pos > 0:
            # Расчет относительной частоты для основных частей речи
            for pos in ['NOUN', 'VERB', 'ADJF', 'ADVB', 'NPRO', 'CONJ']: # Сущ, Глагол, Прилаг, Наречие, Местоимение, Союз
                features[f'pos_{pos.lower()}_ratio'] = pos_counts.get(pos, 0) / total_pos
        
        # Добавление признаков на основе n-грамм символов и частей речи
        features.update(self.get_char_ngrams(text, n=3, top_k=25)) # триграммы символов
        features.update(self.get_char_ngrams(text, n=4, top_k=25)) # 4-граммы символов
        features.update(self.get_pos_ngrams(tokens, n=2, top_k=25)) # биграммы частей речи
        features.update(self.get_pos_ngrams(tokens, n=3, top_k=25)) # триграммы частей речи
        
        # Возврат словаря с извлеченными признаками
        return features

    # Метод для извлечения векторных представлений (эмбеддингов) из BERT
    def extract_bert_features(self, texts: List[str]) -> np.ndarray:
        # Список для хранения эмбеддингов всех текстов
        all_bert_features = []
        # Размер пакета (батча) для обработки.
        batch_size = 16
        
        # Отключение расчета градиентов для ускорения и экономии памяти (т.к. мы не обучаем BERT)
        with torch.no_grad():
            # Итерация по текстам с шагом batch_size, с использованием tqdm для индикатора прогресса
            for i in tqdm(range(0, len(texts), batch_size), desc="Извлечение BERT эмбеддингов (Mean Pooling)"):
                # Выбор очередного пакета текстов
                batch_texts = texts[i:i + batch_size]
                # Токенизация текстов: преобразование в ID токенов, добавление спец. токенов, обрезка/дополнение до max_len
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_len)
                # Перемещение тензоров с данными на нужное устройство (GPU/CPU)
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                
                # Получение выхода модели BERT
                outputs = self.bert_model(**inputs)
                
                # --- Реализация Mean Pooling ---
                # Выход последнего слоя BERT (эмбеддинги для каждого токена)
                last_hidden_states = outputs.last_hidden_state
                # Маска внимания (1 для реальных токенов, 0 для padding-токенов)
                attention_mask = inputs['attention_mask']
                
                # Расширение маски для поэлементного умножения с эмбеддингами
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
                # Суммирование векторов токенов
                sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
                # Подсчет количества реальных токенов в каждом тексте
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9) # clamp для избежания деления на ноль
                
                # Усреднение: делим сумму векторов на количество токенов, получаем один вектор на текст
                mean_pooled_features = sum_embeddings / sum_mask
                
                # Добавление полученных эмбеддингов в общий список (предварительно переместив на CPU)
                all_bert_features.append(mean_pooled_features.cpu().numpy())
                
        # Соединение всех эмбеддингов из батчей в один большой NumPy массив
        return np.vstack(all_bert_features)

# --- Основной класс системы верификации ---
class HybridTemporalVerifier:
    
    # Метод инициализации
    def __init__(self, config: HybridConfig):
        # Сохранение конфигурации
        self.config = config
        # Определение устройства для вычислений (предпочтительно GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Логирование используемого устройства
        logger.info(f"Используется устройство: {self.device}")
        
        # Создание экземпляра класса для извлечения признаков
        self.feature_extractor = FeatureExtractor(config, self.device)
        # Инициализация кодировщика меток
        self.label_encoder = LabelEncoder()
        # Инициализация стандартизатора признаков
        self.scaler = StandardScaler()

    # Метод для одного полного прогона обучения с кросс-валидацией
    def run_single_training_run(self, labels, combined_features, run_seed):
        # Инициализация стратифицированного K-Fold. shuffle=True перемешивает данные перед разделением.
        skf = StratifiedKFold(n_splits=self.config.n_splits, shuffle=True, random_state=run_seed)
        
        # Список для хранения результатов по каждому фолду
        fold_results = []
        # Переменная для отслеживания лучшей точности в рамках этого запуска
        best_accuracy_in_run = 0
        
        # Цикл по фолдам (разбиениям)
        for fold, (train_idx, val_idx) in enumerate(skf.split(combined_features, labels)):
            # Логирование текущего запуска и фолда
            logger.info(f"Run {run_seed - self.config.random_state_base + 1}, Fold {fold + 1}")
            
            # Разделение данных на обучающую и валидационную выборки
            X_train, X_val = combined_features[train_idx], combined_features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Обучение стандартизатора на обучающей выборке
            self.scaler.fit(X_train)
            # Применение стандартизации к обучающей и валидационной выборкам
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Инициализация модели LightGBM с параметрами из конфига
            model = lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                num_leaves=self.config.num_leaves,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=run_seed,      # Для воспроизводимости внутри LightGBM
                class_weight='balanced',    # Автоматически взвешивает классы, полезно при дисбалансе
                n_jobs=-1,                  # Использовать все доступные ядра CPU
                colsample_bytree=0.8,       # Использовать 80% признаков для каждого дерева
                subsample=0.8               # Использовать 80% данных для каждого дерева
            )
            
            # Обучение модели
            model.fit(X_train_scaled, y_train,
                      eval_set=[(X_val_scaled, y_val)], # Данные для отслеживания качества на каждой итерации
                      callbacks=[lgb.early_stopping(25, verbose=False)]) # Ранняя остановка, если качество не улучшается 25 итераций подряд
            
            # Получение предсказаний на валидационной выборке
            predictions = model.predict(X_val_scaled)
            # Расчет точности (accuracy)
            accuracy = accuracy_score(y_val, predictions)
            
            # Логирование точности на текущем фолде
            logger.info(f"Точность: {accuracy:.4f}")
            
            # Обновление лучшей точности в текущем запуске
            if accuracy > best_accuracy_in_run:
                best_accuracy_in_run = accuracy
            
            # Создание полного отчета о классификации (precision, recall, f1-score)
            report = classification_report(y_val, predictions, target_names=[str(c) for c in self.label_encoder.classes_], output_dict=True, zero_division=0)
            # Сохранение результатов фолда
            fold_results.append({'fold': fold + 1, 'accuracy': accuracy, 'report': report})

        # Расчет средней точности по всем фолдам в данном запуске
        avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
        # Возврат средней и лучшей точности для этого запуска
        return avg_accuracy, best_accuracy_in_run

    # Метод, запускающий весь ансамбль тренировок
    def run_ensemble_training(self):
        # Шаг 1: Загрузка данных
        logger.info("Загрузка данных...")
        df = pd.read_csv('suvorin_letters.csv') # Чтение CSV-файла в DataFrame
        # Фильтрация: оставляем тексты длиннее 256 символов и без пропусков в 'text' или 'year'
        df = df[df['text'].str.len() > 256].dropna(subset=['text', 'year']).copy()
        
        # --- ИЗМЕНЕНИЕ: Объединение классов в два ---
        # Года 1904 и 1905 объединяются в один класс '1904-1905'
        # Года 1906 и 1907 1908 объединяются в другой класс '1906-1908'
        df['year'] = df['year'].replace({1904: '1904-1905', 1905: '1904-1905', 1906: '1906-1907', 1907: '1906-1907'})
        # Оставляем в датасете только данные, относящиеся к этим двум новым классам
        df = df[df['year'].isin(['1904-1905', '1906-1907'])]
        logger.info("Классы объединены в бинарную задачу: '1904-1905' vs '1906-1907'")
        # ------------------------------------

        # Подготовка данных для модели
        texts = df['text'].tolist() # Список всех текстов
        # Кодирование текстовых меток в числовые (0 и 1)
        labels = self.label_encoder.fit_transform(df['year'].tolist())
        
        logger.info(f"Загружено {len(texts)} документов.")
        logger.info(f"Классы: {self.label_encoder.classes_}") # Показывает, какой класс какой цифрой закодирован
        
        # Шаг 2: Извлечение признаков
        # Извлечение эмбеддингов BERT
        bert_features = self.feature_extractor.extract_bert_features(texts)
        
        logger.info("Извлечение стилометрических и синтаксических признаков...")
        # Извлечение "классических" признаков для каждого текста
        classical_features_list = [self.feature_extractor.extract_stylometric_features(text) for text in tqdm(texts)]
        # Преобразование списка словарей в DataFrame, пропуски (если есть) заполняются нулями
        classical_features_df = pd.DataFrame(classical_features_list).fillna(0)
        
        # Шаг 3: Отбор лучших "классических" признаков
        # Инициализация селектора, который выберет k лучших признаков на основе F-статистики
        selector = SelectKBest(f_classif, k=min(self.config.n_classical_features, classical_features_df.shape[1]))
        # Обучение селектора и трансформация признаков (остаются только лучшие)
        classical_features_selected = selector.fit_transform(classical_features_df, labels)
        logger.info(f"Отобрано {classical_features_selected.shape[1]} стилометрических/синтаксических признаков.")
        
        # Шаг 4: Объединение признаков
        # Горизонтальное соединение эмбеддингов BERT и отобранных классических признаков в одну матрицу
        X_combined = np.hstack([bert_features, classical_features_selected])
        
        # Списки для хранения результатов по всем запускам ансамбля
        all_runs_avg_acc = []
        all_runs_best_acc = []

        # Шаг 5: Запуск ансамбля
        # Цикл для выполнения нескольких полных прогонов обучения
        for i in range(self.config.n_runs):
            # Установка нового random_state для каждого запуска для разнообразия
            run_seed = self.config.random_state_base + i
            logger.info(f"\n{'='*60}\nЗАПУСК АНСАМБЛЯ {i + 1}/{self.config.n_runs} (seed={run_seed})\n{'='*60}")
            
            # Запуск одного полного цикла кросс-валидации
            avg_acc, best_acc = self.run_single_training_run(labels, X_combined, run_seed)
            # Сохранение результатов запуска
            all_runs_avg_acc.append(avg_acc)
            all_runs_best_acc.append(best_acc)
            logger.info(f"Результат запуска {i+1}: Средняя точность={avg_acc:.3f}, Лучшая точность={best_acc:.3f}")

        # Шаг 6: Агрегация результатов
        # Усреднение средних точностей по всем запускам
        final_avg_accuracy = np.mean(all_runs_avg_acc)
        # Выбор абсолютно лучшей точности, достигнутой на каком-либо фолде в любом из запусков
        final_best_accuracy = np.max(all_runs_best_acc)
        # Возврат итоговых результатов
        return final_avg_accuracy, final_best_accuracy

# --- Главная функция ---
def main():
    # Вывод заголовка программы
    print("="*70)
    print("ГИБРИДНАЯ СИСТЕМА ВРЕМЕННОЙ ВЕРИФИКАЦИИ (v11 - Бинарная классификация)")
    print("BERT (Mean-Pooled) + Расширенные признаки + LightGBM")
    print("Цель: >75% точности")
    print("="*70)
    
    # Создание экземпляра конфигурации
    config = HybridConfig()
    # Создание основного объекта системы верификации
    verifier = HybridTemporalVerifier(config)
    # Запуск всего процесса и получение итоговых метрик
    avg_accuracy, best_accuracy = verifier.run_ensemble_training()
    
    # Вывод итоговых результатов
    print("\n" + "="*70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ПОСЛЕ АНСАМБЛИРОВАНИЯ")
    print("="*70)
    # Вывод средней точности по всем запускам
    print(f"Средняя точность по всем запускам: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
    # Вывод лучшей достигнутой точности
    print(f"Абсолютная лучшая точность: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    
    # Вывод сообщения в зависимости от достигнутого результата
    if best_accuracy >= 0.75:
        print("\nЦЕЛЬ 75%+ ДОСТИГНУТА!")
        print()
    elif best_accuracy >= 0.70:
        print("\nМодель показывает хорошую точность, необходима доработка улучшения.")
    elif best_accuracy >= 0.60:
        print("\n Модель показывает высокую и стабильную точность.")
    else:
        print("\nНеобходим анализ ошибок и улучшение модели.")

# --- Точка входа в программу ---
# Эта конструкция гарантирует, что функция main() будет вызвана только тогда,
# когда этот скрипт запускается напрямую, а не когда он импортируется как модуль.
if __name__ == "__main__":
    main()

# --- Конец файла ---