# --- Начало файла ---

"""
Система Временной Верификации Авторства - Полная Реализация
===============================================================================

Комплексная система, объединяющая дистрибуционную семантику, теорию графов, 
продвинутое извлечение признаков и статистическую валидацию.
Цель: максимизация точности, размеров эффектов (>0.75)

Основные возможности:
- Word2Vec с оптимизированными параметрами
– Построение семантического графа с улучшенными атрибутами
- Многопризнаковый подход (лексический, синтаксический, семантический, читаемость)
- Обработка русского языка
- Ансамблевые методы и отбор признаков
- Bootstrap-анализ и доверительные интервалы
- Визуализация
"""

# Отключение предупреждений для чистого вывода
import warnings
warnings.filterwarnings('ignore')

# ===== БЛОК ИМПОРТОВ =====
# Основные библиотеки 
import numpy as np                    # Численные вычисления и массивы
import pandas as pd                   # Работа с данными в табличном формате
from scipy import stats               # Статистические тесты и функции

# Библиотеки машинного обучения sklearn
from sklearn.preprocessing import StandardScaler, RobustScaler  # Нормализация данных
from sklearn.model_selection import StratifiedKFold, cross_val_score  # Кросс-валидация
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,  # Ансамблевые методы
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression  # Логистическая регрессия
from sklearn.svm import SVC                          # Метод опорных векторов
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score  # Метрики
from sklearn.manifold import TSNE                    # Понижение размерности для визуализации
from sklearn.decomposition import PCA                # Метод главных компонент
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Линейный дискриминантный анализ
from sklearn.feature_selection import SelectKBest, f_classif, RFE  # Отбор признаков

# Библиотеки для обработки естественного языка и анализа графов
import re                             # Регулярные выражения
from collections import Counter, defaultdict  # Счетчики и словари с значениями по умолчанию
from gensim.models import Word2Vec    # Векторные представления слов
import networkx as nx                 # Анализ сетей и графов
from itertools import combinations    # Комбинации элементов

# Библиотеки для визуализации
import matplotlib.pyplot as plt        # Основная библиотека для графиков
import seaborn as sns                  # Статистическая визуализация
from matplotlib.patches import Ellipse # Геометрические фигуры для графиков

# Системные утилиты и типизация
import sys                           # Системные функции
from pathlib import Path             # Работа с путями файлов
from typing import Dict, List, Tuple, Optional, Union  # Типизация
from dataclasses import dataclass    # Декоратор для создания классов данных
import json                          # Работа с JSON
import logging                       # Логирование
from tqdm import tqdm                # Прогресс-бары
import pickle                        # Сериализация объектов
from datetime import datetime        # Работа с датой и временем

# ===== БЛОК ОПЦИОНАЛЬНЫХ БИБЛИОТЕК =====
# Загрузка продвинутых библиотек с обработкой ошибок

try:
    import pymorphy2              # Морфологический анализатор для русского языка
    MORPH_AVAILABLE = True
    print("Pymorphy2 морфологический анализатор загружен")
except ImportError:
    MORPH_AVAILABLE = False
    print("Pymorphy2 недоступен, используется базовая обработка")

try:
    import textstat               # Анализ читаемости текста
    TEXTSTAT_AVAILABLE = True
    print("TextStat анализатор читаемости загружен")
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("TextStat недоступен")

try:
    import xgboost as xgb         # Градиентный бустинг
    XGBOOST_AVAILABLE = True
    print("XGBoost загружен")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost недоступен")

# Настройка системы логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== КЛАСС КОНФИГУРАЦИИ =====
@dataclass
class EnhancedConfig:
    """Улучшенная конфигурация для высокопроизводительной временной верификации авторства."""
    
    # Параметры Word2Vec (улучшенные)
    vector_size: int = 400     # Увеличенная размерность векторов
    window_size: int = 15      # Больший контекст для анализа
    min_count: int = 1         # Включать редкие слова (исторические термины)
    epochs: int = 200          # Больше эпох обучения
    
    # Параметры графа (улучшенные)
    n_core_words: int = 40     # Больше ключевых слов
    k_similar_words: int = 30  # Больше связей между словами
    
    # Параметры машинного обучения
    cv_folds: int = 7                 # Больше фолдов для стабильности
    n_feature_selection: int = 100    # Больше отобранных признаков
    bootstrap_iterations: int = 1000  # Итераций bootstrap
    
    # Другие параметры
    random_state: int = 42            # Зерно для воспроизводимости
    significance_level: float = 0.05  # Уровень значимости
    
    # Продвинутые опции
    use_ensemble: bool = True           # Использовать ансамблевые методы
    use_feature_selection: bool = True  # Использовать отбор признаков
    use_advanced_features: bool = True  # Использовать продвинутые признаки
    use_stacking: bool = True           # Новое: стекинг ансамблей

# ===== КЛАСС ТОКЕНИЗАТОРА =====
class EnhancedTokenizer:
    """Улучшенный токенизатор с морфологическим анализом и комплексной предобработкой."""
    
    def __init__(self, language='russian'):
        """
        Инициализация токенизатора.
        
        Args:
            language: Язык обработки (по умолчанию русский)
        """
        self.language = language
        self.stopwords = self._load_comprehensive_stopwords()    # Загрузка стоп-слов
        self.punctuation_pattern = re.compile(r'[^\w\s\-\'\']')  # Паттерн для пунктуации
        
        # Инициализация морфологического анализатора
        if MORPH_AVAILABLE:
            self.morph = pymorphy2.MorphAnalyzer()
        else:
            self.morph = None
    
    def _load_comprehensive_stopwords(self) -> set:
        """Загрузка комплексного списка русских стоп-слов."""
        return {
            # Местоимения (расширенный список)
            'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
            'мой', 'моя', 'моё', 'мои', 'твой', 'твоя', 'твоё', 'твои',
            'его', 'её', 'их', 'наш', 'наша', 'наше', 'наши',
            'ваш', 'ваша', 'ваше', 'ваши', 'свой', 'своя', 'своё', 'свои',
            'меня', 'мне', 'мной', 'тебя', 'тебе', 'тобой',
            'нас', 'нам', 'нами', 'вас', 'вам', 'вами',
            'себя', 'себе', 'собой', 'собою',
            
            # Предлоги (расширенный список)
            'в', 'во', 'на', 'за', 'с', 'со', 'к', 'ко', 'по', 'от', 'ото',
            'до', 'из', 'изо', 'у', 'о', 'об', 'обо', 'про', 'при',
            'под', 'подо', 'над', 'для', 'без', 'безо', 'через', 'сквозь',
            'между', 'среди', 'меж', 'промеж', 'около', 'возле', 'вокруг',
            'перед', 'пред', 'передо', 'предо', 'после', 'благодаря',
            'согласно', 'вопреки', 'наперекор', 'вследствие', 'ввиду',
            
            # Союзы (расширенный список)
            'и', 'а', 'но', 'да', 'или', 'либо', 'то', 'что', 'чтобы',
            'как', 'когда', 'где', 'куда', 'откуда', 'почему', 'зачем',
            'если', 'ежели', 'коль', 'хотя', 'хоть', 'пусть', 'пускай',
            'потому', 'поэтому', 'так', 'там', 'тут', 'здесь', 'туда',
            'сюда', 'оттуда', 'отсюда', 'тогда', 'сейчас', 'теперь',
            
            # Вспомогательные глаголы и частицы (расширенный список)
            'быть', 'есть', 'был', 'была', 'было', 'были', 'будет', 'будут',
            'буду', 'будешь', 'будем', 'будете', 'бывать', 'бывает',
            'не', 'ни', 'нет', 'же', 'ли', 'ль', 'бы', 'б', 'ведь', 'вон',
            'вот', 'даже', 'уже', 'ужо', 'ещё', 'еще', 'только', 'лишь',
            'именно', 'прямо', 'просто', 'совсем', 'вообще', 'совершенно',
            'абсолютно', 'относительно', 'довольно', 'очень', 'весьма',
            'крайне', 'чрезвычайно', 'исключительно', 'особенно',
            'также', 'тоже', 'всё', 'все', 'всех', 'всем', 'всеми',
            'каждый', 'каждая', 'каждое', 'каждые', 'любой', 'любая',
            'любое', 'любые', 'иной', 'иная', 'иное', 'иные',
            
            # Модальные глаголы (расширенный список)
            'могу', 'можешь', 'может', 'можем', 'можете', 'могут',
            'должен', 'должна', 'должно', 'должны', 'надо', 'нужно',
            'необходимо', 'следует', 'стоит', 'хочу', 'хочешь', 'хочет',
            'хотим', 'хотите', 'хотят', 'желаю', 'желаешь', 'желает',
            'желаем', 'желаете', 'желают'
        }
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Улучшенная предобработка с морфологическим анализом.
        
        Args:
            text: Входной текст
            remove_stopwords: Удалять ли стоп-слова
            
        Returns:
            Список обработанных токенов
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return []
        
        # Нормализация текста
        text = text.lower().strip()                         # Приведение к нижнему регистру
        text = re.sub(r'\s+', ' ', text)                    # Удаление множественных пробелов
        text = self.punctuation_pattern.sub(' ', text)      # Удаление пунктуации
        
        # Токенизация
        tokens = [token.strip() for token in text.split() if token.strip()]
        
        # Фильтрация токенов
        tokens = [
            token for token in tokens 
            if len(token) > 2 and token.isalpha() and not token.isdigit()  # Только буквы, длина >2
        ]
        
        # Морфологический анализ и лемматизация
        if self.morph:
            processed_tokens = []
            for token in tokens:
                parsed = self.morph.parse(token)[0]          # Парсинг слова
                lemma = parsed.normal_form                   # Получение леммы
                pos = str(parsed.tag.POS)                    # Часть речи
                # Исключаем междометия, частицы, союзы, предлоги
                if pos not in ['INTJ', 'PRCL', 'CONJ', 'PREP']:
                    processed_tokens.append(lemma)
            tokens = processed_tokens
        
        # Удаление стоп-слов
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        return tokens
    
    def extract_pos_features(self, text: str) -> Dict[str, float]:
        """
        Извлечение признаков на основе частей речи.
        
        Args:
            text: Входной текст
            
        Returns:
            Словарь с соотношениями частей речи
        """
        if not self.morph:
            return {}
        
        tokens = text.lower().split()
        pos_counts = defaultdict(int)  # Счетчик частей речи
        total_tokens = 0
        
        for token in tokens:
            if token.isalpha() and len(token) > 2:
                parsed = self.morph.parse(token)[0]
                pos = str(parsed.tag.POS)
                if pos:
                    pos_counts[pos] += 1
                    total_tokens += 1
        
        # Вычисление соотношений частей речи
        pos_features = {}
        for pos, count in pos_counts.items():
            pos_features[f'pos_ratio_{pos.lower()}'] = count / max(total_tokens, 1)
        
        return pos_features

# ===== КЛАСС ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ =====
class ComprehensiveFeatureExtractor:
    """Извлечение комплексных признаков для улучшенной временной верификации авторства."""
    
    def __init__(self, config: EnhancedConfig):
        """
        Инициализация экстрактора признаков.
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        self.tokenizer = EnhancedTokenizer()
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """
        Извлечение комплексного набора признаков.
        
        Args:
            text: Входной текст
            
        Returns:
            Словарь со всеми признаками
        """
        features = {}
        
        # 1. Графовые признаки (улучшенные)
        graph_features = self._extract_enhanced_graph_features(text)
        features.update(graph_features)
        
        # 2. Лексические признаки (комплексные)
        lexical_features = self._extract_lexical_features(text)
        features.update(lexical_features)
        
        # 3. Синтаксические признаки
        syntactic_features = self._extract_syntactic_features(text)
        features.update(syntactic_features)
        
        # 4. Признаки читаемости
        if TEXTSTAT_AVAILABLE:
            readability_features = self._extract_readability_features(text)
            features.update(readability_features)
        
        # 5. Признаки частей речи
        pos_features = self.tokenizer.extract_pos_features(text)
        features.update(pos_features)
        
        # 6. Продвинутые n-граммные признаки
        ngram_features = self._extract_ngram_features(text)
        features.update(ngram_features)
        
        # 7. Стилометрические признаки сложности
        complexity_features = self._extract_complexity_features(text)
        features.update(complexity_features)
        
        # 8. Исторические/временные специфические признаки
        temporal_features = self._extract_temporal_markers(text)
        features.update(temporal_features)
        
        # 9. Продвинутые семантические признаки
        semantic_features = self._extract_semantic_markers(text)
        features.update(semantic_features)
        
        return features
    
    def _extract_temporal_markers(self, text: str) -> Dict[str, float]:
        """
        Извлечение признаков, специфичных для временных/исторических периодов.
        
        Args:
            text: Входной текст
            
        Returns:
            Словарь с временными маркерами
        """
        features = {}
        
        # Исторические варианты написания и архаичные формы
        archaic_forms = {
            'старые_формы': ['дѣло', 'тѣм', 'онѣ', 'всѣ', 'одинъ', 'двѣ'],
            'церковнославянизмы': ['благо', 'дабы', 'коли', 'аще', 'паче', 'токмо'],
            'революционная_лексика': ['революція', 'забастовка', 'стачка', 'пролетарій', 'буржуазія', 'капиталъ'],
            'модернизация_лексики': ['телефонъ', 'телеграфъ', 'автомобиль', 'кинематографъ', 'электричество'],
            'политическая_лексика': ['манифестъ', 'конституція', 'парламентъ', 'министръ', 'губернаторъ']
        }
        
        text_lower = text.lower()
        total_words = len(text_lower.split())
        
        # Подсчет каждой категории архаичных форм
        for category, words in archaic_forms.items():
            count = sum(text_lower.count(word.lower()) for word in words)
            features[f'temporal_{category}'] = count / total_words if total_words > 0 else 0
        
        # Маркеры временного стиля письма
        # Дореформенная vs послереформенная орфография
        old_endings = ['ъ', 'ѣ', 'і', 'ѳ', 'ѵ']  # Старые русские буквы
        old_ending_count = sum(text.count(ending) for ending in old_endings)
        features['old_orthography_ratio'] = old_ending_count / len(text) if len(text) > 0 else 0
        
        # Эволюция структуры предложений (1904-1908)
        complex_constructions = ['не только...но и', 'как...так и', 'если...то', 'хотя...но']
        complex_count = sum(1 for constr in complex_constructions if constr in text_lower)
        features['complex_constructions'] = complex_count / len(re.split(r'[.!?]', text)) if len(re.split(r'[.!?]', text)) > 0 else 0
        
        # Французское/иностранное влияние (распространенное в начале 1900х в русском)
        foreign_words = ['комфортъ', 'прогрессъ', 'цивилизація', 'культура', 'элегантный', 'шикарный']
        foreign_count = sum(text_lower.count(word) for word in foreign_words)
        features['foreign_influence'] = foreign_count / total_words if total_words > 0 else 0
        
        return features
    
    def _extract_semantic_markers(self, text: str) -> Dict[str, float]:
        """
        Извлечение продвинутых семантических и тематических маркеров.
        
        Args:
            text: Входной текст
            
        Returns:
            Словарь с семантическими маркерами
        """
        features = {}
        
        # Тематические категории, типичные для разных периодов
        themes = {
            'военная_тематика': ['война', 'воинъ', 'солдатъ', 'офицеръ', 'полкъ', 'армія', 'флотъ', 'победа', 'поражение'],
            'экономическая_тематика': ['торговля', 'промышленность', 'фабрика', 'заводъ', 'рабочій', 'капиталъ', 'прибыль'],
            'социальная_тематика': ['народъ', 'крестьяне', 'дворянство', 'интеллигенція', 'образованіе', 'просвѣщеніе'],
            'технологическая_тематика': ['паровозъ', 'железная дорога', 'телеграфъ', 'газета', 'печать', 'изобретеніе'],
            'культурная_тематика': ['театръ', 'литература', 'искусство', 'музыка', 'живопись', 'поэзія'],
            'религиозная_тематика': ['богъ', 'церковь', 'вѣра', 'молитва', 'священникъ', 'православіе', 'душа']
        }
        
        text_lower = text.lower()
        total_words = len(text_lower.split())
        
        # Подсчет слов по темам
        for theme, words in themes.items():
            count = sum(text_lower.count(word.lower()) for word in words)
            features[f'theme_{theme}'] = count / total_words if total_words > 0 else 0
        
        # Маркеры эмоционального тона
        positive_markers = ['радость', 'счастье', 'успѣхъ', 'надежда', 'любовь', 'красота', 'прекрасный']
        negative_markers = ['печаль', 'горе', 'страданіе', 'бѣда', 'ужасъ', 'страхъ', 'тревога']
        
        positive_count = sum(text_lower.count(word) for word in positive_markers)
        negative_count = sum(text_lower.count(word) for word in negative_markers)
        
        features['positive_emotion_ratio'] = positive_count / total_words if total_words > 0 else 0
        features['negative_emotion_ratio'] = negative_count / total_words if total_words > 0 else 0
        features['emotional_polarity'] = (positive_count - negative_count) / total_words if total_words > 0 else 0
        
        # Абстрактность vs конкретность
        abstract_words = ['идея', 'мысль', 'понятіе', 'принципъ', 'теорія', 'философія', 'смыслъ']
        concrete_words = ['домъ', 'столъ', 'дерево', 'камень', 'вода', 'хлѣбъ', 'деньги']
        
        abstract_count = sum(text_lower.count(word) for word in abstract_words)
        concrete_count = sum(text_lower.count(word) for word in concrete_words)
        
        features['abstractness_ratio'] = abstract_count / total_words if total_words > 0 else 0
        features['concreteness_ratio'] = concrete_count / total_words if total_words > 0 else 0
        
        # Маркеры временной перспективы
        past_markers = ['былъ', 'была', 'было', 'были', 'раньше', 'прежде', 'когда-то']
        future_markers = ['будетъ', 'будущее', 'завтра', 'впередъ', 'предстоитъ', 'намѣреваюсь']
        
        past_count = sum(text_lower.count(word) for word in past_markers)
        future_count = sum(text_lower.count(word) for word in future_markers)
        
        features['past_orientation'] = past_count / total_words if total_words > 0 else 0
        features['future_orientation'] = future_count / total_words if total_words > 0 else 0
        
        return features
    
    def _extract_ngram_features(self, text: str) -> Dict[str, float]:
        """
        Извлечение признаков на основе n-грамм для стилометрического анализа.
        
        Args:
            text: Входной текст
            
        Returns:
            Словарь с n-граммными признаками
        """
        features = {}
        
        # Символьные n-граммы для стиля
        char_2grams = []
        char_3grams = []
        
        clean_text = re.sub(r'\s+', ' ', text.lower())
        
        # Извлечение символьных биграмм и триграмм
        for i in range(len(clean_text) - 1):
            char_2grams.append(clean_text[i:i+2])
        
        for i in range(len(clean_text) - 2):
            char_3grams.append(clean_text[i:i+3])
        
        # Частоты символьных n-грамм
        if char_2grams:
            char_2gram_counts = Counter(char_2grams)
            # Наиболее частые символьные биграммы (стилистические маркеры)
            common_2grams = ['ст', 'ов', 'ен', 'то', 'на', 'ер', 'ни', 'ре', 'ко', 'ан']
            for bigram in common_2grams:
                features[f'char_2gram_{bigram}'] = char_2gram_counts.get(bigram, 0) / len(char_2grams)
        
        if char_3grams:
            char_3gram_counts = Counter(char_3grams)
            # Наиболее частые символьные триграммы
            common_3grams = ['ств', 'ени', 'тор', 'ова', 'при', 'кот', 'что', 'над', 'про', 'дел']
            for trigram in common_3grams:
                features[f'char_3gram_{trigram}'] = char_3gram_counts.get(trigram, 0) / len(char_3grams)
        
        # Словарные n-граммы
        tokens = self.tokenizer.preprocess_text(text, remove_stopwords=False)
        
        if len(tokens) > 1:
            # Паттерны функциональных слов (стилистические маркеры)
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
            bigram_counts = Counter(bigrams)
            
            # Частые русские биграммы функциональных слов
            function_bigrams = ['в_то', 'на_то', 'то_что', 'и_в', 'и_на', 'а_не', 'но_не', 'что_он']
            for bigram in function_bigrams:
                features[f'func_bigram_{bigram}'] = bigram_counts.get(bigram, 0) / len(bigrams) if bigrams else 0
        
        return features
    
    def _extract_complexity_features(self, text: str) -> Dict[str, float]:
        """
        Извлечение продвинутых стилометрических признаков сложности.
        
        Args:
            text: Входной текст
            
        Returns:
            Словарь с признаками сложности
        """
        features = {}
        
        # Сложность структуры предложений
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
        
        if sentence_lengths:
            # Вариация длины предложений
            features['sentence_length_cv'] = np.std(sentence_lengths) / np.mean(sentence_lengths)
            features['sentence_length_range'] = max(sentence_lengths) - min(sentence_lengths)
            features['long_sentence_ratio'] = sum(1 for s in sentence_lengths if s > 20) / len(sentence_lengths)
            features['short_sentence_ratio'] = sum(1 for s in sentence_lengths if s < 5) / len(sentence_lengths)
        
        # Изощренность словаря
        tokens = self.tokenizer.preprocess_text(text, remove_stopwords=True)
        
        if tokens:
            # Распределение длин слов
            word_lengths = [len(word) for word in tokens]
            features['long_word_ratio'] = sum(1 for w in word_lengths if w > 7) / len(word_lengths)
            features['short_word_ratio'] = sum(1 for w in word_lengths if w < 4) / len(word_lengths)
            
            # Лексическая плотность
            content_words = [word for word in tokens if len(word) > 3]
            features['lexical_density'] = len(content_words) / len(tokens) if tokens else 0
            
            # Паттерны повторений
            word_counts = Counter(tokens)
            repeated_words = sum(1 for count in word_counts.values() if count > 1)
            features['word_repetition_ratio'] = repeated_words / len(set(tokens)) if tokens else 0
        
        # Синтаксическая сложность
        # Индикаторы подчинительных предложений
        subordinate_markers = ['что', 'который', 'которая', 'которое', 'которые', 'где', 'когда', 'если', 'хотя']
        subordinate_count = sum(text.lower().count(marker) for marker in subordinate_markers)
        features['subordinate_clause_ratio'] = subordinate_count / len(sentences) if sentences else 0
        
        # Дискурсивные маркеры
        discourse_markers = ['однако', 'впрочем', 'кроме того', 'тем не менее', 'таким образом', 'следовательно']
        discourse_count = sum(text.lower().count(marker) for marker in discourse_markers)
        features['discourse_marker_ratio'] = discourse_count / len(sentences) if sentences else 0
        
        # Модальные выражения
        modal_expressions = ['может быть', 'возможно', 'вероятно', 'по-видимому', 'кажется', 'думаю']
        modal_count = sum(text.lower().count(expr) for expr in modal_expressions)
        features['modal_expression_ratio'] = modal_count / len(sentences) if sentences else 0
        
        return features
    
    def _extract_enhanced_graph_features(self, text: str) -> Dict[str, float]:
        """
        Извлечение улучшенных графовых признаков.
        
        Args:
            text: Входной текст
            
        Returns:
            Словарь с графовыми признаками
        """
        tokens = self.tokenizer.preprocess_text(text)
        
        if len(tokens) < 50:  # Недостаточно данных для анализа
            return {}
        
        try:
            # Построение улучшенной модели Word2Vec
            sentences = self._create_enhanced_sentences(tokens)
            
            model = Word2Vec(
                sentences=sentences,
                vector_size=self.config.vector_size,
                window=self.config.window_size,
                min_count=self.config.min_count,
                workers=1,
                sg=1,               # Skip-gram модель
                epochs=self.config.epochs,
                alpha=0.025,        # Начальная скорость обучения
                min_alpha=0.0001,   # Минимальная скорость обучения
                negative=20,        # Количество негативных примеров
                ns_exponent=0.75,   # Экспонента для negative sampling
                seed=self.config.random_state
            )
            
            graph = self._construct_enhanced_graph(model, tokens)
            
            if graph is None:
                return {}
            
            return self._extract_comprehensive_graph_features(graph)
        
        except Exception as e:
            logger.warning(f"Извлечение улучшенных графовых признаков не удалось: {e}")
            return {}
    
    def _create_enhanced_sentences(self, tokens: List[str]) -> List[List[str]]:
        """
        Создание улучшенных предложений для обучения Word2Vec.
        
        Args:
            tokens: Список токенов
            
        Returns:
            Список предложений для обучения
        """
        sentences = []
        
        # Метод 1: Скользящее окно с перекрытием
        window_size = 20
        step_size = 10
        for i in range(0, len(tokens) - window_size + 1, step_size):
            sentence = tokens[i:i + window_size]
            if len(sentence) >= 8:
                sentences.append(sentence)
        
        # Метод 2: Случайная выборка для разнообразия
        if len(tokens) > 100:
            for _ in range(len(tokens) // 15):
                start = np.random.randint(0, len(tokens) - 25)
                length = np.random.randint(12, 25)
                if start + length <= len(tokens):
                    sentences.append(tokens[start:start + length])
        
        # Метод 3: Окна семантической когерентности
        if len(tokens) > 200:
            chunk_size = len(tokens) // 10
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i:i + chunk_size]
                if len(chunk) >= 15:
                    sentences.append(chunk)
        
        return sentences if sentences else [tokens]
    
    def _construct_enhanced_graph(self, model: Word2Vec, tokens: List[str]) -> Optional[nx.Graph]:
        """
        Построение улучшенного семантического графа.
        
        Args:
            model: Обученная модель Word2Vec
            tokens: Список токенов
            
        Returns:
            Граф NetworkX или None при ошибке
        """
        word_counts = Counter(tokens)
        vocab = set(model.wv.index_to_key)
        
        # Улучшенный отбор ключевых слов
        core_candidates = [
            word for word, count in word_counts.most_common(50)
            if word in vocab and count >= 3
        ]
        
        # Отбор разнообразных ключевых слов
        core_words = []
        for word in core_candidates:
            if len(core_words) >= self.config.n_core_words:
                break
            # Проверка на разнообразие (избежание семантически похожих слов)
            if not core_words or all(model.wv.similarity(word, cw) < 0.8 for cw in core_words):
                core_words.append(word)
        
        if len(core_words) < 5:  # Недостаточно ключевых слов
            return None
        
        G = nx.Graph()
        
        # Добавление ключевых узлов с улучшенными атрибутами
        for word in core_words:
            freq = word_counts[word]
            rel_freq = freq / len(tokens)
            semantic_centrality = self._calculate_semantic_centrality(word, model, core_words)
            
            G.add_node(word, 
                      node_type='core',            # Тип узла
                      frequency=freq,              # Частота
                      relative_frequency=rel_freq, # Относительная частота
                      semantic_centrality=semantic_centrality, # Семантическая центральность
                      word_length=len(word))       # Длина слова
        
        # Добавление похожих узлов с улучшенным анализом схожести
        for core_word in core_words:
            try:
                similar_words = model.wv.most_similar(
                    core_word, 
                    topn=self.config.k_similar_words * 2
                )
                
                added_count = 0
                for similar_word, similarity in similar_words:
                    if (similar_word in vocab and 
                        similar_word not in core_words and 
                        similarity > 0.4 and
                        added_count < self.config.k_similar_words):
                        
                        if not G.has_node(similar_word):
                            sim_freq = word_counts.get(similar_word, 0)
                            G.add_node(similar_word,
                                      node_type='similar',
                                      frequency=sim_freq,
                                      relative_frequency=sim_freq / len(tokens),
                                      semantic_centrality=0.0,
                                      word_length=len(similar_word))
                        
                        # Добавление ребра с весом схожести
                        G.add_edge(core_word, similar_word, 
                                  weight=float(similarity),
                                  edge_type='semantic')
                        added_count += 1
            except:
                continue
        
        return G
    
    def _calculate_semantic_centrality(self, word: str, model: Word2Vec, core_words: List[str]) -> float:
        """
        Вычисление семантической центральности в ключевом словаре.
        
        Args:
            word: Слово для анализа
            model: Модель Word2Vec
            core_words: Список ключевых слов
            
        Returns:
            Значение семантической центральности
        """
        try:
            similarities = []
            for other_word in core_words:
                if other_word != word:
                    sim = model.wv.similarity(word, other_word)
                    similarities.append(sim)
            return np.mean(similarities) if similarities else 0.0
        except:
            return 0.0
    
    def _extract_comprehensive_graph_features(self, graph: nx.Graph) -> Dict[str, float]:
        """
        Извлечение комплексных графовых признаков.
        
        Args:
            graph: Граф NetworkX
            
        Returns:
            Словарь с графовыми признаками
        """
        features = {}
        
        # Базовая статистика графа
        features['total_nodes'] = graph.number_of_nodes()       # Общее количество узлов
        features['total_edges'] = graph.number_of_edges()       # Общее количество ребер
        features['graph_density'] = nx.density(graph)           # Плотность графа
        features['num_components'] = nx.number_connected_components(graph)  # Количество компонент связности
        
        # Статистика степеней узлов
        degrees = [d for n, d in graph.degree()]
        if degrees:
            features.update({
                'average_degree': np.mean(degrees),              # Средняя степень
                'max_degree': max(degrees),                      # Максимальная степень
                'min_degree': min(degrees),                      # Минимальная степень
                'degree_std': np.std(degrees),                   # Стандартное отклонение степеней
                'degree_variance': np.var(degrees),              # Дисперсия степеней
                'degree_skewness': stats.skew(degrees),          # Асимметрия распределения степеней
                'degree_kurtosis': stats.kurtosis(degrees),      # Эксцесс распределения степеней
                'degree_gini': self._calculate_gini(degrees)     # Коэффициент Джини для степеней
            })
        
        # Меры центральности
        try:
            # Центральность по посредничеству
            betweenness = list(nx.betweenness_centrality(graph).values())
            if betweenness:
                features.update({
                    'betweenness_mean': np.mean(betweenness),
                    'betweenness_max': np.max(betweenness),
                    'betweenness_std': np.std(betweenness),
                    'betweenness_entropy': self._calculate_entropy(betweenness)
                })
        except:
            features.update({
                'betweenness_mean': 0, 'betweenness_max': 0, 
                'betweenness_std': 0, 'betweenness_entropy': 0
            })
        
        try:
            # Центральность по близости
            closeness = list(nx.closeness_centrality(graph).values())
            if closeness:
                features.update({
                    'closeness_mean': np.mean(closeness),
                    'closeness_max': np.max(closeness),
                    'closeness_std': np.std(closeness)
                })
        except:
            features.update({'closeness_mean': 0, 'closeness_max': 0, 'closeness_std': 0})
        
        # Кластеризация и пути
        try:
            features['average_clustering'] = nx.average_clustering(graph)   # Средняя кластеризация
            features['transitivity'] = nx.transitivity(graph)               # Транзитивность
            
            if nx.is_connected(graph):
                features['diameter'] = nx.diameter(graph)                    # Диаметр
                features['radius'] = nx.radius(graph)                        # Радиус
                features['average_path_length'] = nx.average_shortest_path_length(graph)  # Средняя длина пути
                features['assortativity'] = nx.degree_assortativity_coefficient(graph)    # Ассортативность
            else:
                # Работа с наибольшей компонентой связности
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                try:
                    features['diameter'] = nx.diameter(subgraph)
                    features['radius'] = nx.radius(subgraph)
                    features['average_path_length'] = nx.average_shortest_path_length(subgraph)
                    features['assortativity'] = nx.degree_assortativity_coefficient(subgraph)
                except:
                    features.update({
                        'diameter': 0, 'radius': 0, 'average_path_length': 0, 'assortativity': 0
                    })
        except:
            features.update({
                'average_clustering': 0, 'transitivity': 0, 'diameter': 0, 
                'radius': 0, 'average_path_length': 0, 'assortativity': 0
            })
        
        # Статистика весов ребер
        edge_weights = [d['weight'] for u, v, d in graph.edges(data=True) if 'weight' in d]
        if edge_weights:
            features.update({
                'edge_weight_mean': np.mean(edge_weights),       # Средний вес ребер
                'edge_weight_std': np.std(edge_weights),         # СКО весов ребер
                'edge_weight_min': np.min(edge_weights),         # Минимальный вес
                'edge_weight_max': np.max(edge_weights),         # Максимальный вес
                'edge_weight_range': np.max(edge_weights) - np.min(edge_weights),  # Диапазон весов
                'edge_weight_skewness': stats.skew(edge_weights),     # Асимметрия весов
                'edge_weight_kurtosis': stats.kurtosis(edge_weights)  # Эксцесс весов
            })
        
        # Статистика атрибутов узлов
        frequencies = [data.get('relative_frequency', 0) for node, data in graph.nodes(data=True)]
        if frequencies:
            features.update({
                'node_freq_mean': np.mean(frequencies),          # Средняя частота узлов
                'node_freq_std': np.std(frequencies),            # СКО частот узлов
                'node_freq_max': np.max(frequencies),            # Максимальная частота
                'node_freq_cv': np.std(frequencies) / np.mean(frequencies) if np.mean(frequencies) > 0 else 0  # Коэффициент вариации
            })
        
        # Статистика семантической центральности
        semantic_centralities = [
            data.get('semantic_centrality', 0) 
            for node, data in graph.nodes(data=True)
            if data.get('node_type') == 'core'
        ]
        if semantic_centralities:
            features.update({
                'semantic_centrality_mean': np.mean(semantic_centralities),    # Средняя семантическая центральность
                'semantic_centrality_std': np.std(semantic_centralities),      # СКО семантической центральности
                'semantic_coherence': np.mean(semantic_centralities) / (1 + np.std(semantic_centralities))  # Семантическая когерентность
            })
        
        return features
    
    def _extract_lexical_features(self, text: str) -> Dict[str, float]:
        """
        Извлечение комплексных лексических признаков разнообразия и сложности.
        
        Args:
            text: Входной текст
            
        Returns:
            Словарь с лексическими признаками
        """
        features = {}
        
        # Базовая статистика текста
        features['text_length'] = len(text)                          # Длина текста
        features['char_count'] = len(text.replace(' ', ''))          # Количество символов без пробелов
        
        # Токенизация
        tokens = self.tokenizer.preprocess_text(text, remove_stopwords=False)  # Все токены
        words = self.tokenizer.preprocess_text(text, remove_stopwords=True)    # Только значимые слова
        
        if not tokens:
            return features
        
        # Базовые подсчеты
        features['word_count'] = len(words)                            # Количество слов
        features['unique_word_count'] = len(set(words))                # Количество уникальных слов
        features['sentence_count'] = len(re.findall(r'[.!?]+', text))  # Количество предложений
        
        # Меры лексического разнообразия
        if len(words) > 0:
            features['type_token_ratio'] = len(set(words)) / len(words)  # Отношение типов к токенам
            features['log_ttr'] = np.log(len(set(words))) / np.log(len(words)) if len(words) > 1 else 0  # Логарифмическое TTR
        
        # Hapax legomena (слова, встречающиеся только один раз)
        word_counts = Counter(words)
        hapax_count = sum(1 for count in word_counts.values() if count == 1)
        features['hapax_legomena_ratio'] = hapax_count / len(words) if len(words) > 0 else 0
        
        # Средняя длина слова
        if words:
            word_lengths = [len(word) for word in words]
            features['avg_word_length'] = np.mean(word_lengths)       # Средняя длина
            features['word_length_std'] = np.std(word_lengths)        # СКО длины слов
            features['max_word_length'] = max(word_lengths)           # Максимальная длина
            features['min_word_length'] = min(word_lengths)           # Минимальная длина
        
        # Статистика длины предложений
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
        if sentence_lengths:
            features['avg_sentence_length'] = np.mean(sentence_lengths)    # Средняя длина предложения
            features['sentence_length_std'] = np.std(sentence_lengths)     # СКО длины предложений
            features['max_sentence_length'] = max(sentence_lengths)        # Максимальная длина предложения
            features['min_sentence_length'] = min(sentence_lengths)        # Минимальная длина предложения
        
        # Богатство словаря
        if len(words) > 0:
            features['vocabulary_richness'] = len(set(words)) / np.sqrt(len(words))  # Богатство словаря
        
        # Доля функциональных слов
        function_words = {'в', 'на', 'с', 'и', 'а', 'но', 'что', 'как', 'это', 'то'}
        function_word_count = sum(1 for word in tokens if word in function_words)
        features['function_word_ratio'] = function_word_count / len(tokens) if len(tokens) > 0 else 0
        
        return features
    
    def _extract_syntactic_features(self, text: str) -> Dict[str, float]:
        """
        Извлечение признаков синтаксической сложности.
        
        Args:
            text: Входной текст
            
        Returns:
            Словарь с синтаксическими признаками
        """
        features = {}
        
        # Анализ пунктуации
        punctuation_chars = set('.,;:!?-()[]{}«»""''')
        punct_count = sum(1 for char in text if char in punctuation_chars)
        features['punctuation_ratio'] = punct_count / len(text) if len(text) > 0 else 0
        
        # Специфические соотношения пунктуации
        features['comma_ratio'] = text.count(',') / len(text) if len(text) > 0 else 0         # Запятые
        features['semicolon_ratio'] = text.count(';') / len(text) if len(text) > 0 else 0     # Точки с запятой
        features['colon_ratio'] = text.count(':') / len(text) if len(text) > 0 else 0         # Двоеточия
        features['question_ratio'] = text.count('?') / len(text) if len(text) > 0 else 0      # Вопросительные знаки
        features['exclamation_ratio'] = text.count('!') / len(text) if len(text) > 0 else 0   # Восклицательные знаки
        
        # Кавычки
        features['quote_ratio'] = (text.count('"') + text.count('«') + text.count('»')) / len(text) if len(text) > 0 else 0
        
        # Скобки
        features['parentheses_ratio'] = (text.count('(') + text.count(')')) / len(text) if len(text) > 0 else 0
        
        # Паттерны заглавных букв
        features['capital_ratio'] = sum(1 for char in text if char.isupper()) / len(text) if len(text) > 0 else 0
        
        # Соотношение цифр
        features['digit_ratio'] = sum(1 for char in text if char.isdigit()) / len(text) if len(text) > 0 else 0
        
        return features
    
    def _extract_readability_features(self, text: str) -> Dict[str, float]:
        """
        Извлечение признаков читаемости и сложности.
        
        Args:
            text: Входной текст
            
        Returns:
            Словарь с признаками читаемости
        """
        features = {}
        
        try:
            # Базовые оценки читаемости (readability)
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)           # Индекс Флеша
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)         # Уровень Флеша-Кинкайда
            features['automated_readability_index'] = textstat.automated_readability_index(text)  # Автоматический индекс читаемости
            features['coleman_liau_index'] = textstat.coleman_liau_index(text)             # Индекс Колмана-Лиау
            features['gunning_fog'] = textstat.gunning_fog(text)                           # Индекс тумана Ганнинга
            features['smog_index'] = textstat.smog_index(text)                             # SMOG индекс
            
            # Меры слогов и сложности
            features['avg_syllables_per_word'] = textstat.avg_syllables_per_word(text)    # Средние слоги на слово
            features['syllable_count'] = textstat.syllable_count(text)                    # Общее количество слогов
            features['lexicon_count'] = textstat.lexicon_count(text)                      # Количество слов в лексиконе
            features['sentence_count_textstat'] = textstat.sentence_count(text)           # Количество предложений
            
            # Продвинутые меры
            features['difficult_words'] = textstat.difficult_words(text)                  # Трудные слова
            features['difficult_words_ratio'] = textstat.difficult_words(text) / max(textstat.lexicon_count(text), 1)  # Доля трудных слов
            
        except Exception as e:
            logger.debug(f"Признаки читаемости не удались: {e}")
            # Возврат значений по умолчанию
            features.update({
                'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0,
                'automated_readability_index': 0, 'coleman_liau_index': 0,
                'gunning_fog': 0, 'smog_index': 0,
                'avg_syllables_per_word': 0, 'syllable_count': 0,
                'lexicon_count': 0, 'sentence_count_textstat': 0,
                'difficult_words': 0, 'difficult_words_ratio': 0
            })
        
        return features
    
    def _calculate_gini(self, values: List[float]) -> float:
        """
        Вычисление коэффициента Джини.
        
        Args:
            values: Список значений
            
        Returns:
            Коэффициент Джини
        """
        if not values or len(values) < 2:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumulative_sum = np.cumsum(sorted_values)
        
        try:
            return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumulative_sum[-1]) - (n + 1) / n
        except:
            return 0.0
    
    def _calculate_entropy(self, values: List[float]) -> float:
        """
        Вычисление энтропии распределения.
        
        Args:
            values: Список значений
            
        Returns:
            Энтропия
        """
        if not values:
            return 0.0
        
        total = sum(values)
        if total == 0:
            return 0.0
        
        probs = [v / total for v in values if v > 0]
        return -sum(p * np.log2(p) for p in probs if p > 0)

# ===== КЛАСС ВАЛИДАЦИИ =====
class EnhancedValidator:
    """Улучшенная статистическая валидация с ансамблевыми методами и bootstrap-анализом."""
    
    def __init__(self, config: EnhancedConfig):
        """
        Инициализация валидатора.
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
    
    def validate_enhanced(self, features_df: pd.DataFrame) -> Dict:
        """
        Выполнение улучшенной валидации с комплексным анализом.
        
        Args:
            features_df: DataFrame с признаками
            
        Returns:
            Словарь с результатами валидации
        """
        results = {}
        
        # Подготовка данных
        feature_cols = [col for col in features_df.columns 
                       if col not in ['label', 'year', 'doc_id', 'text_length', 'original_text_length']]
        X = features_df[feature_cols].fillna(0)  # Заполнение пропусков нулями
        y = features_df['label']
        
        logger.info(f"Размер матрицы признаков: {X.shape}")
        logger.info(f"Доступно признаков: {len(feature_cols)}")
        
        # Отбор признаков
        if self.config.use_feature_selection and len(feature_cols) > self.config.n_feature_selection:
            logger.info("Выполняется отбор признаков...")
            X_selected, selected_features = self._select_features(X, y, feature_cols)
            results['selected_features'] = selected_features
            logger.info(f"Отобрано {len(selected_features)} признаков")
        else:
            X_selected = X
            results['selected_features'] = feature_cols
        
        # Улучшенная кросс-валидация
        results['classification'] = self._cross_validate_enhanced(X_selected, y)
        
        # Статистические тесты
        results['statistical_tests'] = self._enhanced_statistical_tests(X_selected, y)
        
        # Размеры эффектов
        results['effect_sizes'] = self._enhanced_effect_sizes(X_selected, y)
        
        # Важность признаков
        results['feature_importance'] = self._analyze_feature_importance(X_selected, y, results['selected_features'])
        
        # Bootstrap-анализ
        results['bootstrap_results'] = self._bootstrap_analysis(X_selected, y)
        
        return results
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Отбор лучших признаков с использованием множественных методов.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            feature_names: Названия признаков
            
        Returns:
            Отобранные признаки и их названия
        """
        
        # Метод 1: Univariate отбор
        selector_univariate = SelectKBest(score_func=f_classif, k=min(self.config.n_feature_selection, len(feature_names)))
        X_univariate = selector_univariate.fit_transform(X, y)
        univariate_features = [feature_names[i] for i in selector_univariate.get_support(indices=True)]
        
        # Метод 2: Рекурсивное исключение признаков с Random Forest
        rf_selector = RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)
        rfe_selector = RFE(rf_selector, n_features_to_select=min(self.config.n_feature_selection, len(feature_names)))
        rfe_selector.fit(X, y)
        rfe_features = [feature_names[i] for i in range(len(feature_names)) if rfe_selector.support_[i]]
        
        # Метод 3: Важность признаков из Random Forest
        rf_importance = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
        rf_importance.fit(X, y)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_importance.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_features = feature_importance.head(min(self.config.n_feature_selection, len(feature_names)))['feature'].tolist()
        
        # Объединение и отбор финальных признаков с улучшенным отбором
        all_selected = set(univariate_features + rfe_features + importance_features)
        
        # Дополнительный отбор на основе размеров эффектов
        effect_scores = {}
        train_data = X[y == 'train']
        test_data = X[y == 'test']
        
        for feature in all_selected:
            try:
                train_vals = train_data[feature].dropna()
                test_vals = test_data[feature].dropna()
                if len(train_vals) > 1 and len(test_vals) > 1:
                    # Вычисление Cohen's d
                    mean_diff = abs(train_vals.mean() - test_vals.mean())
                    pooled_std = np.sqrt(((len(train_vals)-1)*train_vals.var() + (len(test_vals)-1)*test_vals.var()) / (len(train_vals)+len(test_vals)-2))
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    effect_scores[feature] = cohens_d
            except:
                effect_scores[feature] = 0
        
        # Сортировка по размеру эффекта и взятие топ признаков
        sorted_by_effect = sorted(effect_scores.items(), key=lambda x: x[1], reverse=True)
        final_features = [feat for feat, _ in sorted_by_effect[:self.config.n_feature_selection]]
        
        return X[final_features], final_features
    
    def _cross_validate_enhanced(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Кросс-валидация с улучшенными классификаторами, включая ансамблевые методы.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Результаты кросс-валидации
        """
        
        # Определение ультра-улучшенных классификаторов для цели >75%
        classifiers = {
            'RandomForest_Ultra': RandomForestClassifier(
                n_estimators=500,       # Значительно увеличено
                max_depth=25,           # Более глубокие деревья
                min_samples_split=2,    # Более агрессивное разделение
                min_samples_leaf=1,     # Более детальные листья
                max_features='log2',    # Другой отбор признаков
                bootstrap=True,
                oob_score=True,
                random_state=self.config.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'ExtraTreesClassifier': ExtraTreesClassifier(
                n_estimators=400,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=False,
                random_state=self.config.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'GradientBoosting_Ultra': GradientBoostingClassifier(
                n_estimators=300,      # Гораздо больше оценщиков
                max_depth=15,          # Глубже
                learning_rate=0.02,    # Медленнее обучение
                subsample=0.7,         # Больше регуляризации
                max_features='sqrt',   # Субсэмплинг признаков
                random_state=self.config.random_state
            ),
            'LogisticRegression_Ultra': LogisticRegression(
                random_state=self.config.random_state,
                class_weight='balanced',
                max_iter=5000,         # Гораздо больше итераций
                C=0.01,                # Очень сильная регуляризация
                solver='saga',         # Лучший решатель
                penalty='elasticnet',  # Elastic net штраф
                l1_ratio=0.5           # Баланс между L1 и L2
            ),
            'SVM_Ultra': SVC(
                kernel='rbf',
                C=100,                 # Гораздо выше C
                gamma='auto',          # Авто гамма
                class_weight='balanced',
                probability=True,
                random_state=self.config.random_state
            )
        }
        
        # Добавление XGBoost с ультра параметрами
        if XGBOOST_AVAILABLE:
            classifiers['XGBoost_Ultra'] = xgb.XGBClassifier(
                n_estimators=400,      # Гораздо больше оценщиков
                max_depth=12,          # Более глубокие деревья
                learning_rate=0.02,    # Гораздо медленнее обучение
                subsample=0.7,         # Больше регуляризации
                colsample_bytree=0.7,  # Субсэмплинг признаков
                reg_alpha=0.1,         # L1 регуляризация
                reg_lambda=0.1,        # L2 регуляризация
                scale_pos_weight=1,    # Обработка дисбаланса классов
                random_state=self.config.random_state,
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1
            )
        
        results = {}
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                            random_state=self.config.random_state)
        
        # Масштабирование признаков
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Кодирование меток для совместимости с XGBoost
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Индивидуальные классификаторы
        individual_results = {}
        for name, clf in classifiers.items():
            try:
                # Использование кодированных меток для XGBoost, оригинальных для остальных
                target_y = y_encoded if 'XGBoost' in name else y
                scores = cross_val_score(clf, X_scaled, target_y, cv=cv, scoring='accuracy')
                individual_results[name] = {
                    'mean_accuracy': np.mean(scores),
                    'std_accuracy': np.std(scores),
                    'min_accuracy': np.min(scores),
                    'max_accuracy': np.max(scores),
                    'scores': scores.tolist()
                }
                logger.info(f"   {name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            except Exception as e:
                individual_results[name] = {'error': str(e)}
                logger.warning(f"   {name}: ОШИБКА - {e}")
        
        results['individual'] = individual_results
        
        # Ансамблевые методы
        if self.config.use_ensemble:
            ensemble_results = {}
            
            # Ультра Voting Classifier (Мягкое голосование)
            try:
                voting_estimators = [
                    ('rf_ultra', classifiers['RandomForest_Ultra']),
                    ('et', classifiers['ExtraTreesClassifier']),
                    ('gb_ultra', classifiers['GradientBoosting_Ultra'])
                ]
                
                voting_clf = VotingClassifier(
                    estimators=voting_estimators,
                    voting='soft'
                )
                
                voting_scores = cross_val_score(voting_clf, X_scaled, y, cv=cv, scoring='accuracy')
                ensemble_results['VotingClassifier_Ultra_Soft'] = {
                    'mean_accuracy': np.mean(voting_scores),
                    'std_accuracy': np.std(voting_scores),
                    'scores': voting_scores.tolist()
                }
                logger.info(f"   VotingClassifier_Ultra_Soft: {np.mean(voting_scores):.3f} ± {np.std(voting_scores):.3f}")
            except Exception as e:
                ensemble_results['VotingClassifier_Ultra_Soft'] = {'error': str(e)}
            
            # Ультра Voting Classifier (Жесткое голосование)
            try:
                voting_estimators_hard = [
                    ('rf_ultra', classifiers['RandomForest_Ultra']),
                    ('et', classifiers['ExtraTreesClassifier']),
                    ('svm_ultra', classifiers['SVM_Ultra'])
                ]
                
                voting_clf_hard = VotingClassifier(
                    estimators=voting_estimators_hard,
                    voting='hard'
                )
                
                voting_scores_hard = cross_val_score(voting_clf_hard, X_scaled, y, cv=cv, scoring='accuracy')
                ensemble_results['VotingClassifier_Ultra_Hard'] = {
                    'mean_accuracy': np.mean(voting_scores_hard),
                    'std_accuracy': np.std(voting_scores_hard),
                    'scores': voting_scores_hard.tolist()
                }
                logger.info(f"   VotingClassifier_Ultra_Hard: {np.mean(voting_scores_hard):.3f} ± {np.std(voting_scores_hard):.3f}")
            except Exception as e:
                ensemble_results['VotingClassifier_Ultra_Hard'] = {'error': str(e)}
            
            # Ультра Stacking Classifier (Продвинутый ансамбль)
            if self.config.use_stacking:
                try:
                    from sklearn.ensemble import StackingClassifier
                    
                    base_estimators = [
                        ('rf_ultra', classifiers['RandomForest_Ultra']),
                        ('et', classifiers['ExtraTreesClassifier']),
                        ('gb_ultra', classifiers['GradientBoosting_Ultra']),
                        ('lr_ultra', classifiers['LogisticRegression_Ultra'])
                    ]
                    
                    # Использование XGBoost как мета-ученика, если доступен
                    if XGBOOST_AVAILABLE:
                        meta_learner = xgb.XGBClassifier(
                            n_estimators=50,
                            max_depth=3,
                            learning_rate=0.1,
                            random_state=self.config.random_state,
                            eval_metric='logloss',
                            use_label_encoder=False
                        )
                    else:
                        meta_learner = LogisticRegression(
                            random_state=self.config.random_state,
                            max_iter=1000
                        )
                    
                    stacking_clf = StackingClassifier(
                        estimators=base_estimators,
                        final_estimator=meta_learner,
                        cv=5,  # Больше CV фолдов для стекинга
                        n_jobs=-1,
                        passthrough=True  # Включить оригинальные признаки
                    )
                    
                    stacking_scores = cross_val_score(stacking_clf, X_scaled, y, cv=cv, scoring='accuracy')
                    ensemble_results['StackingClassifier_Ultra'] = {
                        'mean_accuracy': np.mean(stacking_scores),
                        'std_accuracy': np.std(stacking_scores),
                        'scores': stacking_scores.tolist()
                    }
                    logger.info(f"   StackingClassifier_Ultra: {np.mean(stacking_scores):.3f} ± {np.std(stacking_scores):.3f}")
                except Exception as e:
                    ensemble_results['StackingClassifier_Ultra'] = {'error': str(e)}
            
            results['ensemble'] = ensemble_results
        
        return results
    
    def _enhanced_statistical_tests(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Выполнение улучшенных статистических тестов.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Результаты статистических тестов
        """
        train_data = X[y == 'train']
        test_data = X[y == 'test']
        
        tests = {}
        significant_features = []
        
        for col in X.columns:
            try:
                train_vals = train_data[col].dropna()
                test_vals = test_data[col].dropna()
                
                if len(train_vals) > 2 and len(test_vals) > 2:
                    # t-тест
                    t_stat, t_p = stats.ttest_ind(train_vals, test_vals)
                    
                    # Тест Манна-Уитни U
                    u_stat, u_p = stats.mannwhitneyu(
                        train_vals, test_vals, alternative='two-sided'
                    )
                    
                    # Тест Колмогорова-Смирнова
                    ks_stat, ks_p = stats.ks_2samp(train_vals, test_vals)
                    
                    # Объединенное p-значение (минимальное)
                    min_p = min(t_p, u_p, ks_p)
                    
                    tests[col] = {
                        't_pvalue': t_p,
                        'mannwhitney_p': u_p,
                        'ks_p': ks_p,
                        'min_pvalue': min_p,
                        'significant': min_p < 0.05,
                        't_statistic': t_stat,
                        'u_statistic': u_stat,
                        'ks_statistic': ks_stat
                    }
                    
                    if min_p < 0.05:
                        significant_features.append(col)
                        
            except Exception as e:
                tests[col] = {'error': str(e)}
        
        tests['summary'] = {
            'total_features': len(X.columns),
            'significant_features': len(significant_features),
            'significant_ratio': len(significant_features) / len(X.columns)
        }
        
        return tests
    
    def _enhanced_effect_sizes(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Вычисление улучшенных размеров эффектов.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Размеры эффектов
        """
        train_data = X[y == 'train']
        test_data = X[y == 'test']
        
        effect_sizes = {}
        large_effects = []
        
        for col in X.columns:
            try:
                train_vals = train_data[col].dropna()
                test_vals = test_data[col].dropna()
                
                if len(train_vals) > 1 and len(test_vals) > 1:
                    # Cohen's d
                    mean1, mean2 = train_vals.mean(), test_vals.mean()
                    std1, std2 = train_vals.std(), test_vals.std()
                    n1, n2 = len(train_vals), len(test_vals)
                    
                    # Объединенное стандартное отклонение
                    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                    
                    if pooled_std > 0:
                        cohens_d = (mean1 - mean2) / pooled_std
                        effect_sizes[col] = {
                            'cohens_d': cohens_d,
                            'abs_cohens_d': abs(cohens_d),
                            'effect_size_category': self._categorize_effect_size(abs(cohens_d))
                        }
                        
                        if abs(cohens_d) > 0.8:
                            large_effects.append(col)
                    
            except Exception as e:
                effect_sizes[col] = {'error': str(e)}
        
        # Сводная статистика
        valid_effects = [data['abs_cohens_d'] for data in effect_sizes.values() 
                        if isinstance(data, dict) and 'abs_cohens_d' in data]
        
        if valid_effects:
            effect_sizes['summary'] = {
                'mean_effect_size': np.mean(valid_effects),
                'max_effect_size': np.max(valid_effects),
                'large_effects_count': len(large_effects),
                'large_effects_ratio': len(large_effects) / len(valid_effects),
                'large_effect_features': large_effects
            }
        
        return effect_sizes
    
    def _categorize_effect_size(self, effect_size: float) -> str:
        """
        Категоризация размера эффекта согласно соглашениям Коэна.
        
        Args:
            effect_size: Размер эффекта
            
        Returns:
            Категория эффекта
        """
        if effect_size < 0.2:
            return 'negligible'      # незначительный
        elif effect_size < 0.5:
            return 'small'           # малый
        elif effect_size < 0.8:
            return 'medium'          # средний
        else:
            return 'large'           # большой
    
    def _analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Dict:
        """
        Анализ важности признаков с использованием множественных методов.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            feature_names: Названия признаков
            
        Returns:
            Результаты анализа важности
        """
        importance_results = {}
        
        # Важность признаков Random Forest
        try:
            rf = RandomForestClassifier(n_estimators=200, random_state=self.config.random_state)
            rf.fit(X, y)
            
            rf_importance = pd.DataFrame({
                'feature': feature_names,
                'rf_importance': rf.feature_importances_
            }).sort_values('rf_importance', ascending=False)
            
            importance_results['random_forest'] = rf_importance.to_dict('records')
            importance_results['top_10_rf'] = rf_importance.head(10)['feature'].tolist()
            
        except Exception as e:
            importance_results['random_forest'] = {'error': str(e)}
        
        # Важность признаков Gradient Boosting
        try:
            gb = GradientBoostingClassifier(n_estimators=150, random_state=self.config.random_state)
            gb.fit(X, y)
            
            gb_importance = pd.DataFrame({
                'feature': feature_names,
                'gb_importance': gb.feature_importances_
            }).sort_values('gb_importance', ascending=False)
            
            importance_results['gradient_boosting'] = gb_importance.to_dict('records')
            importance_results['top_10_gb'] = gb_importance.head(10)['feature'].tolist()
            
        except Exception as e:
            importance_results['gradient_boosting'] = {'error': str(e)}
        
        return importance_results
    
    def _bootstrap_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Bootstrap-анализ для надежных доверительных интервалов.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Результаты bootstrap-анализа
        """
        bootstrap_results = {}
        
        try:
            # Bootstrap точности классификации
            def bootstrap_accuracy(X_sample, y_sample):
                try:
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    cv_scores = cross_val_score(rf, X_sample, y_sample, cv=3)
                    return np.mean(cv_scores)
                except:
                    return 0.5
            
            # Подготовка данных для bootstrap
            X_array = X.values
            y_array = y.values
            
            # Bootstrap выборка
            bootstrap_accuracies = []
            for i in range(min(100, self.config.bootstrap_iterations)):
                # Выборка с возвращением
                n_samples = len(X_array)
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                
                X_boot = X_array[indices]
                y_boot = y_array[indices]
                
                # Обеспечение представленности обоих классов
                if len(np.unique(y_boot)) > 1:
                    accuracy = bootstrap_accuracy(X_boot, y_boot)
                    bootstrap_accuracies.append(accuracy)
            
            if bootstrap_accuracies:
                bootstrap_results['classification_accuracy'] = {
                    'mean': np.mean(bootstrap_accuracies),
                    'std': np.std(bootstrap_accuracies),
                    'confidence_interval_95': np.percentile(bootstrap_accuracies, [2.5, 97.5])
                }
        
        except Exception as e:
            logger.warning(f"Bootstrap-анализ не удался: {e}")
            bootstrap_results['error'] = str(e)
        
        return bootstrap_results

# ===== ИСПРАВЛЕННЫЙ КЛАСС ВИЗУАЛИЗАЦИИ =====
class EnhancedVisualizerFixed:
    """Исправленная визуализация - полная реализация всех методов."""
    
    def __init__(self, config: EnhancedConfig):
        """
        Инициализация визуализатора.
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
    
    def create_enhanced_visualizations(self, features_df: pd.DataFrame, 
                                     validation_results: Dict, title: str):
        """
        Создание улучшенных визуализаций с полной реализацией всех графиков.
        
        Args:
            features_df: DataFrame с признаками
            validation_results: Результаты валидации
            title: Заголовок для графиков
        """
        plt.rcParams['figure.figsize'] = (20, 16)
        plt.rcParams['font.size'] = 10
        
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle(f'{title}\nУлучшенная Временная Верификация Авторства', 
                    fontsize=18, fontweight='bold')
        
        # Все 9 графиков с полной реализацией
        self._plot_enhanced_pca(features_df, axes[0, 0])
        self._plot_enhanced_classification(validation_results, axes[0, 1])
        self._plot_feature_importance_comparison(validation_results, axes[0, 2])
        self._plot_effect_sizes_distribution(validation_results, axes[1, 0])
        self._plot_statistical_significance(validation_results, axes[1, 1])
        self._plot_enhanced_temporal_evolution(features_df, axes[1, 2])
        self._plot_feature_correlation(features_df, axes[2, 0])
        self._plot_model_performance_comparison(validation_results, axes[2, 1])
        self._plot_bootstrap_results(validation_results, axes[2, 2])
        
        plt.tight_layout()
        plt.show()
        
        # Сохранение графика с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_analysis_{timestamp}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 График сохранен как: {filename}")
    
    def _plot_enhanced_pca(self, features_df: pd.DataFrame, ax):
        """
        График PCA с улучшенным анализом главных компонент.
        
        Args:
            features_df: DataFrame с признаками
            ax: Объект axes для рисования
        """
        try:
            feature_cols = [col for col in features_df.columns 
                           if col not in ['label', 'year', 'doc_id', 'original_text_length']]
            X = features_df[feature_cols].fillna(0)
            
            if len(X) > 0 and X.shape[1] > 2:
                # Стандартизация данных
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # PCA анализ
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                
                # Цветовая схема для разных лет
                colors = {'1904': '#FF6B6B', '1905': '#4ECDC4', '1906': '#FFE66D', 
                         '1907': '#95E1D3', '1908': '#A8E6CF'}
                
                # Построение точек для каждого года
                for year in features_df['year'].unique():
                    mask = features_df['year'] == year
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              c=colors.get(str(year), 'gray'), 
                              label=f'{year}', alpha=0.8, s=80, 
                              edgecolors='black', linewidth=1)
                
                # Добавление эллипсов для визуализации групп
                for year in features_df['year'].unique():
                    mask = features_df['year'] == year
                    if np.sum(mask) > 2:  # Нужно минимум 3 точки для эллипса
                        data_points = X_pca[mask]
                        try:
                            # Вычисление эллипса доверия
                            mean = np.mean(data_points, axis=0)
                            cov = np.cov(data_points.T)
                            eigenvals, eigenvecs = np.linalg.eigh(cov)
                            order = eigenvals.argsort()[::-1]
                            eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
                            
                            angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
                            width, height = 2 * np.sqrt(eigenvals)
                            
                            ellipse = Ellipse(xy=mean, width=width, height=height, 
                                            angle=angle, alpha=0.2, 
                                            color=colors.get(str(year), 'gray'))
                            ax.add_patch(ellipse)
                        except:
                            pass
                
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)')
                ax.set_title('PCA: Комплексные Признаки')
                ax.legend()
                ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'PCA ошибка: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_enhanced_classification(self, validation_results: Dict, ax):
        """
        График производительности классификации с улучшенной визуализацией.
        
        Args:
            validation_results: Результаты валидации
            ax: Объект axes для рисования
        """
        try:
            individual = validation_results.get('classification', {}).get('individual', {})
            ensemble = validation_results.get('classification', {}).get('ensemble', {})
            all_results = {**individual, **ensemble}
            
            if all_results:
                classifiers = list(all_results.keys())
                accuracies = [all_results[clf].get('mean_accuracy', 0) for clf in classifiers]
                stds = [all_results[clf].get('std_accuracy', 0) for clf in classifiers]
                
                # Цветовое кодирование по производительности
                colors = ['#2ECC71' if acc > 0.75 else '#F39C12' if acc > 0.65 else '#E74C3C' 
                         for acc in accuracies]
                
                # Построение столбчатой диаграммы
                bars = ax.bar(range(len(classifiers)), accuracies, yerr=stds, 
                             capsize=5, alpha=0.8, color=colors)
                
                # Линии целей
                ax.axhline(y=0.75, color='green', linestyle='--', alpha=0.7, label='Цель (75%)')
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Случайный')
                
                # Настройка осей
                ax.set_xticks(range(len(classifiers)))
                ax.set_xticklabels([name.replace('_', '\n')[:15] for name in classifiers], 
                                  rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Точность')
                ax.set_title('Производительность Классификации')
                ax.set_ylim(0, 1)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Подписи значений на столбцах
                for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f'Классификация ошибка: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_feature_importance_comparison(self, validation_results: Dict, ax):
        """
        График сравнения важности признаков.
        
        Args:
            validation_results: Результаты валидации
            ax: Объект axes для рисования
        """
        try:
            importance = validation_results.get('feature_importance', {})
            if 'top_10_rf' in importance:
                rf_features = importance['top_10_rf'][:8]  # Топ 8 для читаемости
                
                # Получение важности из Random Forest
                rf_data = importance.get('random_forest', [])
                if isinstance(rf_data, list) and rf_data:
                    feature_scores = {item['feature']: item['rf_importance'] for item in rf_data}
                    scores = [feature_scores.get(f, 1) for f in rf_features]
                else:
                    scores = [1] * len(rf_features)
                
                y_pos = np.arange(len(rf_features))
                bars = ax.barh(y_pos, scores, alpha=0.7, color='#3498DB')
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels([f.replace('_', ' ')[:20] for f in rf_features])
                ax.set_xlabel('Важность признака')
                ax.set_title('Топ Важных Признаков')
                ax.grid(True, alpha=0.3)
                
                # Подписи значений
                for i, (bar, score) in enumerate(zip(bars, scores)):
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                           f'{score:.3f}', ha='left', va='center', fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f'Важность ошибка: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_effect_sizes_distribution(self, validation_results: Dict, ax):
        """
        График распределения размеров эффектов.
        
        Args:
            validation_results: Результаты валидации
            ax: Объект axes для рисования
        """
        try:
            effect_sizes = validation_results.get('effect_sizes', {})
            valid_effects = [data['abs_cohens_d'] for data in effect_sizes.values() 
                            if isinstance(data, dict) and 'abs_cohens_d' in data]
            
            if valid_effects:
                # Построение гистограммы
                n, bins, patches = ax.hist(valid_effects, bins=15, alpha=0.7, 
                                         color='skyblue', edgecolor='black')
                
                # Линии для категорий размеров эффектов
                ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Малый (0.2)')
                ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Средний (0.5)')
                ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.9, label='Большой (0.8)')
                
                # Цветовое кодирование гистограммы
                for i, (patch, bin_center) in enumerate(zip(patches, (bins[:-1] + bins[1:]) / 2)):
                    if bin_center >= 0.8:
                        patch.set_facecolor('#E74C3C')  # Красный для больших эффектов
                    elif bin_center >= 0.5:
                        patch.set_facecolor('#F39C12')  # Оранжевый для средних
                    elif bin_center >= 0.2:
                        patch.set_facecolor('#3498DB')  # Синий для малых
                    else:
                        patch.set_facecolor('#95A5A6')  # Серый для незначительных
                
                ax.set_xlabel('Размер эффекта (Cohen\'s d)')
                ax.set_ylabel('Количество признаков')
                ax.set_title('Распределение Размеров Эффектов')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Статистика в заголовке
                mean_effect = np.mean(valid_effects)
                large_count = sum(1 for e in valid_effects if e >= 0.8)
                ax.text(0.02, 0.98, f'Средний: {mean_effect:.3f}\nБольших: {large_count}', 
                       transform=ax.transAxes, va='top', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        except Exception as e:
            ax.text(0.5, 0.5, f'Эффекты ошибка: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_statistical_significance(self, validation_results: Dict, ax):
        """
        График статистической значимости признаков.
        
        Args:
            validation_results: Результаты валидации
            ax: Объект axes для рисования
        """
        try:
            tests = validation_results.get('statistical_tests', {})
            if tests:
                test_types = ['t_pvalue', 'mannwhitney_p', 'ks_p', 'min_pvalue']
                test_names = ['t-тест', 'Mann-Whitney', 'K-S', 'Минимальное']
                
                # Подсчет значимых признаков для каждого теста
                counts = []
                for test_type in test_types:
                    count = sum(1 for key, data in tests.items() 
                               if isinstance(data, dict) and data.get(test_type, 1) < 0.05 and key != 'summary')
                    counts.append(count)
                
                # Построение столбчатой диаграммы
                colors = ['#E74C3C', '#F39C12', '#9B59B6', '#2ECC71']
                bars = ax.bar(test_names, counts, alpha=0.7, color=colors)
                
                ax.set_ylabel('Количество значимых признаков')
                ax.set_title('Статистическая Значимость (p < 0.05)')
                ax.grid(True, alpha=0.3)
                
                # Подписи значений
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           str(count), ha='center', va='bottom', fontweight='bold')
                
                # Общая статистика
                summary = tests.get('summary', {})
                total = summary.get('total_features', 0)
                significant = summary.get('significant_features', 0)
                if total > 0:
                    ratio = significant / total
                    ax.text(0.02, 0.98, f'Всего: {total}\nЗначимых: {significant}\nДоля: {ratio:.1%}', 
                           transform=ax.transAxes, va='top', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        except Exception as e:
            ax.text(0.5, 0.5, f'Значимость ошибка: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_enhanced_temporal_evolution(self, features_df: pd.DataFrame, ax):
        """
        График временной эволюции ключевых признаков.
        
        Args:
            features_df: DataFrame с признаками
            ax: Объект axes для рисования
        """
        try:
            # Отбор ключевых признаков для анализа временной эволюции
            key_features = ['avg_word_length', 'punctuation_ratio', 'type_token_ratio', 
                           'avg_sentence_length', 'vocabulary_richness']
            available = [f for f in key_features if f in features_df.columns]
            
            if available and 'year' in features_df.columns:
                # Группировка по годам и вычисление средних значений
                profiles = features_df.groupby('year')[available].mean()
                
                # Нормализация для сравнения
                profiles_norm = profiles / (profiles.max() + 1e-10)
                
                # Цветовая схема
                colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3', '#A8E6CF']
                markers = ['o', 's', '^', 'D', 'v']
                
                # Построение линий для каждого признака
                for i, feature in enumerate(available):
                    ax.plot(profiles_norm.index, profiles_norm[feature], 
                           marker=markers[i % len(markers)], 
                           label=feature.replace('_', ' ').title(), 
                           linewidth=2, markersize=8,
                           color=colors[i % len(colors)])
                
                ax.set_xlabel('Год')
                ax.set_ylabel('Нормализованное значение')
                ax.set_title('Временная Эволюция Признаков')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
                # Выделение трендов
                for i, feature in enumerate(available):
                    values = profiles_norm[feature].values
                    years = profiles_norm.index.values
                    if len(values) > 2:
                        # Простая линейная регрессия для тренда
                        z = np.polyfit(years, values, 1)
                        trend_line = np.poly1d(z)
                        ax.plot(years, trend_line(years), '--', 
                               color=colors[i % len(colors)], alpha=0.5)
        except Exception as e:
            ax.text(0.5, 0.5, f'Эволюция ошибка: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_feature_correlation(self, features_df: pd.DataFrame, ax):
        """
        Тепловая карта корреляции признаков.
        
        Args:
            features_df: DataFrame с признаками
            ax: Объект axes для рисования
        """
        try:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols 
                           if col not in ['year', 'doc_id', 'original_text_length']][:15]  # Топ 15
            
            if feature_cols:
                # Вычисление корреляционной матрицы
                corr_matrix = features_df[feature_cols].corr()
                
                # Создание тепловой карты
                im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                
                # Настройка меток осей
                ax.set_xticks(range(len(feature_cols)))
                ax.set_yticks(range(len(feature_cols)))
                ax.set_xticklabels([col.replace('_', '\n')[:10] for col in feature_cols], 
                                  rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels([col.replace('_', ' ')[:10] for col in feature_cols], 
                                  fontsize=8)
                
                # Добавление значений корреляции в ячейки
                for i in range(len(feature_cols)):
                    for j in range(len(feature_cols)):
                        corr_val = corr_matrix.iloc[i, j]
                        color = 'white' if abs(corr_val) > 0.5 else 'black'
                        ax.text(j, i, f'{corr_val:.2f}', ha="center", va="center", 
                               color=color, fontsize=6)
                
                ax.set_title('Корреляция Признаков')
                
                # Добавление цветовой шкалы
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Корреляция')
        except Exception as e:
            ax.text(0.5, 0.5, f'Корреляция ошибка: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_model_performance_comparison(self, validation_results: Dict, ax):
        """
        График сравнения производительности моделей.
        
        Args:
            validation_results: Результаты валидации
            ax: Объект axes для рисования
        """
        try:
            classification = validation_results.get('classification', {})
            individual = classification.get('individual', {})
            ensemble = classification.get('ensemble', {})
            
            all_results = {**individual, **ensemble}
            if all_results:
                # Сортировка по точности и взятие топ моделей
                sorted_models = sorted(all_results.items(), 
                                     key=lambda x: x[1].get('mean_accuracy', 0), 
                                     reverse=True)[:6]
                
                models = [item[0] for item in sorted_models]
                accuracies = [item[1].get('mean_accuracy', 0) for item in sorted_models]
                
                # Цветовое кодирование: ансамбли vs индивидуальные
                colors = ['#2ECC71' if any(ens in m for ens in ['Voting', 'Stack']) 
                         else '#3498DB' for m in models]
                
                bars = ax.bar(range(len(models)), accuracies, alpha=0.8, color=colors)
                
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels([m.replace('_', '\n')[:10] for m in models], 
                                  rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Точность')
                ax.set_title('Рейтинг Моделей')
                ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='Цель')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Подписи значений
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
                
                # Легенда для цветов
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='#2ECC71', label='Ансамбли'),
                                 Patch(facecolor='#3498DB', label='Индивидуальные')]
                ax.legend(handles=legend_elements, loc='upper right')
        except Exception as e:
            ax.text(0.5, 0.5, f'Рейтинг ошибка: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_bootstrap_results(self, validation_results: Dict, ax):
        """
        График результатов Bootstrap-анализа с доверительными интервалами.
        
        Args:
            validation_results: Результаты валидации
            ax: Объект axes для рисования
        """
        try:
            bootstrap = validation_results.get('bootstrap_results', {})
            if 'classification_accuracy' in bootstrap:
                acc_data = bootstrap['classification_accuracy']
                mean_acc = acc_data.get('mean', 0)
                std_acc = acc_data.get('std', 0)
                ci = acc_data.get('confidence_interval_95', [0, 0])
                
                # Построение столбца с доверительным интервалом
                bar = ax.bar(['Bootstrap\nТочность'], [mean_acc], 
                           yerr=[[mean_acc - ci[0]], [ci[1] - mean_acc]], 
                           capsize=10, alpha=0.7, color='#3498db',
                           error_kw=dict(elinewidth=3, capthick=3))
                
                # Линии референсных значений
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Случайный')
                ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='Цель')
                
                ax.set_ylabel('Точность')
                ax.set_title('Bootstrap ДИ (95%)')
                ax.set_ylim(0, 1)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Подписи значений
                ax.text(0, mean_acc + 0.05, f'μ = {mean_acc:.3f}\nσ = {std_acc:.3f}', 
                       ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                # Подписи доверительного интервала
                ax.text(0, ci[0] - 0.03, f'{ci[0]:.3f}', ha='center', va='top', fontsize=8)
                ax.text(0, ci[1] + 0.03, f'{ci[1]:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'Bootstrap анализ\nнедоступен', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        except Exception as e:
            ax.text(0.5, 0.5, f'Bootstrap ошибка: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')

# ===== ОСНОВНОЙ КЛАСС ВЕРИФИКАТОРА =====
class EnhancedTemporalVerifier:
    """Улучшенная система временной верификации авторства с комплексными признаками."""
    
    def __init__(self, config: Optional[EnhancedConfig] = None):
        """
        Инициализация верификатора.
        
        Args:
            config: Конфигурация системы (опционально)
        """
        self.config = config or EnhancedConfig()
        self.feature_extractor = ComprehensiveFeatureExtractor(self.config)
        self.validator = EnhancedValidator(self.config)
        # Использование исправленного класса визуализации
        self.visualizer = EnhancedVisualizerFixed(self.config)
        
        logger.info("Улучшенная Система Временной Верификации Авторства инициализирована")
        logger.info(f"Цель: >75% точности, >0.8 размеры эффектов")
    
    def run_enhanced_analysis(self, df: pd.DataFrame, 
                            train_years: List[int], 
                            test_years: List[int],
                            title: str = "Улучшенный Временной Анализ") -> Dict:
        """
        Запуск улучшенного анализа с комплексными признаками.
        
        Args:
            df: DataFrame с данными
            train_years: Годы для обучения
            test_years: Годы для тестирования
            title: Заголовок анализа
            
        Returns:
            Словарь с результатами анализа
        """
        
        logger.info(f"Начало улучшенного анализа: {title}")
        logger.info(f"Годы обучения: {train_years}")
        logger.info(f"Годы тестирования: {test_years}")
        
        # Извлечение комплексных признаков
        logger.info("Извлечение комплексных признаков...")
        features_df = self._extract_enhanced_features(df, train_years, test_years)
        
        if features_df is None or len(features_df) == 0:
            logger.error("Извлечение улучшенных признаков не удалось")
            return {'error': 'Извлечение улучшенных признаков не удалось'}
        
        logger.info(f"Улучшенные признаки извлечены из {len(features_df)} документов")
        logger.info(f"Размерность признаков: {features_df.shape}")
        
        # Улучшенная валидация
        logger.info("Выполнение улучшенной валидации...")
        validation_results = self.validator.validate_enhanced(features_df)
        
        # Быстрое сохранение результатов
        pickle_file = self._save_results_quick(features_df, validation_results)
        
        # Создание улучшенных визуализаций
        logger.info("Создание улучшенных визуализаций...")
        try:
            self.visualizer.create_enhanced_visualizations(features_df, validation_results, title)
            print("Визуализация успешна!")
        except Exception as e:
            print(f"Ошибка визуализации: {e}")
            print(f"Запустите отдельно: visualize_only('{pickle_file}')")
        
        # Генерация улучшенного резюме
        summary = self._generate_enhanced_summary(features_df, validation_results)
        
        # Печать резюме
        self._print_enhanced_summary(validation_results)
        
        return {
            'features_df': features_df,
            'validation_results': validation_results,
            'summary': summary,
            'pickle_file': pickle_file
        }
    
    def _save_results_quick(self, features_df: pd.DataFrame, validation_results: Dict) -> str:
        """
        Быстрое сохранение результатов анализа в pickle-файл.
        
        Args:
            features_df: DataFrame с признаками
            validation_results: Результаты валидации
            
        Returns:
            Имя файла с сохраненными результатами
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_results_{timestamp}.pkl"
        
        with open(filename, "wb") as f:
            pickle.dump({
                "features_df": features_df, 
                "validation_results": validation_results
            }, f)
        
        logger.info(f"Результаты сохранены в файл: {filename}")
        print(f"\nРезультаты сохранены: {filename}")
        return filename
    
    def _extract_enhanced_features(self, df: pd.DataFrame, 
                                 train_years: List[int], 
                                 test_years: List[int]) -> Optional[pd.DataFrame]:
        """
        Извлечение улучшенных признаков из корпуса.
        
        Args:
            df: DataFrame с данными
            train_years: Годы для обучения
            test_years: Годы для тестирования
            
        Returns:
            DataFrame с признаками или None при ошибке
        """
        
        # Фильтрация данных
        df_filtered = df[df['text'].str.len() > 500].copy()  # Только длинные тексты
        
        def assign_label(year):
            """Назначение меток на основе года."""
            if year in train_years:
                return 'train'
            elif year in test_years:
                return 'test'
            else:
                return 'exclude'
        
        df_filtered['temporal_label'] = df_filtered['year'].apply(assign_label)
        df_final = df_filtered[df_filtered['temporal_label'] != 'exclude'].copy()
        
        logger.info(f"Распределение данных: {df_final.groupby('temporal_label').size().to_dict()}")
        
        print(f"\nДАННЫЕ ПОДГОТОВЛЕНЫ:")
        print(f"   Обучающие документы: {len(df_final[df_final['temporal_label'] == 'train'])}")
        print(f"   Тестовые документы: {len(df_final[df_final['temporal_label'] == 'test'])}")
        print(f"   Минимальная длина текста: {df_final['text'].str.len().min()} символов")
        print(f"   Максимальная длина текста: {df_final['text'].str.len().max()} символов")
        print(f"\nИЗВЛЕЧЕНИЕ ПРИЗНАКОВ...")
        print(f"   {len(df_final)} документов")
        
        # Извлечение улучшенных признаков с прогресс-баром
        features_list = []
        
        for idx, row in tqdm(df_final.iterrows(), total=len(df_final), desc="Извлечение улучшенных признаков"):
            try:
                # Извлечение комплексных признаков
                features = self.feature_extractor.extract_all_features(row['text'])
                
                if features:
                    features.update({
                        'doc_id': idx,                              # ID документа
                        'year': row['year'],                        # Год написания
                        'label': row['temporal_label'],             # Метка (train/test)
                        'original_text_length': len(row['text'])    # Исходная длина текста
                    })
                    features_list.append(features)
                    
            except Exception as e:
                logger.debug(f"Извлечение признаков не удалось для документа {idx}: {e}")
                continue
        
        if not features_list:
            return None
        
        features_df = pd.DataFrame(features_list)
        
        # Обработка пропущенных значений
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
        
        # Удаление константных признаков
        constant_features = [col for col in numeric_cols if features_df[col].std() == 0]
        if constant_features:
            features_df = features_df.drop(columns=constant_features)
            logger.info(f"Удалено {len(constant_features)} константных признаков")
        
        logger.info(f"Итоговая матрица улучшенных признаков: {features_df.shape}")
        
        print(f"\nИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ЗАВЕРШЕНО!")
        print(f"   Создана матрица признаков: {features_df.shape}")
        print(f"   Извлечено {features_df.shape[1]-4} стилометрических признаков")
        print(f"   Удалено {len(constant_features)} константных признаков")
        print(f"\nНАЧИНАЮ МАШИННОЕ ОБУЧЕНИЕ И ВАЛИДАЦИЮ...")
        
        return features_df
    
    def _generate_enhanced_summary(self, features_df: pd.DataFrame, 
                                 validation_results: Dict) -> Dict:
        """
        Генерация улучшенного резюме.
        
        Args:
            features_df: DataFrame с признаками
            validation_results: Результаты валидации
            
        Returns:
            Словарь с резюме
        """
        
        # Поиск лучшей модели
        classification = validation_results.get('classification', {})
        individual = classification.get('individual', {})
        ensemble = classification.get('ensemble', {})
        
        all_models = {**individual, **ensemble}
        
        best_accuracy = 0
        best_model = 'None'
        
        for name, results in all_models.items():
            if 'mean_accuracy' in results:
                acc = results['mean_accuracy']
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = name
        
        # Определение общего вердикта
        if best_accuracy > 0.8:
            verdict = "Сильное свидетельство временной стилистической вариации"
            confidence = "Очень Высокая"
        elif best_accuracy > 0.75:
            verdict = "Хорошее свидетельство временной стилистической вариации"
            confidence = "Высокая"
        elif best_accuracy > 0.65:
            verdict = "Умеренное свидетельство временной стилистической вариации"
            confidence = "Средняя"
        else:
            verdict = "Слабое свидетельство временной стилистической вариации"
            confidence = "Низкая"
        
        # Резюме размеров эффектов
        effect_sizes = validation_results.get('effect_sizes', {})
        effect_summary = effect_sizes.get('summary', {})
        
        # Резюме статистической значимости
        stats_summary = validation_results.get('statistical_tests', {}).get('summary', {})
        
        return {
            'dataset_info': {
                'total_documents': len(features_df),
                'train_documents': len(features_df[features_df['label'] == 'train']),
                'test_documents': len(features_df[features_df['label'] == 'test']),
                'years_analyzed': sorted(features_df['year'].unique().tolist()),
                'features_extracted': len([col for col in features_df.columns 
                                         if col not in ['label', 'year', 'doc_id', 'original_text_length']])
            },
            'performance': {
                'best_model': best_model,
                'best_accuracy': best_accuracy,
                'target_achieved': best_accuracy > 0.75,
                'all_models': all_models
            },
            'effect_sizes': {
                'mean_effect_size': effect_summary.get('mean_effect_size', 0),
                'large_effects_count': effect_summary.get('large_effects_count', 0),
                'large_effects_ratio': effect_summary.get('large_effects_ratio', 0),
                'target_achieved': effect_summary.get('large_effects_ratio', 0) > 0.2
            },
            'statistical_significance': {
                'total_features': stats_summary.get('total_features', 0),
                'significant_features': stats_summary.get('significant_features', 0),
                'significance_ratio': stats_summary.get('significant_ratio', 0)
            },
            'conclusions': {
                'verdict': verdict,
                'confidence': confidence,
                'targets_achieved': {
                    'accuracy_target': best_accuracy > 0.75,
                    'effect_size_target': effect_summary.get('large_effects_ratio', 0) > 0.2
                },
                'recommendation': self._get_enhanced_recommendation(best_accuracy, effect_summary.get('large_effects_ratio', 0))
            }
        }
    
    def _get_enhanced_recommendation(self, accuracy: float, effect_ratio: float) -> str:
        """
        Получение улучшенной рекомендации на основе результатов.
        
        Args:
            accuracy: Точность модели
            effect_ratio: Соотношение больших эффектов
            
        Returns:
            Рекомендация в виде строки
        """
        if accuracy > 0.8 and effect_ratio > 0.2:
            return "Отличные результаты"
        elif accuracy > 0.75 and effect_ratio > 0.15:
            return "Хорошие результаты достигнуты"
        elif accuracy > 0.7 or effect_ratio > 0.1:
            return "Обнадеживающие результаты."
        else:
            return "Результаты нуждаются в улучшении."
    
    def _print_enhanced_summary(self, validation_results: Dict):
        """
        Печать улучшенного резюме.
        
        Args:
            validation_results: Результаты валидации
        """
        print("\n" + "="*70)
        print("УЛУЧШЕННЫЙ ОТЧЕТ РЕЗЮМЕ")
        print("="*70)
        
        # Производительность классификации
        classification = validation_results.get('classification', {})
        individual = classification.get('individual', {})
        ensemble = classification.get('ensemble', {})
        
        print("\nПРОИЗВОДИТЕЛЬНОСТЬ КЛАССИФИКАЦИИ:")
        
        # Индивидуальные классификаторы
        if individual:
            print("   Индивидуальные классификаторы:")
            for name, results in individual.items():
                if 'mean_accuracy' in results:
                    acc = results['mean_accuracy']
                    std = results['std_accuracy']
                    status = "ЦЕЛЬ ДОСТИГНУТА!" if acc > 0.75 else "Хороший прогресс" if acc > 0.65 else "Нуждается в улучшении"
                    print(f"      {name}: {acc:.3f} ± {std:.3f} {status}")
        
        # Ансамблевые классификаторы
        if ensemble:
            print("   Ансамблевые классификаторы:")
            for name, results in ensemble.items():
                if 'mean_accuracy' in results:
                    acc = results['mean_accuracy']
                    std = results['std_accuracy']
                    status = "ЦЕЛЬ ДОСТИГНУТА!" if acc > 0.75 else "Хороший прогресс" if acc > 0.65 else "Нуждается в улучшении"
                    print(f"      {name}: {acc:.3f} ± {std:.3f} {status}")
        
        # Статистическая значимость
        tests = validation_results.get('statistical_tests', {})
        if 'summary' in tests:
            summary = tests['summary']
            total_features = summary.get('total_features', 0)
            significant_features = summary.get('significant_features', 0)
            ratio = summary.get('significant_ratio', 0)
            
            print(f"\nСТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ:")
            print(f"   Всего проанализировано признаков: {total_features}")
            print(f"   Статистически значимых: {significant_features}")
            print(f"   Коэффициент значимости: {ratio:.1%}")
            
            if ratio > 0.3:
                print("   Сильное статистическое свидетельство!")
            elif ratio > 0.15:
                print("   Умеренное статистическое свидетельство")
            else:
                print("   Слабое статистическое свидетельство")
        
        # Размеры эффектов
        effect_sizes = validation_results.get('effect_sizes', {})
        if 'summary' in effect_sizes:
            effect_summary = effect_sizes['summary']
            mean_effect = effect_summary.get('mean_effect_size', 0)
            max_effect = effect_summary.get('max_effect_size', 0)
            large_effects = effect_summary.get('large_effects_count', 0)
            large_ratio = effect_summary.get('large_effects_ratio', 0)
            
            print(f"\nРАЗМЕРЫ ЭФФЕКТОВ:")
            print(f"   Средний размер эффекта: {mean_effect:.3f}")
            print(f"   Максимальный размер эффекта: {max_effect:.3f}")
            print(f"   Большие эффекты (>0.8): {large_effects}")
            print(f"   Соотношение больших эффектов: {large_ratio:.1%}")
            
            if large_ratio > 0.2:
                print("   Сильные размеры эффектов достигнуты!")
            elif large_ratio > 0.1:
                print("   Умеренные размеры эффектов")
            else:
                print("   Слабые размеры эффектов")
        
        # Важность признаков
        importance = validation_results.get('feature_importance', {})
        if 'top_10_rf' in importance:
            print(f"\nТОП ДИСКРИМИНИРУЮЩИЕ ПРИЗНАКИ:")
            for i, feature in enumerate(importance['top_10_rf'][:5], 1):
                print(f"   {i}. {feature.replace('_', ' ').title()}")
        
        print("\n" + "="*70)

# ===== ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ =====
def load_suvorin_data_enhanced():
    """Загрузка и предобработка данных Суворина с улучшенной фильтрацией."""
    print("ЗАГРУЗКА ДАННЫХ СУВОРИНА - УЛУЧШЕННАЯ ВЕРСИЯ")
    print("="*60)
    
    try:
        df = pd.read_csv('suvorin_letters.csv')  # Загрузка CSV файла
        print(f"Загружено {len(df)} документов")
        
        print("\nУЛУЧШЕННАЯ ФИЛЬТРАЦИЯ ДАННЫХ:")
        
        # Фильтр по длине текста (минимум 500 символов)
        df = df[df['text'].str.len() > 500]
        print(f"    После фильтра длины (>500 символов): {len(df)} документов")
        
        # Удаление дубликатов по тексту
        df = df.drop_duplicates(subset=['text'])
        print(f"    После удаления дубликатов: {len(df)} документов")
        
        # Фильтр баланса по годам (минимум документов на год)
        min_docs_per_year = 5
        year_counts = df['year'].value_counts()
        valid_years = year_counts[year_counts >= min_docs_per_year].index
        df = df[df['year'].isin(valid_years)]
        print(f"    После фильтра баланса лет (≥{min_docs_per_year} док/год): {len(df)} документов")
        
        print(f"\nУЛУЧШЕННАЯ ЗАГРУЗКА ДАННЫХ ЗАВЕРШЕНА")
        print(f"Готово для улучшенного анализа: {len(df)} высококачественных документов")
        print("="*60)
        
        return df
        
    except FileNotFoundError:
        print("\nОШИБКА: Файл 'suvorin_letters.csv' не найден")
        print("Пожалуйста, убедитесь, что файл с данными находится в той же папке, что и скрипт.")
        return None
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None

# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====
def visualize_only(pickle_filename: str):
    """
    Загрузка и визуализация сохраненных результатов.
    
    Args:
        pickle_filename: Имя pickle-файла с результатами
    """
    try:
        with open(pickle_filename, 'rb') as f:
            data = pickle.load(f)
        
        features_df = data['features_df']
        validation_results = data['validation_results']
        
        print(f" Загружено из {pickle_filename}")
        print(f"Данные: {features_df.shape}")
        
        # Создание визуализации
        config = EnhancedConfig()
        visualizer = EnhancedVisualizerFixed(config)
        visualizer.create_enhanced_visualizations(
            features_df, validation_results, "Загруженный Анализ"
        )
        
        print("Визуализация завершена!")
        
    except Exception as e:
        print(f"Ошибка: {e}")

# ===== ГЛАВНАЯ ФУНКЦИЯ =====
def main_enhanced():
    """Главная функция для улучшенного анализа."""
    print("УЛУЧШЕННАЯ ВРЕМЕННАЯ ВЕРИФИКАЦИЯ АВТОРСТВА")
    print("Цель: >75% Точности, >0.8 Размеры Эффектов")
    print("="*70)
    
    # Загрузка данных
    df = load_suvorin_data_enhanced()
    if df is None:
        return
    
    # Инициализация ультра-улучшенной системы для цели 75%+
    config = EnhancedConfig()
    verifier = EnhancedTemporalVerifier(config)
    
    print(f"\nКОНФИГУРАЦИЯ ДЛЯ ДОСТИЖЕНИЯ 75%+ ТОЧНОСТИ:")
    print(f"   Word2Vec: размер={config.vector_size}, окно={config.window_size}, эпох={config.epochs}")
    print(f"   Граф: {config.n_core_words} ключевых слов, {config.k_similar_words} связей")
    print(f"   Признаки: {config.n_feature_selection} лучших из всех извлеченных")
    print(f"   CV: {config.cv_folds} фолдов для надежности")
    print(f"   Ансамбли: Voting + Stacking + ExtraTrees")
    
    # Запуск улучшенного анализа
    print("\n" + "="*70)
    print("УЛУЧШЕННЫЙ ОСНОВНОЙ ЭКСПЕРИМЕНТ")
    print("="*70)
    
    results = verifier.run_enhanced_analysis(
        df=df,
        train_years=[1904, 1907],      # Годы для обучения модели
        test_years=[1905, 1906, 1908], # Годы для тестирования модели
        title="Улучшенный Анализ Суворина"
    )
    
    # Печать финальных результатов
    if results and 'summary' in results:
        print("\n" + "="*70)
        print("РЕЗУЛЬТАТЫ УЛУЧШЕННОГО АНАЛИЗА")
        print("="*70)
        
        summary = results['summary']
        performance = summary.get('performance', {})
        effect_sizes = summary.get('effect_sizes', {})
        conclusions = summary.get('conclusions', {})
        
        print(f"ЛУЧШАЯ МОДЕЛЬ: {performance.get('best_model', 'N/A')}")
        print(f"ЛУЧШАЯ ТОЧНОСТЬ: {performance.get('best_accuracy', 0):.3f}")
        print(f"ЦЕЛЬ ПО ТОЧНОСТИ: {'ДОСТИГНУТА' if performance.get('target_achieved', False) else 'НЕ ДОСТИГНУТА'}")
        
        print(f"\nРАЗМЕРЫ ЭФФЕКТОВ:")
        print(f"   Средний размер эффекта: {effect_sizes.get('mean_effect_size', 0):.3f}")
        print(f"   Большие эффекты: {effect_sizes.get('large_effects_count', 0)}")
        print(f"   Соотношение больших эффектов: {effect_sizes.get('large_effects_ratio', 0):.1%}")
        print(f"ЦЕЛЬ ПО РАЗМЕРУ ЭФФЕКТА: {'ДОСТИГНУТА' if effect_sizes.get('target_achieved', False) else 'НЕ ДОСТИГНУТА'}")
        
        print(f"\nОБЩИЙ РЕЗУЛЬТАТ: {conclusions.get('verdict', 'N/A')}")
        print(f"УРОВЕНЬ ДОВЕРИЯ: {conclusions.get('confidence', 'N/A')}")
        
        print(f"\n💡 РЕКОМЕНДАЦИЯ:")
        print(f"   {conclusions.get('recommendation', 'N/A')}")
        
        # Индикатор успеха с улучшенными критериями
        accuracy_achieved = performance.get('target_achieved', False)
        effect_achieved = effect_sizes.get('target_achieved', False)
        best_acc = performance.get('best_accuracy', 0)
        
        print(f"\nИТОГОВАЯ ОЦЕНКА:")
        if best_acc > 0.8:
            print(f"ПРЕВОСХОДНЫЙ РЕЗУЛЬТАТ! Точность {best_acc:.1%} > 80%")
        elif best_acc > 0.75:
            print(f"ЦЕЛЬ ДОСТИГНУТА! Точность {best_acc:.1%} > 75%")
        elif best_acc > 0.7:
            print(f"БЛИЗКО К ЦЕЛИ! Точность {best_acc:.1%} (нужно +{(0.75-best_acc)*100:.1f}%)")
        else:
            print(f"ТРЕБУЕТСЯ УЛУЧШЕНИЕ! Точность {best_acc:.1%}")
        
        if accuracy_achieved and effect_achieved:
            print(f"\nЦЕЛИ ДОСТИГНУТЫ!")
        elif accuracy_achieved or effect_achieved:
            print(f"\nЧАСТИЧНЫЙ УСПЕХ.")
            print(f"Рассмотрите дополнительные улучшения")
            if not accuracy_achieved:
                print(f"   - Добавьте больше исторических признаков")
                
            if not effect_achieved:
                print(f"   - Увеличьте размер выборки")
                
        else:
            print(f"\nЦЕЛИ НЕ ДОСТИГНУТЫ")
          
        print("="*70)



# ===== ТОЧКА ВХОДА В ПРОГРАММУ =====
if __name__ == "__main__":
    """
    Точка входа в программу.
    Запускает главную функцию улучшенного анализа при прямом выполнении скрипта.
    """
    main_enhanced()

    # --- Конец файла ---