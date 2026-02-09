"""
Модуль для обработки рыночных данных.
Включает очистку, ресемплинг и добавление технических индикаторов.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta


class DataProcessor:
    """Класс для обработки рыночных данных"""

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Очистка данных:
        - Удаление NaN значений
        - Проверка корректности цен
        - Удаление выбросов
        """
        if df.empty:
            return df

        # Создаем копию
        clean_df = df.copy()

        # Удаляем строки с NaN в основных колонках
        clean_df = clean_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

        # Проверяем корректность цен (high >= low, цена в разумных пределах)
        mask = (
                (clean_df['high'] >= clean_df['low']) &
                (clean_df['high'] >= clean_df['close']) &
                (clean_df['low'] <= clean_df['close']) &
                (clean_df['close'] > 0) &
                (clean_df['volume'] >= 0)
        )

        clean_df = clean_df[mask]

        # Удаляем выбросы (цены, отличающиеся более чем на 50% от предыдущей)
        if len(clean_df) > 1:
            returns = clean_df['close'].pct_change().abs()
            clean_df = clean_df[returns <= 0.5]

        return clean_df

    @staticmethod
    def resample_data(df: pd.DataFrame,
                      timeframe: str,
                      volume: bool = True) -> pd.DataFrame:
        """
        Ресемплинг данных на другой таймфрейм

        Args:
            df: DataFrame с данными
            timeframe: Целевой таймфрейм ('15m', '1h', '4h', '1d')
            volume: Включать ли объем
        """
        if df.empty:
            return df

        # Правила агрегации
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }

        if volume:
            agg_rules['volume'] = 'sum'

        # Ресемплинг
        resampled = df.resample(timeframe).agg(agg_rules)

        # Удаляем NaN
        resampled = resampled.dropna()

        return resampled

    @staticmethod
    def add_indicators(df: pd.DataFrame,
                       indicators: List[str],
                       params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Добавление технических индикаторов к данным

        Args:
            df: DataFrame с данными OHLCV
            indicators: Список индикаторов для добавления
            params: Параметры индикаторов
        """
        if df.empty:
            return df

        result_df = df.copy()
        params = params or {}

        for indicator in indicators:
            if indicator == 'sma':
                periods = params.get('sma_periods', [20, 50, 200])
                for period in periods:
                    result_df[f'SMA_{period}'] = result_df['close'].rolling(window=period).mean()

            elif indicator == 'ema':
                periods = params.get('ema_periods', [12, 26])
                for period in periods:
                    result_df[f'EMA_{period}'] = result_df['close'].ewm(
                        span=period, adjust=False
                    ).mean()

            elif indicator == 'rsi':
                period = params.get('rsi_period', 14)
                delta = result_df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                result_df['RSI'] = 100 - (100 / (1 + rs))

            elif indicator == 'macd':
                fast = params.get('macd_fast', 12)
                slow = params.get('macd_slow', 26)
                signal = params.get('macd_signal', 9)

                ema_fast = result_df['close'].ewm(span=fast, adjust=False).mean()
                ema_slow = result_df['close'].ewm(span=slow, adjust=False).mean()

                result_df['MACD'] = ema_fast - ema_slow
                result_df['MACD_Signal'] = result_df['MACD'].ewm(span=signal, adjust=False).mean()
                result_df['MACD_Hist'] = result_df['MACD'] - result_df['MACD_Signal']

            elif indicator == 'bb':
                period = params.get('bb_period', 20)
                std_dev = params.get('bb_std', 2)

                sma = result_df['close'].rolling(window=period).mean()
                rolling_std = result_df['close'].rolling(window=period).std()

                result_df['BB_Upper'] = sma + (rolling_std * std_dev)
                result_df['BB_Middle'] = sma
                result_df['BB_Lower'] = sma - (rolling_std * std_dev)

            elif indicator == 'atr':
                period = params.get('atr_period', 14)

                high_low = result_df['high'] - result_df['low']
                high_close = np.abs(result_df['high'] - result_df['close'].shift())
                low_close = np.abs(result_df['low'] - result_df['close'].shift())

                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = np.max(ranges, axis=1)

                result_df['ATR'] = true_range.rolling(window=period).mean()

        return result_df

    @staticmethod
    def detect_support_resistance(df: pd.DataFrame,
                                  window: int = 20,
                                  threshold: float = 0.005) -> Tuple[List[float], List[float]]:
        """
        Обнаружение уровней поддержки и сопротивления

        Args:
            df: DataFrame с данными
            window: Окно для поиска экстремумов
            threshold: Порог для объединения близких уровней
        """
        if len(df) < window * 2:
            return [], []

        supports = []
        resistances = []

        # Ищем локальные минимумы (поддержки)
        lows = df['low'].rolling(window=window, center=True).min()
        for i in range(window, len(df) - window):
            if df['low'].iloc[i] == lows.iloc[i]:
                supports.append(df['low'].iloc[i])

        # Ищем локальные максимумы (сопротивления)
        highs = df['high'].rolling(window=window, center=True).max()
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == highs.iloc[i]:
                resistances.append(df['high'].iloc[i])

        # Объединяем близкие уровни
        supports = DataProcessor._merge_close_levels(supports, threshold)
        resistances = DataProcessor._merge_close_levels(resistances, threshold)

        return supports, resistances

    @staticmethod
    def _merge_close_levels(levels: List[float], threshold: float) -> List[float]:
        """Объединение близких уровней"""
        if not levels:
            return []

        levels = sorted(levels)
        merged = []

        current_level = levels[0]
        current_count = 1

        for level in levels[1:]:
            if abs(level - current_level) / current_level <= threshold:
                # Объединяем уровни
                current_level = (current_level * current_count + level) / (current_count + 1)
                current_count += 1
            else:
                merged.append(current_level)
                current_level = level
                current_count = 1

        merged.append(current_level)
        return merged

    @staticmethod
    def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
        """
        Расчет уровней Фибоначчи

        Args:
            high: Максимальная цена
            low: Минимальная цена
        """
        diff = high - low

        levels = {
            '0.0': low,
            '0.236': high - diff * 0.236,
            '0.382': high - diff * 0.382,
            '0.5': high - diff * 0.5,
            '0.618': high - diff * 0.618,
            '0.786': high - diff * 0.786,
            '1.0': high,
            '1.618': high + diff * 0.618,
            '2.618': high + diff * 1.618
        }

        return levels

    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame,
                                 price_bins: int = 20) -> Dict[str, Any]:
        """
        Расчет Volume Profile

        Args:
            df: DataFrame с данными
            price_bins: Количество ценовых уровней
        """
        if df.empty:
            return {}

        # Определяем ценовые уровни
        min_price = df['low'].min()
        max_price = df['high'].max()

        bin_edges = np.linspace(min_price, max_price, price_bins + 1)

        # Распределяем объем по ценовым уровням
        volume_profile = {}
        for i in range(len(bin_edges) - 1):
            price_range = (bin_edges[i], bin_edges[i + 1])

            # Объем для свечей, которые пересекают этот диапазон
            mask = (df['low'] <= price_range[1]) & (df['high'] >= price_range[0])
            volume_in_range = df.loc[mask, 'volume'].sum()

            volume_profile[f'{price_range[0]:.2f}-{price_range[1]:.2f}'] = volume_in_range

        # Находим Point of Control (POC) - уровень с максимальным объемом
        poc_range = max(volume_profile, key=volume_profile.get)
        poc_price = (float(poc_range.split('-')[0]) + float(poc_range.split('-')[1])) / 2

        return {
            'volume_profile': volume_profile,
            'poc_price': poc_price,
            'price_range': (min_price, max_price)
        }