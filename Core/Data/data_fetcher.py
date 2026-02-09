"""
Менеджер данных для получения и обработки рыночных данных.
Поддерживает multiple timeframes, исторические данные и real-time потоки.
"""
import asyncio
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
from datetime import datetime, timedelta
import time
from collections import defaultdict
import threading

from ..broker.base_broker import BaseBroker


class DataFetcher:
    """Класс для получения и управления рыночными данными"""

    def __init__(self, broker: BaseBroker, cache_size: int = 10000):
        self.broker = broker
        self.cache_size = cache_size

        # Кэш данных по символам и таймфреймам
        self.data_cache: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)

        # Callback-функции для обновлений
        self.update_callbacks: List[Callable] = []

        # Флаг для остановки потоков
        self._stop_event = threading.Event()
        self._data_threads: Dict[str, threading.Thread] = {}

    async def fetch_historical_data(self, symbol: str, timeframe: str,
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None,
                                    limit: int = 1000) -> pd.DataFrame:
        """Получение исторических данных"""
        try:
            # Конвертация дат в timestamp
            since = None
            if start_date:
                since = int(start_date.timestamp() * 1000)

            # Получение данных
            df = await self.broker.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since
            )

            # Фильтрация по end_date если указан
            if end_date and not df.empty:
                df = df[df.index <= end_date]

            # Кэширование
            self._cache_data(symbol, timeframe, df)

            return df

        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_multiple_timeframes(self, symbol: str,
                                        timeframes: List[str],
                                        days_back: int = 7) -> Dict[str, pd.DataFrame]:
        """Получение данных на нескольких таймфреймах"""
        results = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        for tf in timeframes:
            df = await self.fetch_historical_data(
                symbol=symbol,
                timeframe=tf,
                start_date=start_date,
                end_date=end_date
            )
            if not df.empty:
                results[tf] = df
                # Небольшая задержка для избежания rate limits
                await asyncio.sleep(0.1)

        return results

    def _cache_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Кэширование данных"""
        cache_key = f"{symbol}_{timeframe}"

        if cache_key in self.data_cache[symbol]:
            # Объединение с существующими данными
            existing_df = self.data_cache[symbol][cache_key]
            combined = pd.concat([existing_df, data]).drop_duplicates()

            # Сортировка по времени
            combined = combined.sort_index()

            # Ограничение размера кэша
            if len(combined) > self.cache_size:
                combined = combined.iloc[-self.cache_size:]

            self.data_cache[symbol][cache_key] = combined
        else:
            self.data_cache[symbol][cache_key] = data

    def get_cached_data(self, symbol: str, timeframe: str,
                        lookback: Optional[int] = None) -> pd.DataFrame:
        """Получение кэшированных данных"""
        cache_key = f"{symbol}_{timeframe}"

        if symbol in self.data_cache and cache_key in self.data_cache[symbol]:
            df = self.data_cache[symbol][cache_key]
            if lookback and len(df) > lookback:
                return df.iloc[-lookback:]
            return df

        return pd.DataFrame()

    def start_data_stream(self, symbol: str, timeframe: str,
                          callback: Callable[[pd.DataFrame], None]):
        """Запуск потока данных в реальном времени"""
        stream_key = f"{symbol}_{timeframe}"

        if stream_key in self._data_threads:
            print(f"Data stream for {stream_key} already running")
            return

        def data_stream():
            """Функция потока данных"""
            last_update = None

            while not self._stop_event.is_set():
                try:
                    # В реальной реализации здесь будет WebSocket
                    # Для упрощения используем polling

                    # Получаем последние данные
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    df = loop.run_until_complete(
                        self.broker.get_ohlcv(symbol, timeframe, limit=2)
                    )

                    if not df.empty:
                        latest_data = df.iloc[-1:]

                        # Проверяем, не обновляли ли мы уже эту свечу
                        if last_update is None or latest_data.index[-1] != last_update:
                            last_update = latest_data.index[-1]

                            # Обновляем кэш
                            self._cache_data(symbol, timeframe, latest_data)

                            # Вызываем callback
                            callback(latest_data)

                    # Задержка в зависимости от таймфрейма
                    sleep_time = self._get_sleep_time(timeframe)
                    time.sleep(sleep_time)

                except Exception as e:
                    print(f"Error in data stream for {symbol}: {e}")
                    time.sleep(5)

        # Запускаем поток
        thread = threading.Thread(
            target=data_stream,
            name=f"DataStream_{stream_key}",
            daemon=True
        )
        thread.start()
        self._data_threads[stream_key] = thread

        print(f"Started data stream for {stream_key}")

    def stop_data_stream(self, symbol: str, timeframe: str):
        """Остановка потока данных"""
        stream_key = f"{symbol}_{timeframe}"

        if stream_key in self._data_threads:
            self._stop_event.set()
            self._data_threads[stream_key].join(timeout=5)
            del self._data_threads[stream_key]
            self._stop_event.clear()
            print(f"Stopped data stream for {stream_key}")

    def stop_all_streams(self):
        """Остановка всех потоков данных"""
        for stream_key in list(self._data_threads.keys()):
            symbol, timeframe = stream_key.split("_", 1)
            self.stop_data_stream(symbol, timeframe)

    def _get_sleep_time(self, timeframe: str) -> int:
        """Получение времени сна для polling"""
        timeframe_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        return timeframe_seconds.get(timeframe, 60)

    def add_technical_indicators(self, df: pd.DataFrame,
                                 indicators: List[Dict[str, Any]]) -> pd.DataFrame:
        """Добавление технических индикаторов к данным"""
        result_df = df.copy()

        for indicator_config in indicators:
            indicator_type = indicator_config.get('type')
            params = indicator_config.get('params', {})

            if indicator_type == 'sma':
                period = params.get('period', 20)
                result_df[f'SMA_{period}'] = result_df['close'].rolling(window=period).mean()

            elif indicator_type == 'ema':
                period = params.get('period', 20)
                result_df[f'EMA_{period}'] = result_df['close'].ewm(span=period, adjust=False).mean()

            elif indicator_type == 'rsi':
                period = params.get('period', 14)
                delta = result_df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                result_df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

            elif indicator_type == 'bb':
                period = params.get('period', 20)
                std_dev = params.get('std_dev', 2)
                sma = result_df['close'].rolling(window=period).mean()
                rolling_std = result_df['close'].rolling(window=period).std()
                result_df[f'BB_upper_{period}'] = sma + (rolling_std * std_dev)
                result_df[f'BB_middle_{period}'] = sma
                result_df[f'BB_lower_{period}'] = sma - (rolling_std * std_dev)

        return result_df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Получение последней цены из кэша"""
        if symbol in self.data_cache:
            # Берем первый доступный таймфрейм
            for timeframe, df in self.data_cache[symbol].items():
                if not df.empty:
                    return float(df['close'].iloc[-1])
        return None