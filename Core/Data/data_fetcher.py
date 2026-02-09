"""
Модуль для получения рыночных данных (исторических и реальных).
Поддерживает multiple timeframes и кэширование.
"""
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

from ..broker.base_broker import BaseBroker


class DataFetcher:
    """Класс для получения и управления рыночными данными"""

    def __init__(self, broker: BaseBroker):
        self.broker = broker
        self.data_cache = defaultdict(dict)  # symbol -> timeframe -> DataFrame
        self.max_cache_size = 10000  # Максимальное количество баров в кэше

    async def fetch_historical_data(self,
                                    symbol: str,
                                    timeframe: str,
                                    limit: int = 1000,
                                    since: Optional[datetime] = None) -> pd.DataFrame:
        """Получение исторических данных"""
        try:
            since_timestamp = int(since.timestamp() * 1000) if since else None

            df = await self.broker.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since_timestamp
            )

            if not df.empty:
                self._update_cache(symbol, timeframe, df)

            return df

        except Exception as e:
            print(f"Error fetching data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    async def fetch_multiple_timeframes(self,
                                        symbol: str,
                                        timeframes: List[str],
                                        days_back: int = 30) -> Dict[str, pd.DataFrame]:
        """Получение данных на нескольких таймфреймах"""
        results = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        for timeframe in timeframes:
            df = await self.fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                since=start_date
            )

            if not df.empty:
                results[timeframe] = df

            # Небольшая задержка между запросами
            await asyncio.sleep(0.1)

        return results

    async def fetch_recent_data(self,
                                symbol: str,
                                timeframe: str,
                                lookback_bars: int = 100) -> pd.DataFrame:
        """Получение последних данных"""
        df = await self.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=lookback_bars
        )

        return df

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Получение последней цены"""
        try:
            ticker = await self.broker.get_ticker(symbol)
            return ticker.get('last_price')
        except:
            return None

    def _update_cache(self, symbol: str, timeframe: str, new_data: pd.DataFrame):
        """Обновление кэша данных"""
        cache_key = timeframe

        if cache_key in self.data_cache[symbol]:
            # Объединяем с существующими данными
            existing_df = self.data_cache[symbol][cache_key]

            # Удаляем дубликаты
            combined = pd.concat([existing_df, new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()

            # Ограничиваем размер кэша
            if len(combined) > self.max_cache_size:
                combined = combined.iloc[-self.max_cache_size:]

            self.data_cache[symbol][cache_key] = combined
        else:
            self.data_cache[symbol][cache_key] = new_data

    def get_cached_data(self,
                        symbol: str,
                        timeframe: str,
                        lookback_bars: Optional[int] = None) -> pd.DataFrame:
        """Получение данных из кэша"""
        if symbol in self.data_cache and timeframe in self.data_cache[symbol]:
            df = self.data_cache[symbol][timeframe]
            if lookback_bars and len(df) > lookback_bars:
                return df.iloc[-lookback_bars:]
            return df
        return pd.DataFrame()

    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """Очистка кэша"""
        if symbol and timeframe:
            if symbol in self.data_cache:
                self.data_cache[symbol].pop(timeframe, None)
        elif symbol:
            self.data_cache.pop(symbol, None)
        else:
            self.data_cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Получение информации о кэше"""
        info = {
            'total_symbols': len(self.data_cache),
            'symbols': {}
        }

        for symbol, timeframes in self.data_cache.items():
            info['symbols'][symbol] = {
                'timeframes': list(timeframes.keys()),
                'total_bars': sum(len(df) for df in timeframes.values())
            }

        return info