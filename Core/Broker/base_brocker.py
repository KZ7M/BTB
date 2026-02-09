"""
Базовый абстрактный класс для унифицированного интерфейса брокера.
Все конкретные реализации брокеров должны наследоваться от этого класса.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
from datetime import datetime


@dataclass
class Order:
    """Унифицированная структура ордера"""
    id: str
    symbol: str
    side: str  # 'buy' или 'sell'
    type: str  # 'market', 'limit', 'stop', 'stop_market'
    quantity: float
    price: Optional[float] = None
    status: str = 'pending'
    timestamp: datetime = None
    reduce_only: bool = False


@dataclass
class Position:
    """Унифицированная структура позиции"""
    symbol: str
    side: str  # 'long' или 'short'
    quantity: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: int = 1
    liquidation_price: Optional[float] = None


@dataclass
class Balance:
    """Унифицированная структура баланса"""
    total: float
    available: float
    currency: str = 'USDT'


class BaseBroker(ABC):
    """Абстрактный базовый класс брокера"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Подключение к бирже"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Отключение от биржи"""
        pass

    @abstractmethod
    async def get_balance(self) -> Balance:
        """Получение баланса"""
        pass

    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Получение позиций"""
        pass

    @abstractmethod
    async def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Получение ордеров"""
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Размещение ордера"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Отмена ордера"""
        pass

    @abstractmethod
    async def get_ohlcv(self,
                        symbol: str,
                        timeframe: str,
                        limit: int = 500,
                        since: Optional[int] = None) -> pd.DataFrame:
        """Получение исторических данных OHLCV"""
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Получение текущего тикера"""
        pass

    # Утилитарные методы с реализацией по умолчанию

    async def close_position(self, symbol: str) -> bool:
        """Закрытие позиции по символу"""
        positions = await self.get_positions(symbol)
        if not positions:
            return False

        position = positions[0]
        side = 'sell' if position.side == 'long' else 'buy'

        order = Order(
            id='',
            symbol=symbol,
            side=side,
            type='market',
            quantity=position.quantity,
            reduce_only=True
        )

        result = await self.place_order(order)
        return result.status in ['filled', 'closed']

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Получение текущей цены"""
        ticker = await self.get_ticker(symbol)
        return ticker.get('last_price') if ticker else None

    def calculate_fees(self,
                       quantity: float,
                       price: float,
                       is_maker: bool = False) -> float:
        """Расчет комиссий (базовая реализация)"""
        fee_rate = 0.0004 if is_maker else 0.0006  # 0.04% maker, 0.06% taker
        return quantity * price * fee_rate