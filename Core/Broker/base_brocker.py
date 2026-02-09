"""
Абстрактный интерфейс для подключения к биржам.
Реализует паттерн Стратегия для поддержки различных брокеров.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from enum import Enum


class OrderType(Enum):
    """Типы ордеров"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_MARKET = "stop_market"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Сторона ордера"""
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    """Сторона позиции"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Класс ордера"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    price: Optional[float] = None
    quantity: float = 0.0
    stop_price: Optional[float] = None
    reduce_only: bool = False
    time_in_force: str = "GTC"
    status: str = "new"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.status == "new":
            self.created_at = datetime.now()
        self.updated_at = datetime.now()


@dataclass
class Position:
    """Класс позиции"""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: int = 1
    liquidation_price: Optional[float] = None
    margin: float = 0.0
    update_time: datetime = field(default_factory=datetime.now)

    def update(self, current_price: float):
        """Обновление позиции"""
        self.current_price = current_price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity


@dataclass
class Balance:
    """Баланс"""
    total: float
    available: float
    locked: float
    timestamp: datetime = field(default_factory=datetime.now)


class BaseBroker(ABC):
    """Абстрактный базовый класс брокера"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.connected = False
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self._session = None

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
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Получение позиции по символу"""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Получение всех позиций"""
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Размещение ордера"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера"""
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Получение информации об ордере"""
        pass

    @abstractmethod
    async def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Получение всех ордеров"""
        pass

    @abstractmethod
    async def get_ohlcv(self, symbol: str, timeframe: str,
                        limit: int = 500, since: Optional[int] = None) -> pd.DataFrame:
        """Получение исторических данных OHLCV"""
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Получение текущей цены"""
        pass

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Установка кредитного плеча"""
        pass

    async def place_market_order(self, symbol: str, side: OrderSide,
                                 quantity: float, reduce_only: bool = False) -> Order:
        """Размещение рыночного ордера"""
        order = Order(
            id="",
            symbol=symbol,
            side=side,
            type=OrderType.MARKET,
            quantity=quantity,
            reduce_only=reduce_only
        )
        return await self.place_order(order)

    async def place_limit_order(self, symbol: str, side: OrderSide,
                                price: float, quantity: float,
                                reduce_only: bool = False) -> Order:
        """Размещение лимитного ордера"""
        order = Order(
            id="",
            symbol=symbol,
            side=side,
            type=OrderType.LIMIT,
            price=price,
            quantity=quantity,
            reduce_only=reduce_only
        )
        return await self.place_order(order)

    async def close_position(self, symbol: str) -> bool:
        """Закрытие позиции по символу"""
        position = await self.get_position(symbol)
        if not position:
            return False

        side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        order = await self.place_market_order(
            symbol=symbol,
            side=side,
            quantity=position.quantity,
            reduce_only=True
        )
        return order.status in ["filled", "closed"]

    def calculate_fees(self, symbol: str, quantity: float, price: float,
                       is_maker: bool = False) -> float:
        """Расчет комиссий"""
        # Базовая реализация, должна быть переопределена в конкретных брокерах
        fee_rate = 0.0004 if is_maker else 0.0006  # 0.04% / 0.06%
        return quantity * price * fee_rate

    async def health_check(self) -> Dict[str, Any]:
        """Проверка состояния подключения"""
        try:
            balance = await self.get_balance()
            return {
                "connected": self.connected,
                "balance_available": balance.available,
                "positions_count": len(self.positions),
                "orders_count": len(self.orders)
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }