
import asyncio
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode
import aiohttp
import pandas as pd
import json

from .base_broker import BaseBroker, Order, OrderType, OrderSide, Position, PositionSide, Balance


class BinanceFuturesBroker(BaseBroker):
    """Брокер для Binance Futures"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")
        self.testnet = config.get("testnet", True)

        if self.testnet:
            self.base_url = "https://testnet.binancefuture.com"
            self.ws_url = "wss://stream.binancefuture.com/ws"
        else:
            self.base_url = "https://fapi.binance.com"
            self.ws_url = "wss://fstream.binance.com/ws"

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connections: Dict[str, Any] = {}

    async def connect(self) -> bool:
        """Подключение к Binance Futures"""
        try:
            self._session = aiohttp.ClientSession(
                headers={
                    "X-MBX-APIKEY": self.api_key,
                    "Content-Type": "application/json"
                }
            )

            # Тестовый запрос для проверки подключения
            server_time = await self._get_server_time()
            if server_time:
                self.connected = True
                print(f"Connected to Binance Futures (Testnet: {self.testnet})")
                return True

        except Exception as e:
            print(f"Connection error: {e}")

        return False

    async def disconnect(self) -> bool:
        """Отключение от Binance Futures"""
        if self._session:
            await self._session.close()

        # Закрываем WebSocket соединения
        for ws in self._ws_connections.values():
            await ws.close()

        self.connected = False
        return True

    async def _make_request(self, method: str, endpoint: str,
                            signed: bool = False, **params) -> Dict[str, Any]:
        """Выполнение HTTP запроса к API"""
        url = f"{self.base_url}{endpoint}"

        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = urlencode(params)
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature

        try:
            async with self._session.request(method, url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"API Error {response.status}: {error_text}")
        except Exception as e:
            print(f"Request error: {e}")
            raise

    async def _get_server_time(self) -> int:
        """Получение времени сервера"""
        endpoint = "/fapi/v1/time"
        response = await self._make_request("GET", endpoint)
        return response.get("serverTime", 0)

    async def get_balance(self) -> Balance:
        """Получение баланса"""
        endpoint = "/fapi/v2/balance"
        response = await self._make_request("GET", endpoint, signed=True)

        # Ищем USDT баланс
        usdt_balance = next(
            (item for item in response if item['asset'] == 'USDT'),
            {'availableBalance': '0', 'totalWalletBalance': '0'}
        )

        return Balance(
            total=float(usdt_balance.get('totalWalletBalance', 0)),
            available=float(usdt_balance.get('availableBalance', 0)),
            locked=float(usdt_balance.get('totalWalletBalance', 0)) -
                   float(usdt_balance.get('availableBalance', 0))
        )

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Получение позиции по символу"""
        endpoint = "/fapi/v2/positionRisk"
        params = {"symbol": symbol.replace("/", "")}
        response = await self._make_request("GET", endpoint, signed=True, **params)

        if not response or len(response) == 0:
            return None

        position_data = response[0]
        position_amt = float(position_data.get('positionAmt', 0))

        if position_amt == 0:
            return None

        side = PositionSide.LONG if position_amt > 0 else PositionSide.SHORT

        return Position(
            symbol=symbol,
            side=side,
            quantity=abs(position_amt),
            entry_price=float(position_data.get('entryPrice', 0)),
            current_price=float(position_data.get('markPrice', 0)),
            unrealized_pnl=float(position_data.get('unRealizedProfit', 0)),
            leverage=int(float(position_data.get('leverage', 1))),
            liquidation_price=float(position_data.get('liquidationPrice', 0)) if position_data.get(
                'liquidationPrice') else None,
            margin=float(position_data.get('positionInitialMargin', 0))
        )

    async def get_positions(self) -> List[Position]:
        """Получение всех позиций"""
        endpoint = "/fapi/v2/positionRisk"
        response = await self._make_request("GET", endpoint, signed=True)

        positions = []
        for pos_data in response:
            position_amt = float(pos_data.get('positionAmt', 0))
            if position_amt != 0:
                symbol = pos_data.get('symbol', '')
                side = PositionSide.LONG if position_amt > 0 else PositionSide.SHORT

                position = Position(
                    symbol=symbol,
                    side=side,
                    quantity=abs(position_amt),
                    entry_price=float(pos_data.get('entryPrice', 0)),
                    current_price=float(pos_data.get('markPrice', 0)),
                    unrealized_pnl=float(pos_data.get('unRealizedProfit', 0)),
                    leverage=int(float(pos_data.get('leverage', 1))),
                    liquidation_price=float(pos_data.get('liquidationPrice', 0)) if pos_data.get(
                        'liquidationPrice') else None,
                    margin=float(pos_data.get('positionInitialMargin', 0))
                )
                positions.append(position)

        self.positions = {pos.symbol: pos for pos in positions}
        return positions

    async def place_order(self, order: Order) -> Order:
        """Размещение ордера"""
        endpoint = "/fapi/v1/order"

        params = {
            "symbol": order.symbol.replace("/", ""),
            "side": order.side.value.upper(),
            "type": order.type.value.upper(),
            "quantity": order.quantity,
            "reduceOnly": order.reduce_only,
            "newOrderRespType": "RESULT"
        }

        if order.price:
            params["price"] = order.price

        if order.stop_price and order.type in [OrderType.STOP, OrderType.STOP_MARKET]:
            params["stopPrice"] = order.stop_price

        if order.type == OrderType.LIMIT:
            params["timeInForce"] = order.time_in_force

        response = await self._make_request("POST", endpoint, signed=True, **params)

        order.id = str(response.get("orderId", ""))
        order.status = response.get("status", "").lower()
        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера"""
        endpoint = "/fapi/v1/order"
        params = {"orderId": order_id}

        try:
            response = await self._make_request("DELETE", endpoint, signed=True, **params)
            return response.get("status", "") == "CANCELED"
        except:
            return False

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Получение информации об ордере"""
        endpoint = "/fapi/v1/order"
        params = {"orderId": order_id}

        try:
            response = await self._make_request("GET", endpoint, signed=True, **params)

            order_type = OrderType(response.get("type", "").lower())
            order_side = OrderSide(response.get("side", "").lower())

            return Order(
                id=str(response.get("orderId", "")),
                symbol=response.get("symbol", ""),
                side=order_side,
                type=order_type,
                price=float(response.get("price", 0)),
                quantity=float(response.get("origQty", 0)),
                stop_price=float(response.get("stopPrice", 0)) if response.get("stopPrice") else None,
                status=response.get("status", "").lower()
            )
        except:
            return None

    async def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Получение всех ордеров"""
        endpoint = "/fapi/v1/openOrders"
        params = {}
        if symbol:
            params["symbol"] = symbol.replace("/", "")

        response = await self._make_request("GET", endpoint, signed=True, **params)

        orders = []
        for order_data in response:
            order_type = OrderType(order_data.get("type", "").lower())
            order_side = OrderSide(order_data.get("side", "").lower())

            order = Order(
                id=str(order_data.get("orderId", "")),
                symbol=order_data.get("symbol", ""),
                side=order_side,
                type=order_type,
                price=float(order_data.get("price", 0)),
                quantity=float(order_data.get("origQty", 0)),
                stop_price=float(order_data.get("stopPrice", 0)) if order_data.get("stopPrice") else None,
                status=order_data.get("status", "").lower()
            )
            orders.append(order)

        self.orders = {order.id: order for order in orders}
        return orders

    async def get_ohlcv(self, symbol: str, timeframe: str,
                        limit: int = 500, since: Optional[int] = None) -> pd.DataFrame:
        """Получение исторических данных OHLCV"""
        endpoint = "/fapi/v1/klines"

        params = {
            "symbol": symbol.replace("/", ""),
            "interval": timeframe,
            "limit": limit
        }

        if since:
            params["startTime"] = since

        response = await self._make_request("GET", endpoint, **params)

        df = pd.DataFrame(
            response,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                     'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                     'taker_buy_quote', 'ignore']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Конвертация типов
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)

        return df[['open', 'high', 'low', 'close', 'volume']]

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Получение текущей цены"""
        endpoint = "/fapi/v1/ticker/24hr"
        params = {"symbol": symbol.replace("/", "")}

        response = await self._make_request("GET", endpoint, **params)
        return response

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Установка кредитного плеча"""
        endpoint = "/fapi/v1/leverage"
        params = {
            "symbol": symbol.replace("/", ""),
            "leverage": leverage
        }

        try:
            response = await self._make_request("POST", endpoint, signed=True, **params)
            return response.get("leverage", 0) == leverage
        except:
            return False

    def calculate_fees(self, symbol: str, quantity: float, price: float,
                       is_maker: bool = False) -> float:
        """Расчет комиссий для Binance"""
        # Комиссии Binance Futures: 0.02% для мейкера, 0.04% для тейкера
        fee_rate = 0.0002 if is_maker else 0.0004
        return quantity * price * fee_rate