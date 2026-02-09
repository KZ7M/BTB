"""
Реализация брокера для Binance Futures API.
Поддерживает тестовую сеть (testnet) и реальную торговлю.
"""
import hashlib
import hmac
import time
from typing import Dict, List, Optional, Any
import aiohttp
import pandas as pd
import numpy as np
from urllib.parse import urlencode

from .base_broker import BaseBroker, Order, Position, Balance


class BinanceFuturesBroker(BaseBroker):
    """Реализация брокера для Binance Futures"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.testnet = config.get('testnet', True)

        # URL в зависимости от режима
        if self.testnet:
            self.base_url = "https://testnet.binancefuture.com"
            self.ws_url = "wss://stream.binancefuture.com/ws"
        else:
            self.base_url = "https://fapi.binance.com"
            self.ws_url = "wss://fstream.binance.com/ws"

        self.session: Optional[aiohttp.ClientSession] = None
        self.recv_window = 5000

    async def connect(self) -> bool:
        """Подключение к Binance Futures"""
        try:
            headers = {
                'X-MBX-APIKEY': self.api_key,
                'Content-Type': 'application/json'
            }
            self.session = aiohttp.ClientSession(headers=headers)

            # Проверяем подключение
            server_time = await self._get_server_time()
            if server_time:
                self.is_connected = True
                return True
        except Exception as e:
            print(f"Connection error: {e}")
        return False

    async def disconnect(self) -> bool:
        """Отключение от Binance Futures"""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_connected = False
        return True

    async def _make_request(self,
                            method: str,
                            endpoint: str,
                            signed: bool = False,
                            **params) -> Dict[str, Any]:
        """Выполнение HTTP запроса"""
        url = f"{self.base_url}{endpoint}"

        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = self.recv_window

            query_string = urlencode(params)
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature

        try:
            async with self.session.request(method, url, params=params) as response:
                data = await response.json()
                if response.status != 200:
                    raise Exception(f"API Error: {data}")
                return data
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
        for asset in response:
            if asset['asset'] == 'USDT':
                return Balance(
                    total=float(asset['balance']),
                    available=float(asset['availableBalance']),
                    currency='USDT'
                )
        return Balance(total=0, available=0, currency='USDT')

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Получение позиций"""
        endpoint = "/fapi/v2/positionRisk"
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()

        response = await self._make_request("GET", endpoint, signed=True, **params)

        positions = []
        for pos in response:
            position_amt = float(pos['positionAmt'])
            if position_amt != 0:
                positions.append(Position(
                    symbol=pos['symbol'],
                    side='long' if position_amt > 0 else 'short',
                    quantity=abs(position_amt),
                    entry_price=float(pos['entryPrice']),
                    current_price=float(pos['markPrice']),
                    unrealized_pnl=float(pos['unRealizedProfit']),
                    leverage=int(float(pos['leverage'])),
                    liquidation_price=float(pos['liquidationPrice']) if pos['liquidationPrice'] else None
                ))

        return positions

    async def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Получение ордеров"""
        endpoint = "/fapi/v1/openOrders"
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()

        response = await self._make_request("GET", endpoint, signed=True, **params)

        orders = []
        for order_data in response:
            orders.append(Order(
                id=str(order_data['orderId']),
                symbol=order_data['symbol'],
                side=order_data['side'].lower(),
                type=order_data['type'].lower(),
                quantity=float(order_data['origQty']),
                price=float(order_data['price']) if order_data['price'] != '0' else None,
                status=order_data['status'].lower(),
                timestamp=pd.to_datetime(order_data['time'], unit='ms')
            ))

        return orders

    async def place_order(self, order: Order) -> Order:
        """Размещение ордера"""
        endpoint = "/fapi/v1/order"

        params = {
            'symbol': order.symbol.upper(),
            'side': order.side.upper(),
            'type': order.type.upper(),
            'quantity': order.quantity,
            'reduceOnly': order.reduce_only
        }

        if order.price:
            params['price'] = order.price

        if order.type in ['stop', 'stop_market']:
            params['stopPrice'] = order.price

        response = await self._make_request("POST", endpoint, signed=True, **params)

        order.id = str(response['orderId'])
        order.status = response['status'].lower()
        order.timestamp = pd.to_datetime(response['updateTime'], unit='ms')

        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Отмена ордера"""
        endpoint = "/fapi/v1/order"
        params = {
            'symbol': symbol.upper(),
            'orderId': order_id
        }

        try:
            response = await self._make_request("DELETE", endpoint, signed=True, **params)
            return response['status'] == 'CANCELED'
        except:
            return False

    async def get_ohlcv(self,
                        symbol: str,
                        timeframe: str,
                        limit: int = 500,
                        since: Optional[int] = None) -> pd.DataFrame:
        """Получение исторических данных OHLCV"""
        endpoint = "/fapi/v1/klines"

        params = {
            'symbol': symbol.upper(),
            'interval': timeframe,
            'limit': limit
        }

        if since:
            params['startTime'] = since

        response = await self._make_request("GET", endpoint, **params)

        df = pd.DataFrame(
            response,
            columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ]
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Конвертируем числовые колонки
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)

        return df[['open', 'high', 'low', 'close', 'volume']]

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Получение текущего тикера"""
        endpoint = "/fapi/v1/ticker/24hr"
        params = {'symbol': symbol.upper()}

        response = await self._make_request("GET", endpoint, **params)

        return {
            'symbol': response['symbol'],
            'last_price': float(response['lastPrice']),
            'bid_price': float(response['bidPrice']),
            'ask_price': float(response['askPrice']),
            'volume': float(response['volume']),
            'quote_volume': float(response['quoteVolume'])
        }

    # Переопределяем метод расчета комиссий для Binance
    def calculate_fees(self,
                       quantity: float,
                       price: float,
                       is_maker: bool = False) -> float:
        """Расчет комиссий для Binance Futures"""
        # Binance Futures: 0.02% maker, 0.04% taker
        fee_rate = 0.0002 if is_maker else 0.0004
        return quantity * price * fee_rate