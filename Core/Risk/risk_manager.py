"""
Система управления рисками.
Контролирует размеры позиций, максимальные убытки и ограничения.
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from enum import Enum


class RiskLevel(Enum):
    """Уровни риска"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskMetrics:
    """Метрики риска"""
    total_exposure: float = 0.0
    max_position_size: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class TradeRecord:
    """Запись о сделке"""
    id: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'buy' or 'sell'
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    fees: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""


class RiskManager:
    """Менеджер управления рисками"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_level = RiskLevel(config.get("risk_level", "medium"))

        # Ограничения
        self.max_position_size = config.get("max_position_size", 1000.0)
        self.max_daily_loss = config.get("max_daily_loss", 0.05)  # 5%
        self.max_portfolio_risk = config.get("max_portfolio_risk", 0.02)  # 2% на сделку
        self.max_open_positions = config.get("max_open_positions", 5)
        self.max_leverage = config.get("max_leverage", 3)

        # Дневные лимиты
        self.daily_loss_limit = config.get("daily_loss_limit", 0.1)  # 10%
        self.daily_trade_limit = config.get("daily_trade_limit", 20)

        # Текущее состояние
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.open_positions: Dict[str, Dict] = {}
        self.trade_history: List[TradeRecord] = []

        # Время начала дня для сброса дневных лимитов
        self.day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Максимальная просадка
        self.peak_balance = config.get("initial_balance", 10000.0)
        self.current_balance = self.peak_balance
        self.max_drawdown = 0.0

    def update_day(self):
        """Обновление дневных счетчиков"""
        now = datetime.now()
        if now.date() > self.day_start.date():
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def calculate_position_size(self, entry_price: float, stop_loss: float,
                                account_balance: float, risk_per_trade: float = None) -> float:
        """Расчет размера позиции на основе риска"""
        if risk_per_trade is None:
            risk_per_trade = self.max_portfolio_risk

        # Риск в процентах от цены входа
        price_risk_pct = abs(entry_price - stop_loss) / entry_price

        # Минимальный риск 0.5%
        if price_risk_pct < 0.005:
            price_risk_pct = 0.005

        # Максимальный риск 5%
        if price_risk_pct > 0.05:
            price_risk_pct = 0.05

        # Расчет размера позиции
        risk_amount = account_balance * risk_per_trade
        position_size = risk_amount / (entry_price * price_risk_pct)

        # Ограничение максимальным размером позиции
        max_size_usd = self.max_position_size
        max_size = max_size_usd / entry_price

        return min(position_size, max_size)

    def validate_trade(self, symbol: str, quantity: float, price: float,
                       side: str, stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> Tuple[bool, str]:
        """Валидация сделки перед исполнением"""

        # Обновляем дневные счетчики
        self.update_day()

        # 1. Проверка максимального количества открытых позиций
        if len(self.open_positions) >= self.max_open_positions:
            return False, "Max open positions reached"

        # 2. Проверка размера позиции
        position_value = quantity * price
        if position_value > self.max_position_size:
            return False, f"Position size {position_value:.2f} exceeds max {self.max_position_size:.2f}"

        # 3. Проверка общего экспозьюра
        new_exposure = self.current_exposure + position_value
        max_exposure = self.peak_balance * self.max_leverage
        if new_exposure > max_exposure:
            return False, f"Exposure {new_exposure:.2f} exceeds max {max_exposure:.2f}"

        # 4. Проверка дневного лимита убытков
        if self.daily_pnl < -self.peak_balance * self.daily_loss_limit:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"

        # 5. Проверка дневного лимита сделок
        if self.daily_trades >= self.daily_trade_limit:
            return False, f"Daily trade limit reached: {self.daily_trades}"

        # 6. Проверка стоп-лосса и тейк-профита
        if stop_loss:
            risk_pct = abs(price - stop_loss) / price
            if risk_pct > 0.1:  # Максимальный риск 10%
                return False, f"Stop loss risk too high: {risk_pct:.2%}"

        if take_profit:
            reward_pct = abs(take_profit - price) / price
            if reward_pct > 0.2:  # Максимальная прибыль 20%
                return False, f"Take profit too high: {reward_pct:.2%}"

        # 7. Проверка соотношения риск/прибыль
        if stop_loss and take_profit:
            risk = abs(price - stop_loss)
            reward = abs(take_profit - price)
            if reward / risk < 1:  # Минимальное соотношение 1:1
                return False, f"Risk/reward ratio too low: {reward / risk:.2f}"

        return True, "Trade validated"

    def record_trade_entry(self, trade_id: str, symbol: str, entry_price: float,
                           quantity: float, side: str, stop_loss: Optional[float],
                           take_profit: Optional[float]):
        """Запись входа в сделку"""
        self.open_positions[symbol] = {
            'id': trade_id,
            'entry_time': datetime.now(),
            'entry_price': entry_price,
            'quantity': quantity,
            'side': side,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

        self.current_exposure += quantity * entry_price
        self.daily_trades += 1

    def record_trade_exit(self, symbol: str, exit_price: float,
                          fees: float = 0.0, reason: str = ""):
        """Запись выхода из сделки"""
        if symbol not in self.open_positions:
            return

        position = self.open_positions.pop(symbol)

        # Расчет PnL
        if position['side'] == 'buy':
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']

        pnl -= fees
        pnl_percentage = (pnl / (position['entry_price'] * position['quantity'])) * 100

        # Обновление метрик
        self.current_exposure -= position['quantity'] * position['entry_price']
        self.daily_pnl += pnl
        self.current_balance += pnl

        # Обновление максимальной просадки
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Запись в историю
        trade_record = TradeRecord(
            id=position['id'],
            symbol=symbol,
            entry_time=position['entry_time'],
            exit_time=datetime.now(),
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            side=position['side'],
            pnl=pnl,
            pnl_percentage=pnl_percentage,
            fees=fees,
            stop_loss=position['stop_loss'],
            take_profit=position['take_profit'],
            reason=reason
        )

        self.trade_history.append(trade_record)

        return trade_record

    def check_position_risk(self, symbol: str, current_price: float) -> Optional[str]:
        """Проверка риска по открытой позиции"""
        if symbol not in self.open_positions:
            return None

        position = self.open_positions[symbol]

        # Проверка стоп-лосса
        if position['stop_loss']:
            if position['side'] == 'buy' and current_price <= position['stop_loss']:
                return "stop_loss"
            elif position['side'] == 'sell' and current_price >= position['stop_loss']:
                return "stop_loss"

        # Проверка тейк-профита
        if position['take_profit']:
            if position['side'] == 'buy' and current_price >= position['take_profit']:
                return "take_profit"
            elif position['side'] == 'sell' and current_price <= position['take_profit']:
                return "take_profit"

        return None

    def get_risk_metrics(self) -> RiskMetrics:
        """Получение текущих метрик риска"""
        # Расчет дополнительных метрик
        if self.trade_history:
            winning_trades = [t for t in self.trade_history if t.pnl > 0]
            losing_trades = [t for t in self.trade_history if t.pnl <= 0]

            win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0

            total_profit = sum(t.pnl for t in winning_trades)
            total_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            # Расчет Sharpe ratio (упрощенный)
            returns = [t.pnl_percentage / 100 for t in self.trade_history[-30:]]
            if len(returns) > 1:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365)
            else:
                sharpe = 0.0
        else:
            win_rate = 0.0
            profit_factor = 0.0
            sharpe = 0.0

        return RiskMetrics(
            total_exposure=self.current_exposure,
            max_position_size=self.max_position_size,
            daily_pnl=self.daily_pnl,
            daily_trades=self.daily_trades,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            profit_factor=profit_factor
        )

    def reset_daily_limits(self):
        """Сброс дневных лимитов"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    def get_trade_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Получение статистики по сделкам"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [t for t in self.trade_history if t.exit_time > cutoff_date]

        if not recent_trades:
            return {}

        winning_trades = [t for t in recent_trades if t.pnl > 0]
        losing_trades = [t for t in recent_trades if t.pnl <= 0]

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0

        return {
            'total_trades': len(recent_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(recent_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': abs(sum(t.pnl for t in winning_trades) /
                                 sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        }