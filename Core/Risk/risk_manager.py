"""
Модуль управления рисками.
Контролирует размеры позиций, максимальные убытки и ограничения.
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np


@dataclass
class RiskMetrics:
    """Метрики риска"""
    total_exposure: float = 0.0
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
    side: str
    pnl: float = 0.0
    pnl_percent: float = 0.0
    fees: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""


class RiskManager:
    """Менеджер управления рисками"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Основные параметры риска
        self.max_position_size = config.get('max_position_size', 1000.0)
        self.max_daily_loss_pct = config.get('max_daily_loss_pct', 0.05)  # 5%
        self.max_portfolio_risk_pct = config.get('max_portfolio_risk_pct', 0.02)  # 2%
        self.max_open_positions = config.get('max_open_positions', 5)
        self.max_leverage = config.get('max_leverage', 3)

        # Дневные лимиты
        self.daily_loss_limit_pct = config.get('daily_loss_limit_pct', 0.1)  # 10%
        self.daily_trade_limit = config.get('daily_trade_limit', 20)

        # Состояние
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.open_positions: Dict[str, TradeRecord] = {}
        self.trade_history: List[TradeRecord] = []

        # Для расчета просадки
        self.peak_balance = config.get('initial_balance', 10000.0)
        self.current_balance = self.peak_balance
        self.max_drawdown = 0.0

        # Время начала дня
        self.day_start = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    def calculate_position_size(self,
                                entry_price: float,
                                stop_loss: float,
                                account_balance: float,
                                risk_per_trade_pct: Optional[float] = None) -> float:
        """
        Расчет размера позиции на основе риска

        Args:
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса
            account_balance: Баланс счета
            risk_per_trade_pct: Риск на сделку в процентах
        """
        if risk_per_trade_pct is None:
            risk_per_trade_pct = self.max_portfolio_risk_pct

        # Рассчитываем риск в процентах от цены
        price_risk_pct = abs(entry_price - stop_loss) / entry_price

        # Минимальный и максимальный риск
        price_risk_pct = max(0.005, min(price_risk_pct, 0.05))  # 0.5% - 5%

        # Сумма риска на сделку
        risk_amount = account_balance * risk_per_trade_pct

        # Размер позиции
        position_size = risk_amount / (entry_price * price_risk_pct)

        # Ограничение максимальным размером позиции
        max_size_usd = self.max_position_size
        max_size = max_size_usd / entry_price

        return min(position_size, max_size)

    def validate_trade(self,
                       symbol: str,
                       quantity: float,
                       price: float,
                       side: str,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> Tuple[bool, str]:
        """
        Валидация сделки перед исполнением

        Returns:
            (is_valid, message)
        """
        # Обновляем дневные счетчики
        self._update_day()

        # 1. Максимальное количество открытых позиций
        if len(self.open_positions) >= self.max_open_positions:
            return False, "Достигнут лимит открытых позиций"

        # 2. Размер позиции
        position_value = quantity * price
        if position_value > self.max_position_size:
            return False, f"Размер позиции {position_value:.2f} превышает максимум {self.max_position_size:.2f}"

        # 3. Общий экспозьюр
        new_exposure = self.current_exposure + position_value
        max_exposure = self.peak_balance * self.max_leverage
        if new_exposure > max_exposure:
            return False, f"Общий экспозьюр {new_exposure:.2f} превышает максимум {max_exposure:.2f}"

        # 4. Дневной лимит убытков
        if self.daily_pnl < -self.peak_balance * self.daily_loss_limit_pct:
            return False, f"Достигнут дневной лимит убытков: {self.daily_pnl:.2f}"

        # 5. Дневной лимит сделок
        if self.daily_trades >= self.daily_trade_limit:
            return False, f"Достигнут дневной лимит сделок: {self.daily_trades}"

        # 6. Проверка стоп-лосса
        if stop_loss:
            risk_pct = abs(price - stop_loss) / price
            if risk_pct > 0.1:  # Максимальный риск 10%
                return False, f"Слишком большой риск стоп-лосса: {risk_pct:.2%}"

            # Для long: stop_loss должен быть ниже цены входа
            if side == 'buy' and stop_loss >= price:
                return False, "Для long позиции стоп-лосс должен быть ниже цены входа"

            # Для short: stop_loss должен быть выше цены входа
            if side == 'sell' and stop_loss <= price:
                return False, "Для short позиции стоп-лосс должен быть выше цены входа"

        # 7. Проверка тейк-профита
        if take_profit:
            reward_pct = abs(take_profit - price) / price
            if reward_pct > 0.2:  # Максимальная прибыль 20%
                return False, f"Слишком большой тейк-профит: {reward_pct:.2%}"

        # 8. Соотношение риск/прибыль
        if stop_loss and take_profit:
            risk = abs(price - stop_loss)
            reward = abs(take_profit - price)
            if reward / risk < 1.0:  # Минимум 1:1
                return False, f"Соотношение риск/прибыль слишком низкое: {reward / risk:.2f}"

        return True, "Сделка валидна"

    def record_trade_entry(self,
                           trade_id: str,
                           symbol: str,
                           entry_price: float,
                           quantity: float,
                           side: str,
                           stop_loss: Optional[float] = None,
                           take_profit: Optional[float] = None):
        """Запись входа в сделку"""
        trade = TradeRecord(
            id=trade_id,
            symbol=symbol,
            entry_time=datetime.now(),
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        self.open_positions[symbol] = trade
        self.current_exposure += quantity * entry_price
        self.daily_trades += 1

    def record_trade_exit(self,
                          symbol: str,
                          exit_price: float,
                          fees: float = 0.0,
                          reason: str = "") -> Optional[TradeRecord]:
        """Запись выхода из сделки"""
        if symbol not in self.open_positions:
            return None

        trade = self.open_positions.pop(symbol)

        # Расчет PnL
        if trade.side == 'buy':
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity

        pnl -= fees
        pnl_percent = (pnl / (trade.entry_price * trade.quantity)) * 100

        # Обновление метрик
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_percent = pnl_percent
        trade.fees = fees
        trade.reason = reason

        self.current_exposure -= trade.quantity * trade.entry_price
        self.daily_pnl += pnl
        self.current_balance += pnl

        # Обновление максимальной просадки
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        self.trade_history.append(trade)
        return trade

    def check_position_risk(self,
                            symbol: str,
                            current_price: float) -> Optional[str]:
        """Проверка риска по открытой позиции"""
        if symbol not in self.open_positions:
            return None

        trade = self.open_positions[symbol]

        # Проверка стоп-лосса
        if trade.stop_loss:
            if trade.side == 'buy' and current_price <= trade.stop_loss:
                return "stop_loss"
            elif trade.side == 'sell' and current_price >= trade.stop_loss:
                return "stop_loss"

        # Проверка тейк-профита
        if trade.take_profit:
            if trade.side == 'buy' and current_price >= trade.take_profit:
                return "take_profit"
            elif trade.side == 'sell' and current_price <= trade.take_profit:
                return "take_profit"

        return None

    def get_risk_metrics(self) -> RiskMetrics:
        """Получение текущих метрик риска"""
        # Расчет дополнительных метрик
        if self.trade_history:
            winning_trades = [t for t in self.trade_history if t.pnl > 0]
            losing_trades = [t for t in self.trade_history if t.pnl <= 0]

            win_rate = len(winning_trades) / len(self.trade_history)

            total_profit = sum(t.pnl for t in winning_trades)
            total_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            # Упрощенный расчет Sharpe ratio
            returns = [t.pnl_percent / 100 for t in self.trade_history[-30:]]
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
            daily_pnl=self.daily_pnl,
            daily_trades=self.daily_trades,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            profit_factor=profit_factor
        )

    def get_trade_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Получение статистики по сделкам за последние N дней"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [t for t in self.trade_history
                         if t.exit_time and t.exit_time > cutoff_date]

        if not recent_trades:
            return {}

        winning_trades = [t for t in recent_trades if t.pnl > 0]
        losing_trades = [t for t in recent_trades if t.pnl <= 0]

        stats = {
            'total_trades': len(recent_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(recent_trades),
            'total_pnl': sum(t.pnl for t in recent_trades),
            'avg_pnl': np.mean([t.pnl for t in recent_trades]),
            'largest_win': max([t.pnl for t in winning_trades]) if winning_trades else 0,
            'largest_loss': min([t.pnl for t in losing_trades]) if losing_trades else 0,
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        }

        return stats

    def _update_day(self):
        """Обновление дневных счетчиков"""
        now = datetime.now()
        if now.date() > self.day_start.date():
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def reset(self):
        """Сброс всех метрик"""
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.open_positions.clear()
        self.trade_history.clear()
        self.peak_balance = self.config.get('initial_balance', 10000.0)
        self.current_balance = self.peak_balance
        self.max_drawdown = 0.0
        self.day_start = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )