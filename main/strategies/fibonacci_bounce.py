import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal


class FibonacciBounceStrategy(BaseStrategy):
    """Стратегия отскока от уровней Фибоначчи"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.min_touch_count = config.get('min_touch_count', 3)
        self.support_zones = {}

    def calculate_fib_levels(self, high: float, low: float) -> Dict[float, float]:
        """Расчет уровней Фибоначчи"""
        diff = high - low
        return {level: high - level * diff for level in self.fib_levels}

    def detect_bounce(self, price: float, fib_levels: Dict) -> Optional[float]:
        """Определение отскока от уровня"""
        threshold = self.config.get('threshold_percent', 0.001)

        for level, value in fib_levels.items():
            if abs(price - value) / value < threshold:
                return level
        return None

    def analyze(self, data: pd.DataFrame) -> Signal:
        daily_high = data['high'].max()
        daily_low = data['low'].min()

        fib_levels = self.calculate_fib_levels(daily_high, daily_low)
        current_price = data['close'].iloc[-1]

        # Проверяем отскок
        bounce_level = self.detect_bounce(current_price, fib_levels)

        if bounce_level:
            # Чем больше линий, тем сильнее сигнал
            confidence = min(0.9, len(fib_levels) * 0.15)

            # Определяем направление
            if current_price > data['close'].iloc[-2]:  # Восходящий отскок
                return Signal(
                    symbol=self.config['symbol'],
                    action='BUY',
                    strength=confidence,
                    price=current_price,
                    timestamp=data.index[-1],
                    metadata={'fib_level': bounce_level}
                )
            else:  # Нисходящий отскок
                return Signal(
                    symbol=self.config['symbol'],
                    action='SELL',
                    strength=confidence,
                    price=current_price,
                    timestamp=data.index[-1],
                    metadata={'fib_level': bounce_level}
                )

        return Signal(
            symbol=self.config['symbol'],
            action='HOLD',
            strength=0.0,
            price=current_price,
            timestamp=data.index[-1]
        )