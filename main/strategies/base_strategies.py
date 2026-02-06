from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd


@dataclass
class Signal:
    symbol: str
    action: str  # BUY/SELL/HOLD
    strength: float  # 0-1 confidence
    price: float
    timestamp: pd.Timestamp
    metadata: Dict[str, Any] = None


class BaseStrategy(ABC):
    """Базовый класс для всех стратегий"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.signals_history = []

    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Signal:
        """Анализ данных и генерация сигнала"""
        pass

    def update_parameters(self, **params):
        """Динамическое обновление параметров"""
        self.config.update(params)

    def add_ml_signal(self, ml_prediction: float):
        """Интеграция ML прогноза"""
        self.ml_prediction = ml_prediction