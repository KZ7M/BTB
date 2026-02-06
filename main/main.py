import asyncio
import pandas as pd
from typing import Dict, Any
from strategies.fibonacci_bounce import FibonacciBounceStrategy
from utils.notifications import TelegramNotifier
from ml.model_handler import MLModelHandler
from core.risk_manager import RiskManager


class TradingEngine:
    """Основной торговый движок"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy = FibonacciBounceStrategy(config['strategy'])
        self.notifier = TelegramNotifier(
            config['telegram']['token'],
            config['telegram']['chat_id']
        )
        self.ml_handler = MLModelHandler(config['ml']['model_path'])
        self.risk_manager = RiskManager(config['risk'])
        self.running = False

    async def run(self):
        """Запуск торгового движка"""
        self.running = True
        print("Торговый движок запущен...")

        while self.running:
            try:
                # 1. Получение данных
                data = await self.get_market_data()

                # 2. Генерация ML фичей и прогноз
                features = self.ml_handler.prepare_features(data)
                ml_prediction = self.ml_handler.predict(features)

                # 3. Получение сигнала от стратегии
                signal = self.strategy.analyze(data)

                # 4. Корректировка с учетом ML
                if ml_prediction > 0.7:  # Сильный ML сигнал
                    signal.strength = min(1.0, signal.strength * 1.2)
                elif ml_prediction < 0.3:
                    signal.strength = max(0.0, signal.strength * 0.8)

                # 5. Проверка рисков
                if self.risk_manager.check_signal(signal):
                    # 6. Исполнение
                    await self.execute_signal(signal)

                    # 7. Уведомление
                    self.notifier.send_signal(signal)

                # 8. Логирование метрик
                metrics = self.collect_metrics(signal, data)
                self.notifier.send_metrics(metrics)

                await asyncio.sleep(self.config['update_interval'])

            except Exception as e:
                print(f"Ошибка: {e}")
                await asyncio.sleep(5)