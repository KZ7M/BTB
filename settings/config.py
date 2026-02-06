import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

CONFIG = {
    "broker": {
        "name": "tinkoff",  # или binance, bybit и т.д.
        "api_key": os.getenv("BROKER_API_KEY"),
        "api_secret": os.getenv("BROKER_API_SECRET"),
        "testnet": True  # для тестирования
    },
    "strategy": {
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "threshold_percent": 0.001,
        "min_touch_count": 3
    },
    "risk": {
        "max_position_size": 1000,  # USD
        "max_daily_loss": 0.05,  # 5%
        "stop_loss": 0.02,  # 2%
        "take_profit": 0.04  # 4%
    },
    "telegram": {
        "token": os.getenv("TELEGRAM_TOKEN"),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID")
    },
    "ml": {
        "model_path": BASE_DIR / "models" / "lstm_model.h5",
        "retrain_interval": 86400  # секунды
    },
    "update_interval": 60,  # секунды
    "log_level": "INFO"
}