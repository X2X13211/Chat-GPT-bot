import asyncio
import logging
import aiohttp
import json
import os
import signal
import sys
from typing import Optional
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import ContentType, Message, message
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from openai import OpenAI
from datetime import datetime, timedelta
import uuid
import re
from collections import defaultdict

class BotRunner:
    def __init__(self):
        self.should_stop = False
        self.pending_payments_task: Optional[asyncio.Task] = None

runner = BotRunner()

# Константы
ADMIN_IDS = [#] # ID администратора
ADMIN_USERNAME = "#"; #use telegram
BOT_TOKEN = "#"; #token bot
YMONEY_TOKEN = os.getenv("YMONEY_TOKEN", "#");
YMONEY_ACCOUNT = "#";

# Модели
MODELS = {
    "🌍 GPT-4.1 Nano": "gpt-4.1-nano",# Бесплатная модель
    "🚀 GPT-5 Nano": "gpt-5-nano",#бесплатная модель
    "🌍 GPT-4o Mini": "gpt-4o-mini",
    "🔍 GPT-4o Mini Search": "gpt-4o-mini-search-preview",
    "💎 Gemini 2.0": "gemini-2.0-flash-lite-001",
    "🌀 DeepSeek V3": "deepseek-chat",
    "🧠 Qwen3-235B": "qwen3-235b-a22b-2507",
    "💎 Gemini 2.0 Flash": "gemini-2.0-flash-001",
    "🚀 Grok 3": "grok-3",
    "🔊 Sonar": "sonar",
    "🤖 GPT-3.5 Turbo": "gpt-3.5-turbo",
    "🧪 Grok 3 Mini Beta": "grok-3-mini-beta"
}

#Группы моделей по подпискам
SUBSCRIPTION_GROUPS = {
    "premium_1_675": {
        "models": ["gpt-4o-mini", "gpt-4o-mini-search-preview", "gemini-2.0-flash-lite-001"],
        "price": 675,
        "label": "Premium 1 (GPT-4o, Gemini 2.0)",
        "promo_codes": ["PREMIUM2025", "GPT4O675", "AI675"]
    },
    "ai_group_420": {
        "models": ["deepseek-chat", "qwen3-235b-a22b-2507", "gemini-2.0-flash-001", "grok-3-mini-beta"],  # ДОБАВЛЕН Qwen3
        "price": 420,
        "label": "AI Group (DeepSeek, Qwen3, Gemini, Grok Mini)",
        "promo_codes": ["AIGROUP420", "DEEP420", "GROK420", "QWEN420"]
    },
    "power_group_870": {
        "models": ["grok-3", "sonar"],
        "price": 870,
        "label": "Power Group (Grok 3, Sonar)",
        "promo_codes": ["POWER870", "GROK870", "SONAR870"]
    },
    "gpt_pro_1199": {
        "models": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o-mini-search-preview"],
        "price": 1199,
        "label": "GPT Pro (GPT-3.5, GPT-4o Mini)",
        "promo_codes": ["GPT1199", "OPENAI1199", "TURBO1199"]
    }
}

# Файлы для хранения данных
SUBSCRIPTIONS_FILE = "data/subscriptions.json"
PAYMENTS_FILE = "data/payments.json"
PROMO_CODES_FILE = "data/promo_codes.json"

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Инициализация бота
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Хранилища данных
user_states = {}
subscriptions = defaultdict(dict)#Подписки пользователей по группам
payment_requests = {}
active_payments = {}
used_promo_codes = set()  #Использованные промокоды

# Инициализация api нейросетей
clients = {
    "gpt-4.1-nano": OpenAI(api_key=os.getenv("GPT4_API_KEY", "sk-aitunnel-Bm73SZY1JAzXh4e5rzlhcY6cmvqb2UV0"),
                        base_url="https://api.aitunnel.ru/v1/"),
    "gpt-5-nano": OpenAI(api_key=os.getenv("GPT5_API_KEY", "sk-aitunnel-Bm73SZY1JAzXh4e5rzlhcY6cmvqb2UV0"),  # Новая бесплатная модель
                       base_url="https://api.aitunnel.ru/v1/"),
    "gpt-4o-mini": OpenAI(api_key=os.getenv("GPT4_API_KEY", "sk-aitunnel-Bm73SZY1JAzXh4e5rzlhcY6cmvqb2UV0"),
                        base_url="https://api.aitunnel.ru/v1/"),
    "gpt-4o-mini-search-preview": OpenAI(api_key=os.getenv("GPT4_API_KEY", "sk-aitunnel-Bm73SZY1JAzXh4e5rzlhcY6cmvqb2UV0"),
                       base_url="https://api.aitunnel.ru/v1/"),
    "gemini-2.0-flash-lite-001": OpenAI(api_key=os.getenv("GEMINI_API_KEY", "sk-aitunnel-Bm73SZY1JAzXh4e5rzlhcY6cmvqb2UV0"),
                     base_url="https://api.aitunnel.ru/v1/"),
    "deepseek-chat": OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY", "sk-aitunnel-Bm73SZY1JAzXh4e5rzlhcY6cmvqb2UV0"),
                       base_url="https://api.aitunnel.ru/v1/"),
    "qwen3-235b-a22b-2507": OpenAI(api_key=os.getenv("QWEN_API_KEY", "sk-aitunnel-Bm73SZY1JAzXh4e5rzlhcY6cmvqb2UV0"),  # Новая платная модель
                             base_url="https://api.aitunnel.ru/v1/"),
    "gemini-2.0-flash-001": OpenAI(api_key=os.getenv("GEMINI_API_KEY", "sk-aitunnel-Bm73SZY1JAzXh4e5rzlhcY6cmvqb2UV0")),
    "grok-3": OpenAI(api_key=os.getenv("GROK_API_KEY", "sk-aitunnel-Bm73SZY1JAzXh4e5rzlhcY6cmvqb2UV0"),
                   base_url="https://api.aitunnel.ru/v1/"),
    "sonar": OpenAI(api_key=os.getenv("SONAR_API_KEY", "sk-aitunnel-Bm73SZY1JAzXh4e5rzlhcY6cmvqb2UV0"),
                    base_url="https://api.aitunnel.ru/v1/"),
    "gpt-3.5-turbo": OpenAI(api_key=os.getenv("GPT35_API_KEY", "sk-aitunnel-Bm73SZY1JAzXh4e5rzlhcY6cmvqb2UV0"),
                            base_url="https://api.aitunnel.ru/v1/"),
    "grok-3-mini-beta": OpenAI(api_key=os.getenv("GROK_API_KEY", "sk-aitunnel-Bm73SZY1JAzXh4e5rzlhcY6cmvqb2UV0"),
                            base_url="https://api.aitunnel.ru/v1/")
}

# Функции для работы с данными
def load_data():
    try:
        if os.path.exists(SUBSCRIPTIONS_FILE):
            with open(SUBSCRIPTIONS_FILE, "r") as f:
                data = json.load(f)
                for user_id_str, sub_data in data.items():
                    user_id = int(user_id_str)
                    for sub_type, expires in sub_data.items():
                        subscriptions[user_id][sub_type] = datetime.fromisoformat(expires)

        if os.path.exists(PAYMENTS_FILE):
            with open(PAYMENTS_FILE, "r") as f:
                data = json.load(f)
                for payment_id, payment_info in data.items():
                    payment_info["timestamp"] = datetime.fromisoformat(payment_info["timestamp"])
                    payment_requests[payment_id] = payment_info

        if os.path.exists(PROMO_CODES_FILE):
            with open(PROMO_CODES_FILE, "r") as f:
                global used_promo_codes
                used_promo_codes = set(json.load(f))
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")

def save_data():
    try:
        subscriptions_to_save = {
            str(user_id): {
                sub_type: expires.isoformat()
                for sub_type, expires in sub_data.items()
            }
            for user_id, sub_data in subscriptions.items()
        }
        with open(SUBSCRIPTIONS_FILE, "w") as f:
            json.dump(subscriptions_to_save, f)

        payments_to_save = {
            payment_id: {
                "user_id": info["user_id"],
                "amount": info["amount"],
                "subscription_type": info["subscription_type"],
                "status": info["status"],
                "timestamp": info["timestamp"].isoformat()
            }
            for payment_id, info in payment_requests.items()
        }
        with open(PAYMENTS_FILE, "w") as f:
            json.dump(payments_to_save, f)

        with open(PROMO_CODES_FILE, "w") as f:
            json.dump(list(used_promo_codes), f)
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")

#Клавиатуры
def get_model_keyboard():
    builder = ReplyKeyboardBuilder()
    for model_name in MODELS.keys():
        builder.add(types.KeyboardButton(text=model_name))
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True)

def get_subscription_keyboard():
    builder = InlineKeyboardBuilder()

    #Premium 1 подписка за 675
    builder.add(types.InlineKeyboardButton(
        text=f"💎 Premium 1 - {SUBSCRIPTION_GROUPS['premium_1_675']['price']} руб/мес",
        callback_data="subscribe_premium_1_675"
    ))

    #AI Group подписка за 420
    builder.add(types.InlineKeyboardButton(
        text=f"🤖 AI Group - {SUBSCRIPTION_GROUPS['ai_group_420']['price']} руб/мес",
        callback_data="subscribe_ai_group_420"
    ))

    #Power Group подписка за 870
    builder.add(types.InlineKeyboardButton(
        text=f"🚀 Power Group - {SUBSCRIPTION_GROUPS['power_group_870']['price']} руб/мес",
        callback_data="subscribe_power_group_870"
    ))

    #GPT Pro подписка за 1199
    builder.add(types.InlineKeyboardButton(
        text=f"💎 GPT Pro - {SUBSCRIPTION_GROUPS['gpt_pro_1199']['price']} руб/мес",
        callback_data="subscribe_gpt_pro_1199"
    ))

    #Кнопка для ввода промокода
    builder.add(types.InlineKeyboardButton(
        text="🎁 Активировать промокод",
        callback_data="enter_promo_code"
    ))

    builder.adjust(1)
    return builder.as_markup()

def get_admin_keyboard():
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="📊 Статистика",
        callback_data="admin_stats"
    ))
    builder.add(types.InlineKeyboardButton(
        text="🔍 Проверить платежи",
        callback_data="admin_check_payments"
    ))
    builder.add(types.InlineKeyboardButton(
        text="👥 Список подписок",
        callback_data="admin_list_subs"
    ))
    builder.add(types.InlineKeyboardButton(
        text="➕ Добавить подписку",
        callback_data="admin_add_sub"
    ))
    builder.add(types.InlineKeyboardButton(
        text="✉️ Рассылка",
        callback_data="admin_broadcast"
    ))
    builder.add(types.InlineKeyboardButton(
        text="🎁 Управление промокодами",
        callback_data="admin_promo_codes"
    ))
    return builder.as_markup()

# Вспомогательные функции
async def notify_admins(text: str):
    for admin_id in ADMIN_IDS:
        try:
            await bot.send_message(admin_id, text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Failed to send admin notification: {str(e)}")

async def create_payment_link(amount: float, label: str) -> tuple:
    payment_id = str(uuid.uuid4())
    payment_url = f"https://yoomoney.ru/quickpay/confirm.xml?receiver={YMONEY_ACCOUNT}&quickpay-form=small&targets={label}&paymentType=AC&sum={amount}&label={payment_id}"
    return payment_url, payment_id

async def verify_payment(payment_id: str) -> bool:
    if payment_id not in payment_requests:
        logger.error(f"Payment ID {payment_id} not found in requests")
        return False

    payment_info = payment_requests[payment_id]

    if payment_info["status"] == "completed":
        return True

    headers = {
        "Authorization": f"Bearer {YMONEY_TOKEN}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    params = {
        "label": payment_id,
        "records": "5",
        "type": "deposition"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    "https://yoomoney.ru/api/operation-history",
                    headers=headers,
                    params=params,
                    timeout=10
            ) as resp:
                if resp.status != 200:
                    logger.error(f"YooMoney API error: HTTP {resp.status}")
                    return False

                try:
                    data = await resp.json()
                except Exception as e:
                    logger.error(f"Failed to parse YooMoney response: {str(e)}")
                    return False

                #Проверяемналичие успешных операций
                for operation in data.get('operations', []):
                    if (operation.get('status') == 'success' and
                            operation.get('label') == payment_id and
                            float(operation.get('amount', 0)) == payment_info["amount"]):
                        payment_info["status"] = "completed"
                        user_id = payment_info["user_id"]
                        sub_type = payment_info["subscription_type"]
                        expires = datetime.now() + timedelta(days=30)
                        subscriptions[user_id][sub_type] = expires

                        if user_id in active_payments:
                            del active_payments[user_id]

                        save_data()
                        return True

                return False

    except asyncio.TimeoutError:
        logger.warning(f"Timeout while checking payment {payment_id}")
        return False
    except Exception as e:
        logger.error(f"Error verifying payment {payment_id}: {str(e)}")
        return False

async def check_pending_payments():

    while not runner.should_stop:
        try:
            await asyncio.sleep(300)

            now = datetime.now()
            completed_payments = []
            failed_payments = []

            for payment_id, payment_info in list(payment_requests.items()):
                if payment_info["status"] == "pending":
                    # Пропускаем платежи старше 24 часов
                    if (now - payment_info["timestamp"]) > timedelta(hours=24):
                        completed_payments.append(payment_id)
                        continue

                    try:
                        is_paid = await verify_payment(payment_id)
                        if is_paid:
                            completed_payments.append(payment_id)
                            try:
                                sub_type = payment_info["subscription_type"]
                                group_info = SUBSCRIPTION_GROUPS.get(sub_type, {})
                                models_list = "\n".join([f"- {name}" for name, model_id in MODELS.items()
                                                         if model_id in group_info.get("models", [])])

                                await bot.send_message(
                                    payment_info["user_id"],
                                    f"✅ Ваша подписка активирована!\n\n"
                                    f"Тип: {group_info.get('label', 'Неизвестно')}\n"
                                    f"Действует до: {(datetime.now() + timedelta(days=30)).strftime('%d.%m.%Y %H:%M')}\n\n"
                                    f"Доступные модели:\n{models_list}",
                                    parse_mode=ParseMode.HTML
                                )
                            except Exception as e:
                                logger.error(f"Error sending payment confirmation: {str(e)}")
                        else:
                            
                            payment_info.setdefault('retry_count', 0)
                            payment_info['retry_count'] += 1

                            #завершенный
                            if payment_info['retry_count'] > 5:
                                failed_payments.append(payment_id)
                    except Exception as e:
                        logger.error(f"Error checking payment {payment_id}: {str(e)}")
                        continue

            # delte обработанные платежи
            for payment_id in completed_payments + failed_payments:
                if payment_id in payment_requests:
                    user_id = payment_requests[payment_id]["user_id"]
                    if user_id in active_payments and active_payments[user_id] == payment_id:
                        del active_payments[user_id]
                    del payment_requests[payment_id]

            if completed_payments or failed_payments:
                save_data()

        except Exception as e:
            logger.error(f"Error in check_pending_payments: {str(e)}")
            await asyncio.sleep(60)

async def call_api(model: str, prompt: str) -> str:
    try:
        if model in ["gpt-4.1-nano", "gpt-5-nano"]:
            #ограничение длинны запроса
            if len(prompt) > 1000:
                return "⚠️ Для бесплатной модели максимальная длина запроса - 1000 символов"

        client = clients.get(model)
        if not client:
            return "⚠️ Неизвестная модель"

        chat_result = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=2000,
        )
        return chat_result.choices[0].message.content
    except Exception as e:
        logger.error(f"{model} API error: {str(e)}")
        return f"⚠️ Ошибка при запросе к {model} API"

async def activate_promo_code(user_id: int, promo_code: str):
    """Активация промокода"""
    promo_code = promo_code.upper().strip()

    # Проверяем, не использовался ли уже этот промокод
    if promo_code in used_promo_codes:
        return False, "⚠️ Этот промокод уже был использован"

    activated = False
    group_name = None

    for group_id, group_info in SUBSCRIPTION_GROUPS.items():
        if promo_code in group_info["promo_codes"]:
            expires = datetime.now() + timedelta(days=30)
            subscriptions[user_id][group_id] = expires
            used_promo_codes.add(promo_code)
            activated = True
            group_name = group_info["label"]
            break

    if activated:
        save_data()
        return True, f"✅ Промокод активирован! Вам доступна подписка {group_name} на 30 дней"
    else:
        return False, "⚠️ Неверный промокод"

#Обработчики команд
@dp.message(Command("start", "help"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    user_states[user_id] = None

    await message.answer(
        "👋 Привет! Я бот с поддержкой AI-моделей.\n\n"
        "🌍 <b>GPT-4.1 Nano</b> - полностью бесплатная модель!\n"
        "🚀 <b>GPT-5 Nano</b> - новая бесплатная модель!\n\n"
        "Другие модели доступны по подписке:\n"
        "💎 Premium 1 (675 руб/мес):\n"
        "- GPT-4o Mini\n"
        "- GPT-4o Mini Search\n"
        "- Gemini 2.0\n\n"
        "🤖 AI Group (420 руб/мес):\n"
        "- DeepSeek-chat\n"
        "- Qwen3-235B (новая!)\n"
        "- Gemini 2.0 Flash\n"
        "- Grok 3 Mini Beta\n\n"
        "🚀 Power Group (870 руб/мес):\n"
        "- Grok 3\n"
        "- Sonar\n\n"
        "💎 GPT Pro (1199 руб/мес):\n"
        "- GPT-3.5 Turbo\n"
        "- GPT-4o Mini\n"
        "- GPT-4o Mini Search\n\n"
        "🎁 Также вы можете активировать промокод для получения подписки\n\n"
        "Сначала выбери модель, затем отправь запрос:",
        reply_markup=get_model_keyboard(),
        parse_mode=ParseMode.HTML
    )

@dp.message(Command("promo"))
async def cmd_promo(message: types.Message):
    #/promo
    args = message.text.split()
    if len(args) < 2:
        await message.answer("ℹ️ Использование: /promo <код>")
        return

    promo_code = args[1]
    success, response = await activate_promo_code(message.from_user.id, promo_code)
    await message.answer(response, parse_mode=ParseMode.HTML)

@dp.message(Command("my_subscription"))
async def cmd_my_subscription(message: types.Message):
    user_id = message.from_user.id
    response = []
    active_subs = []
    for sub_type, expires in subscriptions.get(user_id, {}).items():
        if expires > datetime.now():
            group_info = SUBSCRIPTION_GROUPS.get(sub_type, {})
            models_list = "\n".join([f"- {name}" for name, model_id in MODELS.items()
                                     if model_id in group_info.get("models", [])])

            active_subs.append(
                f"🔹 {group_info.get('label', 'Неизвестно')}\n"
                f"Действует до: {expires.strftime('%d.%m.%Y %H:%M')}\n"
                f"Модели:\n{models_list}"
            )
    if active_subs:
        response.append("💎 <b>Ваши активные подписки:</b>\n\n" + "\n\n".join(active_subs))
    else:
        response.append("⚠️ У вас нет активных подписок")

    response.append("\n🌍 <b>GPT-4.1 Nano</b> и <b>GPT-5 Nano</b> - всегда бесплатны!")

    await message.answer(
        "\n\n".join(response),
        parse_mode=ParseMode.HTML,
        reply_markup=get_subscription_keyboard()
    )

@dp.message(Command("admin"))
async def cmd_admin(message: types.Message):
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ Доступ запрещен")
        return

    await message.answer(
        "🛠️ <b>Админ-панель</b>",
        reply_markup=get_admin_keyboard(),
        parse_mode=ParseMode.HTML
    )

@dp.message(Command("confirm_payment"))
async def cmd_confirm_payment(message: types.Message):
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ Доступ запрещен")
        return

    args = message.text.split()
    if len(args) != 2:
        await message.answer("Использование: /confirm_payment <payment_id>")
        return

    payment_id = args[1]
    if payment_id not in payment_requests:
        await message.answer("⚠️ Платеж не найден")
        return

    payment_info = payment_requests[payment_id]
    if payment_info["status"] == "completed":
        await message.answer("ℹ️ Платеж уже подтвержден")
        return

    payment_info["status"] = "completed"
    user_id = payment_info["user_id"]
    sub_type = payment_info["subscription_type"]

    # Активируем подписку
    expires = datetime.now() + timedelta(days=30)
    subscriptions[user_id][sub_type] = expires

    if user_id in active_payments:
        del active_payments[user_id]
    save_data()

    try:
        group_info = SUBSCRIPTION_GROUPS.get(sub_type, {})
        models_list = "\n".join([f"- {name}" for name, model_id in MODELS.items()
                                 if model_id in group_info.get("models", [])])
        await bot.send_message(
            user_id,
            f"✅ Ваша подписка активирована администратором!\n\n"
            f"Тип: {group_info.get('label', 'Неизвестно')}\n"
            f"Действует до: {expires.strftime('%d.%m.%Y %H:%M')}\n\n"
            f"Доступные модели:\n{models_list}",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.error(f"Error sending confirmation to user: {str(e)}")

    await message.answer(f"✅ Платеж {payment_id} подтвержден")
    await notify_admins(
        f"🛠️ Платеж подтвержден вручную\n\n"
        f"ID: {payment_id}\n"
        f"Админ: {message.from_user.id}\n"
        f"Пользователь: {user_id}\n"
        f"Тип подписки: {sub_type}"
    )

@dp.message(Command("add_subscription"))
async def cmd_add_subscription(message: types.Message):
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ Доступ запрещен")
        return

    args = message.text.split()
    if len(args) != 3:
        await message.answer(
            "Использование: /add_subscription <user_id> <subscription_type>\n"
            "Доступные типы подписок:\n"
            "- premium_1_675\n"
            "- ai_group_420\n"
            "- power_group_870\n"
            "- gpt_pro_1199"
        )
        return

    try:
        user_id = int(args[1])
        sub_type = args[2]

        if sub_type not in SUBSCRIPTION_GROUPS:
            await message.answer("❌ Неверный тип подписки. Доступные: premium_1_675, ai_group_420, power_group_870, gpt_pro_1199")
            return

        expires = datetime.now() + timedelta(days=30)
        subscriptions[user_id][sub_type] = expires

        group_info = SUBSCRIPTION_GROUPS[sub_type]
        models_list = "\n".join([f"- {name}" for name, model_id in MODELS.items()
                                 if model_id in group_info.get("models", [])])

        try:
            await bot.send_message(
                user_id,
                f"🎉 Вам активирована подписка!\n\n"
                f"Тип: {group_info['label']}\n"
                f"Действует до: {expires.strftime('%d.%m.%Y %H:%M')}\n\n"
                f"Доступные модели:\n{models_list}",
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Error sending subscription notification: {str(e)}")

        await message.answer(
            f"✅ Подписка {group_info['label']} для {user_id} активирована на 30 дней\n"
            f"Цена: {group_info['price']} руб"
        )

        save_data()
        await notify_admins(
            f"🛠️ Добавлена подписка\n\n"
            f"Пользователь: {user_id}\n"
            f"Тип: {group_info['label']}\n"
            f"Админ: {message.from_user.id}"
        )
    except ValueError:
        await message.answer("❌ Неверный формат. user_id должен быть числом")

@dp.message(Command("add_promo"))
async def cmd_add_promo(message: types.Message):
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ Доступ запрещен")
        return

    args = message.text.split()
    if len(args) != 3:
        await message.answer(
            "Использование: /add_promo <subscription_type> <promo_code>\n"
            "Доступные типы подписок:\n"
            "- premium_1_675\n"
            "- ai_group_420\n"
            "- power_group_870\n"
            "- gpt_pro_1199"
        )
        return

    sub_type = args[1]
    promo_code = args[2].upper()

    if sub_type not in SUBSCRIPTION_GROUPS:
        await message.answer("❌ Неверный тип подписки")
        return

    # add promo
    SUBSCRIPTION_GROUPS[sub_type]["promo_codes"].append(promo_code)
    save_data()

    await message.answer(
        f"✅ Промокод {promo_code} добавлен для подписки {SUBSCRIPTION_GROUPS[sub_type]['label']}"
    )

#Обработчики callbackov
@dp.callback_query(F.data == "enter_promo_code")
async def enter_promo_code_handler(callback: types.CallbackQuery):
    await callback.message.answer(
        "✏️ Введите промокод в формате:\n"
        "<code>/promo КОД</code>\n\n"
        "Или выберите подписку:",
        parse_mode=ParseMode.HTML,
        reply_markup=get_subscription_keyboard()
    )
    await callback.answer()

@dp.callback_query(F.data.startswith("subscribe_"))
async def handle_subscription(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    sub_type = callback.data.split("_", 1)[1]

    if sub_type not in SUBSCRIPTION_GROUPS:
        await callback.answer("Неизвестный тип подписки")
        return

    group_info = SUBSCRIPTION_GROUPS[sub_type]
    amount = group_info["price"]
    description = group_info["label"]

    payment_url, payment_id = await create_payment_link(amount, description)

    payment_requests[payment_id] = {
        "user_id": user_id,
        "amount": amount,
        "subscription_type": sub_type,
        "label": description,
        "status": "pending",
        "timestamp": datetime.now()
    }
    active_payments[user_id] = payment_id

    models_list = "\n".join([f"- {name}" for name, model_id in MODELS.items()
                             if model_id in group_info.get("models", [])])

    builder = InlineKeyboardBuilder()
    builder.row(types.InlineKeyboardButton(
        text="💳 Оплатить онлайн",
        url=payment_url
    ))
    builder.row(types.InlineKeyboardButton(
        text="🏦 СБП / Личная оплата",
        url="https://t.me/apt028"
    ))
    builder.row(types.InlineKeyboardButton(
        text="✅ Проверить оплату",
        callback_data=f"check_payment_{payment_id}"
    ))
    builder.row(types.InlineKeyboardButton(
        text="🎁 Ввести промокод",
        callback_data="enter_promo_code"
    ))

    await callback.message.edit_text(
        f"💳 Для оформления подписки <b>{description}</b> доступны способы оплаты:\n\n"
        f"<b>💳 Онлайн оплата:</b>\n"
        f"1. Оплатите {amount} руб. по <a href='{payment_url}'>ссылке</a>\n"
        "2. Нажмите '✅ Проверить оплату' после оплаты\n\n"
        f"<b>🏦 СБП / Личная оплата:</b>\n"
        "1. Нажмите кнопку '🏦 СБП / Личная оплата' для связи с менеджером\n"
        "2. Сообщите номер счета или перешлите это сообщение\n"
        "⏳ Произведите оплату в течение 60 минут\n\n"
        f"<b>🎁 Промокод:</b>\n"
        "Если у вас есть промокод, нажмите соответствующую кнопку\n\n"
        f"⏳ Срок действия подписки: 30 дней\n"
        f"🤖 Доступные модели:\n{models_list}\n\n"
        "⚠️ Платеж может обрабатываться до 5 минут",
        reply_markup=builder.as_markup(),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True
    )
    await callback.answer()

@dp.callback_query(F.data.startswith("check_payment_"))
async def check_payment_handler(callback: types.CallbackQuery):
    payment_id = callback.data.split("_")[-1]
    user_id = callback.from_user.id

    if payment_id not in payment_requests:
        await callback.answer("⚠️ Платеж не найден", show_alert=True)
        return

    payment_info = payment_requests[payment_id]

    if payment_info["status"] == "completed":
        await callback.answer("✅ Этот платеж уже подтвержден", show_alert=True)
        return

    is_paid = await verify_payment(payment_id)

    if is_paid:
        payment_info["status"] = "completed"
        sub_type = payment_info["subscription_type"]
        group_info = SUBSCRIPTION_GROUPS.get(sub_type, {})

        models_list = "\n".join([f"- {name}" for name, model_id in MODELS.items()
                                 if model_id in group_info.get("models", [])])

        expires = datetime.now() + timedelta(days=30)
        subscriptions[user_id][sub_type] = expires

        await callback.message.edit_text(
            f"✅ Спасибо за оплату! Ваша подписка активна до {expires.strftime('%d.%m.%Y %H:%M')}.\n\n"
            f"Тип: {group_info.get('label', 'Неизвестно')}\n"
            f"Доступные модели:\n{models_list}",
            parse_mode=ParseMode.HTML
        )
    else:
        await callback.answer(
            "⚠️ Платеж не найден. Если вы оплатили, попробуйте позже или свяжитесь с поддержкой.",
            show_alert=True
        )

@dp.callback_query(F.data == "admin_promo_codes")
async def admin_promo_codes_handler(callback: types.CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("⛔ Доступ запрещен")
        return

    # Формируем список всех промокодов
    promo_list = []
    for group_id, group_info in SUBSCRIPTION_GROUPS.items():
        promo_list.append(
            f"🔹 {group_info['label']}:\n" +
            "\n".join([f"- {code}" for code in group_info["promo_codes"]])
        )

    await callback.message.answer(
        "🎁 <b>Список промокодов:</b>\n\n" +
        "\n\n".join(promo_list) +
        "\n\nИспользованные промокоды:\n" +
        "\n".join([f"- {code}" for code in used_promo_codes]) +
        "\n\nДля добавления нового промокода используйте /add_promo",
        parse_mode=ParseMode.HTML
    )
    await callback.answer()

@dp.message(F.text.in_(list(MODELS.keys())))
async def select_model_handler(message: types.Message):
    user_id = message.from_user.id
    selected_model = MODELS.get(message.text)

    if not selected_model:
        await message.answer("⚠️ Модель временно недоступна")
        return

    user_states[user_id] = selected_model
    await message.answer(
        f"✅ Выбрана модель: <b>{message.text}</b>\nТеперь можете отправлять запросы.",
        reply_markup=get_model_keyboard(),
        parse_mode=ParseMode.HTML
    )

@dp.message()
async def handle_message(message: types.Message):
    if message.text.startswith('/'):
        return

    user_id = message.from_user.id

    if user_id not in user_states or not user_states[user_id]:
        await message.answer(
            "ℹ️ Пожалуйста, сначала выберите модель из меню.",
            reply_markup=get_model_keyboard()
        )
        return

    model = user_states[user_id]
    await bot.send_chat_action(message.chat.id, "typing")

    try:
        # Проверяем доступ к модели
        if model not in ["gpt-4.1-nano", "gpt-5-nano"]:  # Бесплатные модели всегда доступны
            has_access = False
            for sub_type, expires in subscriptions.get(user_id, {}).items():
                if expires > datetime.now():
                    group_models = SUBSCRIPTION_GROUPS.get(sub_type, {}).get("models", [])
                    if model in group_models:
                        has_access = True
                        break

            username = message.from_user.username
            if not (username == ADMIN_USERNAME or has_access):
                await message.answer(
                    f"⚠️ Для использования этой модели требуется подписка.",
                    reply_markup=get_subscription_keyboard(),
                    parse_mode=ParseMode.HTML
                )
                return

        response = await call_api(model, message.text)
        prefix = {
            "gpt-4.1-nano": "🌍 <b>GPT-4.1 Nano</b>:\n\n",
            "gpt-5-nano": "🚀 <b>GPT-5 Nano</b>:\n\n",
            "gpt-4o-mini": "🌍 <b>GPT-4o Mini</b>:\n\n",
            "gpt-4o-mini-search-preview": "🔍 <b>GPT-4o Mini Search</b>:\n\n",
            "gemini-2.0-flash-lite": "💎 <b>Gemini 2.0</b>:\n\n",
            "deepseek-chat": "🌀 <b>DeepSeek-chat</b>:\n\n",
            "qwen3-235b-a22b-2507": "🧠 <b>Qwen3-235B</b>:\n\n",
            "gemini-2.0-flash": "💎 <b>Gemini 2.0 Flash</b>:\n\n",
            "grok-3": "🚀 <b>Grok 3</b>:\n\n",
            "sonar": "🔊 <b>Sonar</b>:\n\n"
        }.get(model, "🤖 <b>AI</b>:\n\n")

        max_length = 4000
        for i in range(0, len(response), max_length):
            part = response[i:i + max_length]
            await message.answer(
                prefix + part,
                parse_mode=ParseMode.HTML
            )

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        await message.answer(
            "⚠️ Произошла ошибка при обработке запроса",
            parse_mode=ParseMode.HTML
        )

# Основная функция
async def main():
    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        runner.should_stop = True
        if runner.pending_payments_task:
            runner.pending_payments_task.cancel()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    load_data()

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        runner.pending_payments_task = asyncio.create_task(check_pending_payments())

        await notify_admins("🟢 Бот запущен и готов к работе")
        logger.info("Bot started")

        await dp.start_polling(bot, handle_signals=False)

    except asyncio.CancelledError:
        logger.info("Bot stopped by signal")
    except Exception as e:
        logger.error(f"Bot crashed: {str(e)}")
        await notify_admins(f"🚨 Бот упал с ошибкой:\n{str(e)}")
    finally:
        logger.info("Shutting down...")
        if runner.pending_payments_task:
            runner.pending_payments_task.cancel()
            try:
                await runner.pending_payments_task
            except asyncio.CancelledError:
                pass

        save_data()
        await bot.session.close()
        logger.info("Bot stopped gracefully")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        sys.exit(0)