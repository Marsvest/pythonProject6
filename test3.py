import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from lab5 import ChatBot

logging.basicConfig(level=logging.INFO)
bot = Bot(token="6953241738:AAEJYNhL18l55rl1P7rZwKZg0gPZ9xxuIpw")
dp = Dispatcher()
chatbot = ChatBot()


@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    await message.reply("Привет! Я бот компании 'Город Кранов'. Задайте мне свой вопрос.")


@dp.message()
async def echo(message: types.Message):
    user_message = message.text
    await bot.send_chat_action(message.chat.id, action='typing')

    generated_text = chatbot.generate(user_message)
    await message.reply(generated_text)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
