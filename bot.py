import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command

class ChatBot:
    prompt = '''Ты ассистент на веб сайте, в твои обязанности входит ответ на частозадаваемые вопросы (F.A.Q.)
                    Компания "Город Кранов"
                    Наш сайт занимается установкой кранов в г. Владивостоке и Приморском крае, ссылка на сайт - https://gorod-kranov.ru
                    Мы работает с 2008 года и у нас уже свыше 100 установленных башенных кранов по всему Приморью!
                    На все вопросы не связанные с компанией не отвечай! Отвечай сжато и только на поставленный вопрос!'''

    def __init__(self):
        torch.manual_seed(0)

        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "do_sample": False,
        }

    async def generate(self, msg):
        prompt_with_msg = f"{self.prompt}\n{msg}"
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, self.pipe, prompt_with_msg, **self.generation_args)
        return output[0]['generated_text']

logging.basicConfig(level=logging.INFO)
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
bot = Bot(token=bot_token)
dp = Dispatcher()
chatbot = ChatBot()

@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    await message.reply("Привет! Я бот компании 'Город Кранов'. Задайте мне свой вопрос.")

@dp.message()
async def echo(message: types.Message):
    user_message = message.text
    await bot.send_chat_action(message.chat.id, action='typing')

    try:
        generated_text = await chatbot.generate(user_message)
        await message.reply(generated_text)
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        await message.reply("Извините, произошла ошибка при обработке вашего запроса.")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
