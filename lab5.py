import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class ChatBot:
    prompt = '''Ты ассистент на веб сайте, в твои обязанности входит ответ на частозадаваемые вопросы (F.A.Q.)
                    Компания "Город Кранов"
                    Наш сайт занимается установкой кранов в г. Владивостоке и Приморском крае, ссылка на сайт - https://gorod-kranov.ru
                    Мы работает с 2008 года и у нас уже свыше 100 установленных башенных кранов по всему Приморью!
                    На все вопросы не связанные с компанией не отвечай! Отвечай сжато и только на поставленный вопрос!'''

    def __init__(self):
        torch.manual_seed(0)

        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "do_sample": False,
        }

    def generate(self, msg):
        prompt_with_msg = f"{self.prompt}\n{msg}"
        output = self.pipe(prompt_with_msg, **self.generation_args)
        return output[0]['generated_text']
