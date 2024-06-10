from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Initialize the model and pipeline
torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "do_sample": False,
}


# Define the request model
class GenerateResponse(BaseModel):
    generated_text: str


app = FastAPI()


@app.get("/generate", response_model=GenerateResponse)
async def generate_text(
        user_message: str = Query(..., description="User message for the assistant")
):
    try:
        messages = [
            {"role": "user", "content": '''Ты ассистент на веб сайте, в твои обязанности входит ответ на частозадаваемые вопросы (F.A.Q.)
                        Компания "Город Кранов"
                        Наш сайт занимается установкой кранов в г. Владивостоке и Приморском крае, ссылка на сайт - https://gorod-kranov.ru
                        Мы работает с 2008 года (на данный момент 2024 год) и у нас уже свыше 100 установленных башенных кранов по всему Приморью!
                        
                        На все вопросы не связанные с компанией не отвечай! Отвечай сжато и только на поставленный вопрос!
                        Не используй больше одного символа переноса строки за раз!'''},
            {"role": "user", "content": user_message},
        ]
        output = pipe(messages, **generation_args)
        generated_text = output[0]['generated_text']
        return GenerateResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the server with: uvicorn server:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
