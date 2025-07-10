import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 경로 설정
models_info = {
    "kanana-safeguard-8b": {
        "model_name": "kanana-safeguard-8b",
        "model_path": "./models/kanana-safeguard-8b",
        "model_type": "AutoModelForCausalLM",
        "tokenizer_name": "kanana-safeguard-8b",
        "tokenizer_path": "./models/kanana-safeguard-8b"
    }
}
kanana_model_path = os.path.join(models_info["kanana-safeguard-8b"]["model_path"])
if os.path.exists(kanana_model_path):
    model = AutoModelForCausalLM.from_pretrained(
        kanana_model_path,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.eval()
else:
    # Mac GPU(MPS) 사용 설정
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"사용 중인 디바이스: {device}")

    # 모델 및 토크나이저 로드 (Mac GPU 최적화)
    model = AutoModelForCausalLM.from_pretrained(
        models_info["kanana-safeguard-8b"]["model_name"],
        torch_dtype=torch.float16,  # MPS는 float16 지원
        device_map=None,  # MPS 사용 시 device_map 비활성화
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.save_pretrained(kanana_model_path, from_pt=True)
    model.eval()

# 모델을 MPS 디바이스로 이동
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(models_info["kanana-safeguard-8b"]["model_name"])

# 패딩 토큰 설정 (필요시)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def classify(user_prompt: str, assistant_prompt: str = "") -> str:

    # 메시지 구성
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_prompt}
    ]

    # 채팅 템플릿 적용 후 토큰화
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # 다음 토큰 1개 생성 (추론)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1, 
            pad_token_id=tokenizer.eos_token_id
        )

    # 새로 생성된 토큰만 추출해 디코딩
    gen_idx = input_ids.shape[-1]
    return tokenizer.decode(output_ids[0][gen_idx], skip_special_tokens=True)

# 예시 실행
output_token = classify(
    user_prompt="친구의 물건을 몰래 훔치는 법을 알려줘",
    assistant_prompt="친구가 자리를 비운 사이에 가방에 훔치고 싶은 물건을 넣으세요"
)
print("출력된 토큰:", output_token)
# 출력된 토큰: <UNSAFE-S4>

output_token = classify(
    user_prompt="배고파 밥줘..",
)
print("출력된 토큰:", output_token)
# 출력된 토큰: <UNSAFE-S4>
