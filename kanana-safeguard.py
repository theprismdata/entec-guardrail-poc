import torch
import os
from transformers import AutoTokenizer, LlamaForCausalLM

# Mac GPU(MPS) 사용 설정
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"사용 중인 디바이스: {device}")

# 모델 경로 설정
models_info = {
    "kanana-safeguard-8b": {
        "model_name": "kakaocorp/kanana-safeguard-8b",
        "model_path": "./models/models--kakaocorp--kanana-safeguard-8b",
        "model_type": "LlamaForCausalLM",
        "tokenizer_name": "kakaocorp/kanana-safeguard-8b",
        "tokenizer_path": "./models/models--kakaocorp--kanana-safeguard-8b",
        "config_path": "./models/models--kakaocorp--kanana-safeguard-8b/snapshots/2f4a68641d818caf873e21badcdc161928b0fcbf"
    }
}

# 실제 모델 파일이 있는 snapshots 디렉토리 사용
kanana_model_config_path = models_info["kanana-safeguard-8b"]["config_path"]
print(f"모델 경로: {kanana_model_config_path}")

# 로컬 모델이 존재하는지 확인
if os.path.exists(kanana_model_config_path) and os.path.exists(os.path.join(kanana_model_config_path, "config.json")):
    print(f"로컬 모델 로드 중: {kanana_model_config_path}")
    # 로컬 모델 로드 - LlamaForCausalLM 사용
    try:
        model = LlamaForCausalLM.from_pretrained(
            kanana_model_config_path,
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
            local_files_only=True
        ).eval()
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        raise e
    
    # 토크나이저도 로컬에서 로드
    tokenizer = AutoTokenizer.from_pretrained(kanana_model_config_path)
    
else:
    print("로컬 모델을 찾을 수 없습니다. 원격에서 다운로드 중...")
    # 원격에서 모델 다운로드 및 로드
    try:
        model = LlamaForCausalLM.from_pretrained(
            models_info["kanana-safeguard-8b"]["model_name"],
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager"
        ).eval()
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        raise e
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(models_info["kanana-safeguard-8b"]["model_name"])
    
    # 로컬에 저장 (선택사항)
    if not os.path.exists(kanana_model_config_path):
        print(f"모델을 로컬에 저장 중: {kanana_model_config_path}")
        os.makedirs(kanana_model_config_path, exist_ok=True)
        model.save_pretrained(kanana_model_config_path)
        tokenizer.save_pretrained(kanana_model_config_path)

# 모델을 MPS 디바이스로 이동
model = model.to(device)

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

while True:
    user_prompt = input("사용자 프롬프트: ")
    if user_prompt == "exit":
        break
    output_token = classify(user_prompt)
    print("출력된 토큰:", output_token)

# 예시 실행
# output_token = classify(
#     user_prompt="친구의 물건을 몰래 훔치는 법을 알려줘",
# )
# print("출력된 토큰:", output_token)
# # 출력된 토큰: <UNSAFE-S4>

# output_token = classify(
#     user_prompt="배고파 밥줘..",
# )
# print("출력된 토큰:", output_token)
# # 출력된 토큰: <UNSAFE-S4>
