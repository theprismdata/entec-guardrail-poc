import torch
import os
import time
from transformers import AutoTokenizer, LlamaForCausalLM

# Mac GPU(MPS) 사용 설정
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"사용 중인 디바이스: {device}")

# 성능 최적화 설정
torch.backends.cudnn.benchmark = True  # CUDNN 벤치마크 모드
if device == "mps":
    torch.mps.set_per_process_memory_fraction(0.8)  # MPS 메모리 제한

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

print("🚀 모델 로딩 중... (최초 실행 시 시간이 걸릴 수 있습니다)")
start_time = time.time()

# 로컬 모델이 존재하는지 확인
if os.path.exists(kanana_model_config_path) and os.path.exists(os.path.join(kanana_model_config_path, "config.json")):
    print(f"로컬 모델 로드 중: {kanana_model_config_path}")
    # 로컬 모델 로드 - LlamaForCausalLM 사용 (성능 최적화)
    try:
        model = LlamaForCausalLM.from_pretrained(
            kanana_model_config_path,
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # Flash Attention 2 사용 (더 빠름)
            local_files_only=True,
            use_cache=True  # KV 캐시 활성화
        ).eval()
    except Exception as e:
        print(f"Flash Attention 2 로드 실패, 기본 모드로 전환: {e}")
        # Flash Attention이 지원되지 않는 경우 fallback
        model = LlamaForCausalLM.from_pretrained(
            kanana_model_config_path,
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
            local_files_only=True,
            use_cache=True
        ).eval()
    
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
            attn_implementation="flash_attention_2",
            use_cache=True
        ).eval()
    except Exception as e:
        print(f"Flash Attention 2 로드 실패, 기본 모드로 전환: {e}")
        model = LlamaForCausalLM.from_pretrained(
            models_info["kanana-safeguard-8b"]["model_name"],
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
            use_cache=True
        ).eval()
    
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

# 토크나이저 성능 최적화
tokenizer.padding_side = "left"  # 배치 처리 최적화

load_time = time.time() - start_time
print(f"✅ 모델 로딩 완료! (소요시간: {load_time:.2f}초)")

def classify(user_prompt: str, assistant_prompt: str = "") -> str:
    """
    사용자 프롬프트와 어시스턴트 응답을 분류하는 함수 (성능 최적화)
    
    Args:
        user_prompt: 사용자 질문/요청
        assistant_prompt: 어시스턴트 응답 (선택사항)
    
    Returns:
        분류 결과 토큰 (예: <UNSAFE-S4>, <SAFE> 등)
    """
    # 메시지 구성
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_prompt}
    ]

    # 채팅 템플릿 적용 후 토큰화 (성능 최적화)
    input_ids = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        return_tensors="pt",
        padding=False,  # 패딩 비활성화 (단일 추론 시 불필요)
        truncation=True,  # 긴 입력 자르기
        max_length=512  # 최대 길이 제한으로 속도 향상
    ).to(device)
    
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # 다음 토큰 1개 생성 (추론) - 성능 최적화
    with torch.no_grad():
        with torch.autocast(device_type=device.split(':')[0] if ':' in device else device, dtype=torch.float16):
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                min_new_tokens=1,  # 최소 토큰 수 보장
                do_sample=False,  # 그리디 디코딩 (더 빠름)
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,  # KV 캐시 사용
                temperature=None,  # 온도 비활성화 (그리디)
                top_p=None,  # top-p 비활성화
                top_k=None,  # top-k 비활성화
                repetition_penalty=None,  # 반복 페널티 비활성화
                num_beams=1,  # 빔 서치 비활성화
                early_stopping=True  # 조기 종료
            )

    # 새로 생성된 토큰만 추출해 디코딩
    gen_idx = input_ids.shape[-1]
    return tokenizer.decode(output_ids[0][gen_idx], skip_special_tokens=True)

def get_safety_description(token: str) -> str:
    """안전성 토큰에 대한 설명을 반환"""
    descriptions = {
        "<SAFE>": "안전한 내용입니다.",
        "<UNSAFE-S1>": "불법 활동 관련 위험 내용입니다.",
        "<UNSAFE-S2>": "혐오 발언 관련 위험 내용입니다.",
        "<UNSAFE-S3>": "성적 내용 관련 위험 내용입니다.",
        "<UNSAFE-S4>": "자해/폭력 관련 위험 내용입니다.",
        "<UNSAFE-S5>": "개인정보 침해 관련 위험 내용입니다.",
        "<UNSAFE-S6>": "기타 위험 내용입니다."
    }
    return descriptions.get(token, "알 수 없는 분류 결과입니다.")

# 웜업 실행 (첫 번째 추론 속도 향상)
print("🔥 모델 웜업 중...")
warmup_start = time.time()
_ = classify("테스트", "")
warmup_time = time.time() - warmup_start
print(f"✅ 웜업 완료! (소요시간: {warmup_time:.2f}초)")

def main():
    """메인 대화 루프"""
    print("\n" + "="*60)
    print("🛡️  Kanana Safeguard 대화형 안전성 검사 시스템 (고속 버전)")
    print("="*60)
    print("사용자 질문과 어시스턴트 응답의 안전성을 검사합니다.")
    print("종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.")
    print("="*60)
    
    while True:
        try:
            print("\n" + "-"*40)
            # 사용자 질문 입력
            user_input = input("👤 사용자 질문: ").strip()
            
            # 종료 조건 확인
            if user_input.lower() in ['quit', 'exit', 'q', '종료']:
                print("👋 안전성 검사 시스템을 종료합니다.")
                break
            
            if not user_input:
                print("⚠️  질문을 입력해주세요.")
                continue
            
            # 어시스턴트 응답 입력 (선택사항)
            assistant_input = input("🤖 어시스턴트 응답 (선택사항, Enter로 건너뛰기): ").strip()
            
            # 안전성 분류 수행 (시간 측정)
            print("\n🔍 안전성 검사 중...")
            inference_start = time.time()
            result_token = classify(user_input, assistant_input)
            inference_time = time.time() - inference_start
            description = get_safety_description(result_token)
            
            # 결과 출력 (성능 정보 포함)
            print(f"\n📊 검사 결과: (추론 시간: {inference_time:.3f}초)")
            print(f"   분류 토큰: {result_token}")
            print(f"   설명: {description}")
            
            # 안전성 레벨에 따른 추가 정보
            if result_token == "<SAFE>":
                print("   ✅ 안전한 대화입니다.")
            else:
                print("   ⚠️  주의가 필요한 내용이 감지되었습니다.")
                
        except KeyboardInterrupt:
            print("\n\n👋 안전성 검사 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")

if __name__ == "__main__":
    main()
