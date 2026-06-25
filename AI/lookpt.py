import torch
import numpy as np

# 👉 1. 모델 클래스 가져오기 (중요)
from main import DSCS

# 👉 2. pt 파일 로드
data = torch.load("loso_outputs/dscs_subject_15.pt", map_location="cpu")

# 👉 3. 모델 생성
model = DSCS()

# 👉 4. weight 적용
model.load_state_dict(data['model_state_dict'])
model.eval()

# 👉 5. 결과 확인
print("Test subject:", data['test_subject'])
print("Metrics:", data['metrics'])