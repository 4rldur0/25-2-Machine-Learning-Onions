import pandas as pd
import numpy as np

class ModelError(Exception):
    pass

class Pipeline():
    def __init__(self, binary_classifier = None, multi_classifier = None):
        self._binary_classifier = binary_classifier
        self._multi_classifier = multi_classifier

    def predict(self, X):
        ...

class ClosedWorldPipeline(Pipeline):
    def predict(self, X):
        if not hasattr(self, "_multi_classifier"):
            raise ModelError("Please create/load a multi classifier model first!")

        y_pred = self._multi_classifier.predict(X)
        y_proba = self._multi_classifier.predict_proba(X)
        return y_pred, y_proba

class OpenWorldPipeline(Pipeline):
    def predict(self, X_binary, X_multi):
        if not hasattr(self, "_binary_classifier"):
            raise ModelError("Please create/load a binary classifier model first!")
        if not hasattr(self, "_multi_classifier"):
            raise ModelError("Please create/load a multi classifier model first!")

        # 입력 데이터의 길이만큼 결과를 담을 배열 생성 (기본값 -1: 비감시)
        n_samples = len(X_binary)
        final_preds = np.full(n_samples, -1)

        # 1. Binary Classification (X_binary 사용)
        binary_pred = self._binary_classifier.predict(X_binary)
        
        # 감시 대상으로 예측된 위치(Mask)
        monitored_mask = (binary_pred == 1)

        # 2. Multi-class Classification (X_multi 사용)
        if monitored_mask.sum() > 0:
            # 2단계 모델에는 'X_multi'에서 해당 샘플만 뽑아서 전달
            if isinstance(X_multi, pd.DataFrame):
                X_mon = X_multi.loc[monitored_mask]
            else:
                X_mon = X_multi[monitored_mask]
            
            monitored_pred = self._multi_classifier.predict(X_mon)
            
            # 결과 업데이트
            final_preds[monitored_mask] = monitored_pred

        return final_preds