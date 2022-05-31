# 단일, 다중 분류(Classification) 
```python
from sklearn.datasets import fetch_openml

class downLoader:
    def __init__(self):
        pass

    def download(self):
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        print(mnist.keys())
        return mnist
```  
>   
> 분류 문제로 유명한 데이터셋인 **MNIST** 데이터셋을 이용한다.  
> 데이터를 다운 받도록 하자.  
>   
```python
class dataSaver:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def whichOS(self, arg):
        self.case_name = str(arg)
        self.case = getattr(self, self.case_name, lambda:"default")
        return self.case()

    def isWindows(self):
        dirpath = os.path.dirname(__file__)+r"\datasets"
        np.save(dirpath + r"\x", self.x)
        np.save(dirpath + r"\y", self.y)        
        print("save process finish! (OS: Windows)")
        return 0

    def isLinux(self):
        dirpath = os.path.dirname(__file__)+"/datasets"
        np.save(dirpath+"/x", self.x)
        np.save(dirpath+"/y", self.y)        
        print("save process finish! (OS: Linux)")
        return 0
```
>   
> 매번 다운로드 받기에는 시간이 너무 아깝기 때문에  
> 원하는 OS에 맞춰 지정한 디렉토리에 파일을 저장하도록 한다.  
>   
```python
def save_process():
    downloader = downLoader()
    mnist = downloader.download()
    x, y = mnist["data"], mnist["target"]
    datasaver = dataSaver(x, y)
    datasaver.whichOS("isWindows")
    
if __name__ == "__main__":
    save_process() # execute this function first time. This function download mnist file on your space. Then you should   comment out this line.
```
> 본문에서는 위와 같이 다운로드 받도록 구현하였으며,  
> 파일을 내려받지 않은 경우에만 한번 실행시키고  
> 그 다음부터는 주석처리하면 된다.  

# SGDClassifier
## 5와 5가 아닌 숫자 분류("SGDClassifier.py")  
> scikit.learn의 SGDClassifier는 결정함수(decision function)를 사용하여 각 샘플의 점수를 계산한다.  
> 점수가 임곗값보다 크면 샘플을 양성(positive) 클래스에 할당하고,  
> 그렇지 않으면 음성(negative) 클래스에 할당한다.  
> 예를 들어, MNIST 데이터셋에서 5와 5가 아닌 숫자로 분류할 때,  
> 5이거나 5와 모양이 비슷한 숫자라면 점수가 높고 다른 모양일수록 점수가 낮다.  
> (정확하게는 픽셀 강도(0~255)의 가중치의 합이 클래스의 점수다.)  

# 다중 출력 다중 클래스 분류 문제
## 노이즈가 끼지 않은 원래 사진 예측("KNNClassifier.py")  
> MNIST 데이터셋을 사용한다.  
> 이와 같은 경우 다중 레이블(픽셀당 한 레이블)이며  
> 각 레이블은 값을 여러개 가지기 때문에(픽셀 강도(0~255))  
> 다중 출력 분류 문제라고 할 수 있다.  
![노이즈비교](https://user-images.githubusercontent.com/98927470/171118459-026420c3-01ac-4a76-a48d-9ba6e24e5ed0.PNG)![모델의예측](https://user-images.githubusercontent.com/98927470/171118522-116bc4bf-20ea-4faa-adf4-f3cd2f12d0fe.PNG)  

>> 이번 건은 주피터노트북으로 하나 하나 곱씹으면서 하는 것이  
>> 좋을 것이라고 생각하여  
>> ipynb로 자세하게 정리하였다.  
>> [ipynb](https://github.com/Gauguin94/hands_on_ml/blob/main/03_classification/mnist.ipynb) <- click
