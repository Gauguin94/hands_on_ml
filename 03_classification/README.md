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
> 매번 다운로드 받기에는 시간이 너무 아깝기 때문에 OS에 맞춰 지정한 디렉토리에  
> 파일을 저장하도록 한다.  