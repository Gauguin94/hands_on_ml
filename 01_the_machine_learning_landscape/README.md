# 단순 선형 회귀(Simple Linear Regression) 예제
# 실행
> 1. 국가별 국내총생산(GDP)과 삶의 만족도를 엮어 데이터를 만든다.
```python
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
```
> 2. sk.learn을 통해 linear_model을 만들고 앞서 만든 데이터를 입력으로 하여 모델을 학습시킨다.
```python
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)
```
> 3. 학습이 종료된 모델에 새로운 데이터('키프로스'의 국내총생산)를 입력으로 하여 예측(출력)값을 확인한다.(추론 과정)  
```python
X_new = [[22587]]
print('Republic of Cyprus's Life satisfaction: {}'.format(model.predict(X_new)))
```
> 제작한 코드를 Windows나 Linux 체제에서 실행시킬 것을 고려하여, 'dataloader'라는 클래스를 구현.  
> dataloader 내 존재하는 whichOS는 OS에 맞춰 데이터를 불러오는 역할을 한다.  
> data에는 "oecd_bli_2015.csv"와 "gdp_per_capita.csv"의 데이터가 존재한다.  
> "linear_regression.py"를 실행한다.
> ### **example**
```python
  data = dataloader.whichOS("isWindows")
  oecd_bli = data[0]
  gdp_per_capita = data[1]
```
```python
  data = dataloader.whichOS("isLinux")
  oecd_bli = data[0]
  gdp_per_capita = data[1]
```

# 결과
![Result](https://user-images.githubusercontent.com/98927470/170038411-0431889f-f47c-4048-b50a-678c11c57953.PNG)
# 그래서 회귀가 뭔데?
> 회귀분석이란, 둘 이상의 변수들의 인과관계를 파악함으로써 어떤 특정한 변수(종속변수)의 값을 다른 한 개 또는 두 개 이상의  
> 변수(독립변수)들로부터 설명하고 예측하는 추측 통계의 한 분야이다.  
> 단순 관계 파악은 상관분석을 통해 이뤄지지만, 회귀분석은 강력한 상관관계가 있다는 것이 전제가 된다.  
> 단순 회귀분석은 독립변수가 1개, 다중 회귀분석은 독립변수가 2개 이상이다.   
> x, y좌표를 갖는 2차원, 그리고 단순 선형회귀라는 가정 하에,  
> 우리가 구하고자 하는 회귀식은 아래와 같은 형태의 식을 따른다.  
![선형회귀식](https://user-images.githubusercontent.com/98927470/170662874-99eda917-ecf8-4999-b91c-221c0e1416bc.PNG)  
> 입실론을 제외한 각 항은 각각 미지의 절편과 기울기를 나타낸다.  
> 입실론은 오차항을 나타내며, 대개 오차항은 N(0, σ^2)를 따른다.  
> 데이터들과 회귀식(회귀선) 사이 잔차(Root Mean Square Error)가 가장 낮게되는 회귀식을 찾는 것이 목표이며,  
> 이를 통해 독립변수에 따른 종속변수의 값을 예측할 수 있다.
  
# 최근접 이웃(K-Nearest Neighbors) 예제
# 실행
> 과정은 위와 동일하며, 모델을 아래와 같이 변경한다.  
> 주변 3개의 데이터를 기준으로 데이터를 분류하게 된다.  
> "knn.py"를 실행한다.
```python
import sklearn.neighbors
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
```
  
# 결과
![Result](https://user-images.githubusercontent.com/98927470/170181379-4a9e0d73-57be-4009-be06-6bc44ad6c0de.PNG)
  
