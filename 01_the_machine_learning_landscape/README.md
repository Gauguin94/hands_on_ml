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
# 그래서 단순 선형 회귀가 뭔데?
  
  
# 최근접 이웃(K-Nearest Neighbors) 예제
  
> 과정은 위와 동일하며, 모델을 아래와 같이 변경한다.  
> 주변 3개의 데이터를 기준으로 데이터를 분류하게 된다.  
> "knn.py"를 실행한다.
```python
import sklearn.neighbors
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
```
  
# 결과
![Result](https://user-images.githubusercontent.com/98927470/170181379-4a9e0d73-57be-4009-be06-6bc44ad6c0de.PNG)
  
