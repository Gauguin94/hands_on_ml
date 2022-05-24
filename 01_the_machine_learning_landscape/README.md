# 선형 회귀(Linear Regression) 예제
  
> 1. 국가별 국내총생산(GDP)과 삶의 만족도를 엮어 데이터를 만든다.
```python
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
```
> 2. sk.learn을 통해 linear_model을 만들고 앞서 만든 데이터를 입력으로 하여 모델을 학습시킨다.
```python
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)
```
> 3. 학습이 종료된 모델에 새로운 데이터('키프로스'의 국내총생산)를 입력으로 하여 예측(출력)값을 확인한다.
```python
X_new = [[22587]]
print('Republic of Cyprus's Life satisfaction: {}'.format(model.predict(X_new)))
```
  
# 실행
> 제작한 코드를 Windows나 Linux 체제에서 실행시킬 것을 고려하여, 'dataloader'라는 클래스를 구현.
> dataloader 내 존재하는 whichOS는 OS에 맞춰 데이터를 불러오는 역할을 한다.  
> data에는 "oecd_bli_2015.csv"와 "gdp_per_capita.csv"의 데이터가 존재한다.
> ### **example**
```python
  data = dataloader.whichOS("isWindows")
  oecd_bli = data[0]
  gdp_per_capita = data[1]
```
```python
  data = dataloader.whichOS("isLinux")
```

# 결과
![Result](https://user-images.githubusercontent.com/98927470/170038411-0431889f-f47c-4048-b50a-678c11c57953.PNG)
  
  