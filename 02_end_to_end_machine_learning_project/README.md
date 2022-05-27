# 다중 회귀 분석(Multiple Regression Analysis)
  
> StarLib 저장소에 있는 캘리포니아 주택 가격 데이터셋을 사용한다.  
> 이 데이터셋은 1990년 캘리포니아 인구조사 데이터를 기반으로 한다.  
> 구역의 인구, 중간 소득, 방 개수 등의 여러 특성을 사용하여 추론을 진행하므로 이는 다중 회귀 분석이며,  
> 최종적으로 하나의 값을 도출하기 때문에 단변량 회귀 문제이다.  
  
![히스토그램](https://user-images.githubusercontent.com/98927470/170680197-a25f211c-7654-425a-9b7b-a4c73f4fc380.PNG)
![산점히스토](https://user-images.githubusercontent.com/98927470/170680302-273895f3-6b9f-485a-bcfb-387749af55fb.PNG)
![산점도](https://user-images.githubusercontent.com/98927470/170680151-3f52db02-09fe-4f91-8671-312d75bbf503.PNG)
  
> 목표는 주택 가격을 예측하는 것이기 때문에, 주택 가격에 따른 각 특성과의 관계를 파악한다.  
> 히스토그램을 살펴보면, 중간 주택 가격은 중간 소득과 양의 상관관계에 놓여있음을 확인할 수 있다.  

# 파이프라인(Pipeline)
  
> 본격적으로 데이터를 다루기 전에, 우리는 우리의 목표에 대해 제대로 파악할 필요성과 어떻게 구현할 지에 관에 생각해야 한다.  
> 앞선 히스토그램 등의 시각화를 이용해 독립변수와 종속변수들 간 상관관계를 파악하였다.  
> 하지만 이는 날 것 그 자체를 이용하여 확인한 것이다.  
> 우리는 데이터 분석가를 목표로 하기 때문에 무의미한 데이터를 유의미하게 만들 궁리도 해보는 사고를 할 수 있어야 한다.  
> 여러 특성의 조합을 시도해보자.  
> 특정 구역의 방 개수만을 놓고 보면 이는 무의미한 데이터가 될 수 있다.  
> 하지만 가구당 방 개수를 따져본다면 유의미한 데이터가 될 수도 있다.  
> 마찬가지로, 침실 개수 자체로는 유용하지 않으며 방 개수와 비교해본다면 유의미한 결론을 도출할 수도 있다.  
> (In hands-on machine learning)  
![특성원본](https://user-images.githubusercontent.com/98927470/170694889-6f2e6617-218f-44cb-9e54-5cffd4bd143d.PNG)
![특성조작](https://user-images.githubusercontent.com/98927470/170694976-5f3182d9-4247-42d2-97e5-a9704ca43890.PNG)
```python
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
```
> 상단의 왼쪽 결과는 특성 조작없이 상관관계를 계산한 것이며, 오른쪽 결과는 코드와 같이  
> 가구당 방 개수, 방 개수 대비 침대 개수, 가구당 인구수를 새로운 특성으로 추가한 것이다.  
> 추가된 특성에서 이전보다 조금 더 좋은 결과를 갖는 경우도 있지만 대체적으로 유의미한 데이터는 아닌 것 같다...  
![신뢰도](https://user-images.githubusercontent.com/98927470/170680443-4bd33bb2-2129-4dbb-80aa-94cf661da037.PNG)
![선형회귀오차](https://user-images.githubusercontent.com/98927470/170680509-db10b0cb-a23d-4436-8454-c16f19a21f68.PNG)
![결정트리오차](https://user-images.githubusercontent.com/98927470/170680482-1c7a94bd-b1b6-445f-ac8d-113730e98f4e.PNG)
![랜덤포레스트오차](https://user-images.githubusercontent.com/98927470/170680549-1324a91c-ac72-4725-84d3-4d536705b84f.PNG)
![파라미터찾기](https://user-images.githubusercontent.com/98927470/170680596-96d11b16-324e-4b5e-ae6b-a4ca4655cf6d.PNG)
![특성중요도](https://user-images.githubusercontent.com/98927470/170680633-bfbe3f11-4045-4043-913f-33a03382b4f7.PNG)
