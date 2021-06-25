# Practice Random Forest through python with Iris dataset in scikit-learn


Scikit-learn의 **Iris dataset**과 **RandomForestClassifier**를 사용해 구현한 모델입니다.

Dataset
---
![IrisDatasetTable](https://user-images.githubusercontent.com/66738234/123362680-07fc5280-d5ac-11eb-8470-512c2af0bc92.png)
- feature(분류 기준): 꽃받침(sepal) 길이, 꽃받침 넓이, 꽃잎(petal) 길이, 꽃잎 넓이
- targets(분류되는 결과): setosa=0, versicolor=1, virginica=2
- 총 150개 데이터

Files in 'Code' folder
---
code 폴더에 있는 각 파일들에 대한 설명은 아래와 같습니다.

1. [Simple RF model](https://github.com/SeoK106/RandomForest_with_Iris-data/blob/master/code/Simple%20RF%20model.py): **전체 데이터**를 사용해서 만든 model에 대해 sample_data를 사용해서 Iris specied 예측을 한다.
2. [Advanced RF model](https://github.com/SeoK106/RandomForest_with_Iris-data/blob/master/code/Advanced%20RF%20model.py): **train : test = 7 : 3**으로 만든 데이터로 학습한 modeld을 사용해 predict species from sample_data를 한다. 
3. [FeatureControlled RF model-threshold](https://github.com/SeoK106/RandomForest_with_Iris-data/blob/master/code/FeatureControlledRFmodel_threshold.py): 2번의 코드에서 발전시킨 코드로 **threshold 이상의 중요도 값을 가지는 feature만을 사용**하여 model을 만든다.  
4. [FeatureControlled RF model-topSelect](https://github.com/SeoK106/RandomForest_with_Iris-data/blob/master/code/FeatureControlledRFmodel_topSelect.py): 3번과 비슷하지만 feature의 중요도를 구한 후 **상위 몇개만 사용**하여 model을 만드는 차이가 있다.

Experimental Result
---
모든 코드는 python IDLE를 통해 실행하거나 cmd로 ``` >python fileName.py```이런식으로 실행하면 된다.<br>
cf. Window cmd에서 띄어쓰기가 있는 파일명은 ``` >python "file name.py"```이렇게 쓰면 된다.

1. Simple RF model
![SimpleRFmodel](https://user-images.githubusercontent.com/66738234/123363489-9f15da00-d5ad-11eb-99e0-eec34d4015fb.png)
2. Advanced RF model
![AdvancedRFmodel](https://user-images.githubusercontent.com/66738234/123363502-a63ce800-d5ad-11eb-9dd9-b63f380fcf2c.png)
3. FeatureControlled RF model-threshold
![threshold](https://user-images.githubusercontent.com/66738234/123363811-4266ef00-d5ae-11eb-9589-a7cd57da1093.png)
4. FeatureControlled RF model-topSelect
![topSelect2](https://user-images.githubusercontent.com/66738234/123363544-ba80e500-d5ad-11eb-87a3-5e1f16d56157.png)
![graph](https://user-images.githubusercontent.com/66738234/123363922-793d0500-d5ae-11eb-968b-168ed939fea6.png)

> ### Random Forest model이나 코드에 대한 더 자세한 설명이 필요하다면 [Here](https://velog.io/@seo106/RandomForestwithIris)를 참고하면 됩니다.
