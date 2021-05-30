# Practice Random Forest through python with Iris dataset in scikit-learn


Scikit-learn의 **Iris dataset**과 **RandomForestClassifier**를 사용해 구현한 모델입니다.

code 폴더에 있는 각 파일들에 대한 설명은 아래와 같습니다.

1. Simple RF model: **전체 데이터**를 사용해서 만든 model에 대해 sample_data를 사용해서 Iris specied 예측을 한다.
2. Advanced RF model: **train : test = 7 : 3**으로 만든 데이터로 학습한 modeld을 사용해 predict species from sample_data를 한다. 
3. FeatureControlled RF model-threshold: 2번의 코드에서 발전시킨 코드로 **threshold 이상의 중요도 값을 가지는 feature만을 사용**하여 model을 만든다.  
4. FeatureControlled RF model-topSelect: 3번과 비슷하지만 feature의 중요도를 구한 후 **상위 몇개만 사용**하여 model을 만드는 차이가 있다.

> Random Forest model이나 코드에 대한 더 자세한 설명이 필요하다면 [practiceRFpage](https://velog.io/@seo106/RandomForestwithIris)를 참고하면 됩니다.
