# News Recommendation using Mind Dataset
Abdus Khan & Ankush Gupta

# Table of Content
1. Problem Statement
3. NRMS Model (Neural News Recommendation with Multi-Head Self-Attention)
4. Why NRMS?
5. Data Format
6. News Data
7. Behaviors Data
8. Model Training
9. Evaluation Metrics
10. Conclusion

# Problem Statement
News recommender systems are aimed to personalize users experiences and help them to discover relevant articles from a large and dynamic search space. Therefore, news domain is a challenging scenario for recommendations, due to its sparse user profiling, fast growing number of items, accelerated item’s value decay, and users preferences dynamic shift.

# NRMS Model
NRMS is a neural news recommendation approach with multi-head selfattention. The core of NRMS is a news encoder and a user encoder. In the newsencoder, a multi-head self-attentions is used to learn news representations from news titles by modeling the interactions between words. In the user encoder, we learn representations of users from their browsed news and use multihead self-attention to capture the relatedness between the news. Besides, we apply additive attention to learn more informative news and user representations by selecting important words and news.

# Why NRMS?
1. NRMS is a content-based neural news recommendation approach.
2. It uses multi-self attention to learn news representations by modeling the iteractions between words and learn user representations by capturing the relationship between user browsed news.
3. NRMS uses additive attentions to learn informative news and user representations by selecting important words and news.

# Data Format
For quicker training and evaluaiton, we sample MINDdemo dataset of 5k users from MIND small dataset.
MINDsmall_train is used for training, and MINDsmall_dev is used for evaluation. Training data and evaluation data are composed of a news file and a behaviors file. 

# News Data
In general, each line in data file represents information of one piece of news:

[News ID] [Category] [Subcategory] [News Title] [News Abstrct] [News Url] [Entities in News Title] [Entities in News Abstract]

Example:

N46466 lifestyle lifestyleroyals The Brands Queen Elizabeth, Prince Charles, and Prince Philip Swear By Shop the notebooks, jackets, and more that the royals can't live without. https://www.msn.com/en-us/lifestyle/lifestyleroyals/the-brands-queen-elizabeth,-prince-charles,-and-prince-philip-swear-by/ss-AAGH0ET?ocid=chopendata [{"Label": "Prince Philip, Duke of Edinburgh", "Type": "P", "WikidataId": "Q80976", "Confidence": 1.0, "OccurrenceOffsets": [48], "SurfaceForms": ["Prince Philip"]}, {"Label": "Charles, Prince of Wales", "Type": "P", "WikidataId": "Q43274", "Confidence": 1.0, "OccurrenceOffsets": [28], "SurfaceForms": ["Prince Charles"]}, {"Label": "Elizabeth II", "Type": "P", "WikidataId": "Q9682", "Confidence": 0.97, "OccurrenceOffsets": [11], "SurfaceForms": ["Queen Elizabeth"]}] []

# Behaviors Data
In general, each line in data file represents one instance of an impression. The format is like:

[Impression ID] [User ID] [Impression Time] [User Click History] [Impression News]

Example: 
1 U82271 11/11/2019 3:28:58 PM N3130 N11621 N12917 N4574 N12140 N9748 N13390-0 N7180-0 N20785-0 N6937-0 N15776-0 N25810-0 N20820-0 N6885-0 N27294-0 N18835-0 N16945-0 N7410-0 N23967-0 N22679-0 N20532-0 N26651-0 N22078-0 N4098-0 N16473-0 N13841-0 N15660-0 N25787-0 N2315-0 N1615-0 N9087-0 N23880-0 N3600-0 N24479-0 N22882-0 N26308-0 N13594-0 N2220-0 N28356-0 N17083-0 N21415-0 N18671-0 N9440-0 N17759-0 N10861-0 N21830-0 N8064-0 N5675-0 N15037-0 N26154-0 N15368-1 N481-0 N3256-0 N20663-0 N23940-0 N7654-0 N10729-0 N7090-0 N23596-0 N15901-0 N16348-0 N13645-0 N8124-0 N20094-0 N27774-0 N23011-0 N14832-0 N15971-0 N27729-0 N2167-0 N11186-0 N18390-0 N21328-0 N10992-0 N20122-0 N1958-0 N2004-0 N26156-0 N17632-0 N26146-0 N17322-0 N18403-0 N17397-0 N18215-0 N14475-0 N9781-0 N17958-0 N3370-0 N1127-0 N15525-0 N12657-0 N10537-0 N18224-0

# Model Training

Hyper Parameters
```

hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)
```
                          
Model Initialization
```
model = NRMSModel(hparams, iterator, seed=1)
```

Model Fitting
```
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
```

# Model Evaluation
```
res_syn = model.run_eval(valid_news_file, valid_behaviors_file)
```
# Model Metrics

1. MRR

![1DzpFR1DToUSxBCDKJ4GSNA](https://user-images.githubusercontent.com/71584739/176639669-1043b2d0-8e2a-4df3-ac83-143f25d0f9d8.png)

2. NDCG

- Gain for an item is essentially the same as the relevance score, which can be numerical ratings like search results in Google which can be rated in scale from 1 to 5, or binary in case of implicit data where we only know if a user has consumed certain item or not.
Naturally Cumulative Gain is defined as the sum of gains up to a position k in the recommendation list.

![1GEvXfCqT6hq_KNT_WMnRFA](https://user-images.githubusercontent.com/71584739/176639966-0e2d84ca-3cba-4742-a8e3-e7f5852cec0e.png)

- One obvious drawback of CG is that it does not take into account of ordering. By swapping the relative order of any two items, the CG would be unaffected. This is problematic when ranking order is important. For example, on Google Search results, you would obviously not like placing the most relevant web page at the bottom.
To penalize highly relevant items being placed at the bottom, we introduce the DCG

![DCG(k) =](https://user-images.githubusercontent.com/71584739/176640119-24d97c68-7a0d-4aeb-a73c-d83264f23e11.png)

- By dividng the gain by its rank, we sort of push the algorithm to place highly relevant items to the top to achieve the best DCG score.
There is still a drawback of DCG score. It is that DCG score adds up with the length of recommendation list. Therefore, we cannot consistently compare the DCG score for system recommending top 5 and top 10 items, because the latter will have higher score not because its recommendation quality but pure length.
We tackle this issue by introducing IDCG (ideal DCG). IDCG is the DCG score for the most ideal ranking, which is ranking the items top down according their relevance up to position k.

![1cDC8roXZrP-iUeR1vlmGBQ](https://user-images.githubusercontent.com/71584739/176640203-f5d0da9d-0ccc-43ad-a5ae-5aee05e18872.png)

- And NDCG is simply to normalize the DCG score by IDCG such that its value is always between 0 and 1 regardless of the length.

![NDCG(k)](https://user-images.githubusercontent.com/71584739/176640281-e738edf1-1b4c-4a54-b848-393cca9f3281.png)

# Conclusion

1. AUC: 0.63
2. Mean MRR: 0.29
3. NDCG@10: 0.381
4. NDCG@5: 0.315









