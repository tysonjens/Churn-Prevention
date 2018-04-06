
## Churn Prevention

Online retailers have a difficult time measuring a single customer's activity level. There are "one and dones" who make a single purchase. Some customers establish an intense purchase pattern over a short period and then quickly move on. Still others purchase infrequently but steadily, and stay "loyal" for a longer period. It can be hard to keep track of which customers are active.

Many prediction companies offer sophisticated churn prevention algorithms out of the box for a small fee - but savvy companies can establish simple algorithms that achieve similar results on their own.  

In a period of 5-6 hours, me and two colleagues put together the following recommendation to a ride share company on how it could prevent churn.  

Steps:
1. Establish a churn metric
2. Fit a classification model to determine which customers advance toward the metric most quickly.
3. If a model candidate fits predicts well, commit to a controlled experiment.
4. Each week, as customers' probability of churning increases past a threshold, assign them to a test or control group.
5. Provide a "winback offer" to the test group.
6. At the end of the time (or when the populations in test and control provide enough power for a hypothesis test), evaluate the model's predictive power and the offer's ability to win customers back.

___

![image](https://github.com/tysonjens/tysonjens.github.io/blob/master/img/Predicted%20Churn.png?raw=true)



### Approach
 - __company priority__: retain users, if we think they'll churn, give an incentive
 - __company action__: if p(churn) > 0.8 --> assign to test or control
 - __cost/benefit__: incentive costs $10, 1-yr c.l.v. valued at $100, assumption is that winback incentive will work for 20% of recipients


### Cost-benefit Matrix:

 |           |predicted0 | predicted1(churn) |
 |---------- |---------- |----------- |
 |__actual0__   |  0 |  -10|
 |__actual1(churn)__   |  0| 10 (-10 + 100*.2)|

 * Assumes 20% retention rate with incentive
 * Customers are already churning, so if our algorithm misses a customer who is churning, there is "no harm done".  This leads us to look for a precise algorith - one that, when it predicts some is churning, it is generally correct.

__precision__ is our target score metric:
 * many people will churn - we don't care about getting all of them (for now). However, among riders we think are churning we want to predict correctly so we don't waste incentive funds.

 *If the model is working well, and the offer is winning customers back, we could adjust this - but we need more information first.*

### Choose a churn metric
pick by last active day
 -  June 1 (30 days ago): 62% of users have churned
 -  May 1 (60 days ago): 47% of users haves churned

### Fit classification models

*The fun stuff that marketers don't care about :-). We fit several classification models looking for ones that performed high with precision when applied to previously unseen test data.*

|        model | methods | accuracy | precision |
|---------- |---------- |----------- | ----------|
| logistic regression |  cross-val, cv=5 |  0.745| 0.734 |
| knn | 50 neighbors | 0.786 | 0.753 |
| decision tree  | simple tree | 0.712 | 0.720 |
| bagging | 100 trees | 0.735 | 0.782|
| random forest | grid searched | 0.769 | 0.789 |
| AdaBoost | defaults | 0.761 | 0.782 |
| gradient boost | defaults | 0.770 | 0.794 |
| SVM | kernel=linear, C=5 | 0.74 | 0.69 |
| SVM | kernal=rbf, C=1, gamma=0.1 | 0.607 | 0.575 |
| SVM | kernal=poly, defaults | 0.613 | 0.594 |

### Feature Importances for Logistic Regression

*While not the primary focus of this effort, we were interested to learn which features lend most to the churn decision. Among models where we looked at feature importance, similar features were important.*

|features: |avg_dist|avg_surge|surge_pct|trips_in_first_30_days|gets_rated|
|--|--|--|--|--|--|
|Beta|0.193|0.0396| -0.0759|-0.404|-0.039|
|features:|luxury_car_user|weekday_pct|city_Astapor|city_King's Landing|
|Beta|-0.425|  0.016|0.306|-0.426|
|features|city_Winterfell|phone_Android|phone_iPhone|rate_driver|
|Beta|0.055|  0.157| -0.347|   -0.252|

##### Feature Importances with Gradient Boost

![image](https://github.com/tysonjens/tysonjens.github.io/blob/master/img/feature%20importance%20gb%20rf.png?raw=true)

##### Partial dependency plots from Gradient Boost
![image](https://github.com/tysonjens/tysonjens.github.io/blob/master/img/partial_d_6.png?raw=true)

___

![image](https://github.com/tysonjens/tysonjens.github.io/blob/master/img/roc_6models.png?raw=true)

*We chose the model that kept the lowest "False Positive Rate" for corresponding values of "True Positive Rate". This helps us find the model with the best precision, important for ensuring we identify customers that are actually churning.*

___

![image](https://github.com/tysonjens/tysonjens.github.io/blob/master/img/Predicted%20Churn%20(1).png?raw=true)

*Intuitively, customers drift toward "churning" as time since their last trip elapses, but they do so at different rates. Our model helps marketers determine when a particular customer reaches a specific "likelihood of churn" threshold.*

___

![image](https://github.com/tysonjens/tysonjens.github.io/blob/master/img/Predicted%20Churn%20(2).png?raw=true)

*A controlled experiment sorts customers into 3 groups each week. Once customers eclipse the 80% threshold, they are randomly placed into a test or control group.*

___

![image](https://github.com/tysonjens/tysonjens.github.io/blob/master/img/Predicted%20Churn%20(3).png?raw=true)

*The design allows for testing the model's predictions, as well as whether the winback offer is effective. After testing test two hypotheses, we could adjust according to what we learn.*
