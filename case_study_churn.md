### approach
 - __company priority__: retain users, if we think they'll churn, give an incentive
 - __company action__: if p(churn) > 0.8 --> spend $10 on incentive
 - __cost/benefit__: incentive costs $10, c.l.v. valued at $100


 |           |predicted0 | predicted1(churn) |
 |---------- |---------- |----------- |
 |__actual0__   |  keep $10, keep $100|  spend $10, keep $100|
 |__actual1(churn)__   |  keep $10, lose $100| spend $10, lose $100|

cost-benefit matrix:

 |           |predicted0 | predicted1(churn) |
 |---------- |---------- |----------- |
 |__actual0__   |  0 |  -10|
 |__actual1(churn)__   |  0| 20 (-10 + 100*.2)|

__precision__ is our target score metric:
 -  many people will churn, so we don't care about getting all of them, but of the people who will churn, we want to predict right so we don't waste incentive funds.

### picking the 'churn' target
pick by last active day
 -  June 1 (30 days ago): 62% of users have churned
 -  May 1 (60 days ago): 47% of users haves churned

### evaluating models

|        model | methods | accuracy | precision |
|---------- |---------- |----------- | ----------|
| logistic regression |  cross-val, cv=3 |  0.671| 0.676 |
| logistic regression |  cross-val, cv=5 |  0.670| 0.675 |
| decision tree  | simple tree | 0.712 | 0.720 |

### feature importances
features: ['avg_dist', 'avg_surge', 'surge_pct', 'trips_in_first_30_days',
       'luxury_car_user', 'weekday_pct']

logistic: [ 0.17304385  0.07934547 -0.11524761 -0.46862851 -0.40809665 -0.00793641]

|        model | methods | accuracy | precision |
|---------- |---------- |----------- | ----------|
| logistic regression |  cross-val, cv=3 |  0.671| 0.676 |
| logistic regression |  cross-val, cv=5 |  0.670| 0.675 |
| decision tree  | simple tree | 0.712 | 0.720 |
