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
