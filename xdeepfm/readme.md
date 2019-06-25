# xDeepFM Performance Analysis

与deepfm一样，使用两块gpu训练。训练速度其实主要取决于embedding layers的大小。

论文中CIN最终结果 logloss=0.4493. auc=0.8012。\
实际实现最终结果 logloss=0.4548. auc=0.7950。

略低于论文，可能是ffm数据预处理 & 超参数的原因。

[对比deepfm](../deepfm/readme.md)

![auc](auc.png)

Training speed:
```angular2
INFO:tensorflow:loss = 0.45753595, step = 15350 (4.233 sec)
INFO:tensorflow:global_step/sec: 12.3989
INFO:tensorflow:loss = 0.4691028, step = 15400 (4.033 sec)
INFO:tensorflow:global_step/sec: 12.9861
INFO:tensorflow:loss = 0.469327, step = 15450 (3.850 sec)
INFO:tensorflow:global_step/sec: 12.5842
```

AUC:
```angular2

INFO:tensorflow:Evaluation [200/200]
INFO:tensorflow:Finalize strategy.
INFO:tensorflow:Finished evaluation at 2019-06-25-02:38:15
INFO:tensorflow:Saving dict for global step 20300: AUC = 0.79502136, global_step = 20300, loss = 0.45486304
```


