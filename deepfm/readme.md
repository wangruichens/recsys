# DeepFM Performance Analysis

采用tf.distribute.MirroredStrategy()。2块1080ti GPU，优化input_fn后，GPU资源基本可以满载训练，大概 11 global_step/sec。

estimator打出来的log可以看到，一块GPU和两块GPU的 global_step/sec几乎相同。但是实际上两块GPU每个step跑的是两份数据。[参考这里](http://keep.01ue.com/?pi=774414&_a=app&_c=index&_m=p)。比如同样条数的训练数据,同样epoch, 单GPU需要1000 step, 双GPU可能只需要500,600 step就可以跑完一轮。

```angular2
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 430.09       Driver Version: 430.09       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:01:00.0 Off |                  N/A |
| 55%   74C    P2   145W / 250W |  10736MiB / 11178MiB |     84%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 108...  Off  | 00000000:02:00.0 Off |                  N/A |
| 50%   69C    P2   130W / 250W |  10747MiB / 11178MiB |     82%      Default |
+-------------------------------+----------------------+----------------------+
```


- 两块GPU
```angular2
INFO:tensorflow:loss = 0.37087572, step = 30 (0.877 sec)
INFO:tensorflow:global_step/sec: 11.4237
INFO:tensorflow:loss = 0.35228056, step = 40 (0.875 sec)
INFO:tensorflow:global_step/sec: 11.5256
INFO:tensorflow:loss = 0.3553083, step = 50 (0.868 sec)
INFO:tensorflow:global_step/sec: 11.6155
```

- 一块GPU
```angular2
INFO:tensorflow:loss = 0.24809906, step = 1140 (0.962 sec)
INFO:tensorflow:global_step/sec: 11.379
INFO:tensorflow:loss = 0.23750155, step = 1150 (0.879 sec)
INFO:tensorflow:global_step/sec: 11.0414
INFO:tensorflow:loss = 0.28272614, step = 1160 (0.906 sec)
INFO:tensorflow:global_step/sec: 11.0983

```

使用grpc_client调用deepfm模型，serving在cpu上, 平均时间：
```angular2
Batch size:  500
Predict AUC:  0.656185817244856
Predict time used: 0.37ms

Batch size:  500
Predict AUC:  0.6375953159041394
Predict time used: 0.36ms

Batch size:  200
Predict AUC:  0.6865367839488224
Predict time used: 0.32ms

Batch size:  200
Predict AUC:  0.6835420068953004
Predict time used: 0.29ms
```