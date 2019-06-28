# DCN

实际实现上没有严格按照paper实现。 因为criteo的数据我几乎没有做任何处理，除了使用mean填充null字段。dense num feature很容易造成数值不稳定(numerical stability)。

初始时对number feature 采用tf.log(x + 1.0)。发现还是会经常报错nan loss。 

后来注意到数据集numerical column "_c2" 列，包含一些负值, 最小值为-3。 这也就是加上relu后能解决的原因。一些负无穷的值被抑制到0了。更合适的做法还是应该先把_c2这一列特殊处理一下。


最后采用所有特征embedding, 然后在embedding层上做dcn。