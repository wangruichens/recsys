# DCN

实际实现上没有严格按照paper实现。 因为criteo的数据我几乎没有做任何处理，除了使用mean填充null字段。dense num feature很容易造成数值不稳定(numerical stability)。尝试针对number feature 采用tf.log(x + 1.0)。但还是很不理想。直接丢到cross layer里面很容易造成 Nan loss.

最后采用所有特征embedding, 然后在embedding层上做dcn。