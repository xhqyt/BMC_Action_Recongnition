classifier_coarse 生成h5文件保存粗模型
classifier_fine 生成h5文件保存细模型
数量上fine>coarse
coarse是对于每一个特征训练一个模型，fine是对于每一个特征训练多个模型
cluster_to_divide_group是将特征分类到多个特征集
compute_mean_vector生成均值向量作为特征




代码改动：
目录所在位置应该是当前目录，将basepath:../改为./