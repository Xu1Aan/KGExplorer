---
license: mit
---

# 1 Source
Source: https://github.com/alibaba-research/ChineseBLUE

# 2 Definition of the tagset
```python
tag_set = [
 'B_手术',
 'I_疾病和诊断',
 'B_症状',
 'I_解剖部位',
 'I_药物',
 'B_影像检查',
 'B_药物',
 'B_疾病和诊断',
 'I_影像检查',
 'I_手术',
 'B_解剖部位',
 'O',
 'B_实验室检验',
 'I_症状',
 'I_实验室检验'
 ]
 
tag2id = lambda tag: tag_set.index(tag)
id2tag = lambda id: tag_set[id]
```
 

# 3 Citation
To use this dataset in your work please cite:

Ningyu Zhang, Qianghuai Jia, Kangping Yin, Liang Dong, Feng Gao, Nengwei Hua. Conceptualized Representation Learning for Chinese Biomedical Text Mining

```
@article{zhang2020conceptualized,
  title={Conceptualized Representation Learning for Chinese Biomedical Text Mining},
  author={Zhang, Ningyu and Jia, Qianghuai and Yin, Kangping and Dong, Liang and Gao, Feng and Hua, Nengwei},
  journal={arXiv preprint arXiv:2008.10813},
  year={2020}
}
```
