<div align="center">

# SSTKG: Simple Spatio-Temporal Knowledge Graph for Intepretable and Versatile Dynamic Information Embedding

[![Venue:WWW 2024](https://img.shields.io/badge/Venue-WWW%202024-007CFF)](https://arxiv.org/pdf/2402.12132)

</div>


# Abstract
Knowledge graphs (KGs) have been increasingly employed for link prediction and recommendation using real-world datasets. However, the majority of current methods rely on static data, neglecting the dynamic nature and the hidden spatio-temporal attributes of real-world scenarios. This often results in suboptimal predictions and recommendations. Although there are effective spatio-temporal inference methods, they face challenges such as scalability with large datasets and inadequate semantic understanding, which impede their performance. To address these limitations, this paper introduces a novel framework - Simple Spatio-Temporal Knowledge Graph (SSTKG), for constructing and exploring spatio-temporal KGs. To integrate spatial and temporal data into KGs, our framework exploited through a new 3-step embedding method. Output embeddings can be used for future temporal sequence prediction and spatial information recommendation, providing valuable insights for various applications such as retail sales forecasting and traffic volume prediction. Our framework offers a simple but comprehensive way to understand the underlying patterns and trends in dynamic KG, thereby enhancing the accuracy of predictions and the relevance of recommendations. This work paves the way for more effective utilization of spatio-temporal data in KGs, with potential impacts across a wide range of sectors.
<!-- Code for paper SSTKG: Simple Spatio-Temporal Knowledge Graph for Intepretable and Versatile Dynamic Information Embedding presented The Web Conference 2024 -->

arxiv link: https://arxiv.org/abs/2402.12132

slide link: 

poster link:

## Graph formation
### Static embedding
encapsulates the static attributes of an entity, yielding a representation that remains invariant over time

### Dynamic Embedding - Out
signifies the potential influence an entity may impart upon its linked entities

### Dynamic Embedding - In
quantifies the influence that an entity receives from its associated entities,reflecting the cumulative impact of these relationships on the entity

## Embedding Training Algorithm
two steps, embedding first then influence matrix

# Citing:
The paper can be cited using following:

```
@article{yang2024sstkg,
  title={SSTKG: Simple Spatio-Temporal Knowledge Graph for Intepretable and Versatile Dynamic Information Embedding},
  author={Yang, Ruiyi and Salim, Flora D and Xue, Hao},
  journal={arXiv preprint arXiv:2402.12132},
  year={2024}
}
```
