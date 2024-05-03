
# From simplistic supervised to advances self-supervised OOD Detection

Context : 
This project delves into the issue of Textual Out-Of-Distribution (OOD) detection, which refers to the capability of machine learning models to recognize data samples that significantly deviate from their training data distribution. In Natural Language Processing (NLP) applications, Textual OOD detection is critical to ensuring the robustness and depend- ability of production systems. This study investigates the effectiveness of various methods for OOD detection in NLP, utilizing a transformer-based language model and different datasets with varying degrees of similarity to the training data. Our findings demonstrate that both the Mahalanobis-based score utilizing the last layer representation and the Cosine Projection score utilizing the average latent representation outperform the other scores in terms of AUROC. However, the supervised approach did not perform as well. 

---
abstract: |
  This essay delves into the issue of Textual Out-Of-Distribution (OOD)
  detection, which refers to the capability of machine learning models
  to recognize data samples that significantly deviate from their
  training data distribution. In Natural Language Processing (NLP)
  applications, Textual OOD detection is critical to ensuring the
  robustness and dependability of production systems. This study
  investigates the effectiveness of various methods for OOD detection in
  NLP, utilizing a transformer-based language model and different
  datasets with varying degrees of similarity to the training data. Our
  findings demonstrate that both the Mahalanobis-based score utilizing
  the last layer representation and the Cosine Projection score
  utilizing the average latent representation outperform the other
  scores in terms of AUROC. However, the supervised approach did not
  perform as well. Code is available on github [^1]
author:
- |
  Camille Langlois  
  Ensae  
  `camille.langlois@ensae.fr`  
  Joé Vincent-Galtié  
  Ensae  
  `joe.vincent-galtie@ensae.com`  
bibliography:
- biblio.bib
title: "Exploring Textual Out-Of-Distribution Detection: from simplistic
  supervised to advanced self-supervised techniques"
---

# Problem Framing

Increasing the use of black-box machine learning models comes with
various critical safety issues, among which we can mention the Textual
Out-of-Distribution (OOD) detection . The goal of OOD detection is to
identify instances that are significantly different from the
distribution of the training data, which can be caused by various
factors such as errors, noise, or deliberate attempts to deceive the
system . However, distinguishing OOD from in-distribution (ID) examples
is difficult for modern deep neural architectures , as these models
transform incoming data into latent representations that make reliable
information extraction challenging. In the present paper, we adress the
issue of OOD detection on classifiers for textual data, we will more
particularly focus on models based on Transformer architectures .

The existing methods adressing the OOD detection issue can be
categorized based on their positioning with respect to the network,
including those that use incoming data , robust constraints during
training , and post-processing methods. The post-processing methods are
considered the most promising because they do not require retraining and
can be used on any pretrained model. These methods include softmax-based
tools that compute a confidence score based on predicted probabilities
and threshold , projections of the pre-softmax layer , and the
Mahalanobis distance between a test sample and the in-distribution law
estimated through accessible training data points . Other approaches
based on the concept of data depth have arisen to overcome the drawbacks
of distance-based scores, such as using the Integrated Rank-Weighted
depth .

In this essay, we present an experimentation on the performance of
different OOD detection techniques on a benchmark dataset of text
classification tasks . We first propose a simplistic supervised method
relying on a XGBoost model. We also evaluate the effectiveness of
various methods, including scores based on Mahalanobis-distance
computation and Cosine Projection, in detecting OOD examples in both
in-domain and out-of-domain datasets. Our results demonstrate the
strengths and limitations of different approaches and provide insights
for future research on improving the reliability of OOD detection in NLP
tasks.

# Experiments Protocol

In this section we will introduce the chosen benchmark, the pretrained
encoders and the baseline methods that we experimented in order to
compare the results.

## Datasets selection

During the experiments, we are going to consider three different
datasets. The models will be trained on one of the datasets, which will
then correspond to the in-distribution data. In this case, the three
datasets chosen are SST2[^2], IMDB[^3] and RTE[^4]. SST2 (see Table
<a href="#table:SST2" data-reference-type="ref" data-reference="table:SST2">1</a>)
is a sentiment analysis dataset that contains movie reviews with binary
labels indicating positive or negative sentiment. On the other hand,
IMDB (see Table
<a href="#table:IMDB" data-reference-type="ref" data-reference="table:IMDB">2</a>)
is also a sentiment analysis dataset but it contains reviews of a wider
range of products, such as books, electronics, and home appliances. RTE
(see Table
<a href="#table:RTE" data-reference-type="ref" data-reference="table:RTE">3</a>)
consists of pairs of sentences, where the task is to determine whether
one sentence entails, or contradicts with respect to the other. In our
case SST2 represents the ID data. Since IMDB and RTE are not part of the
distribution on which the models were trained on, it serves as the OOD
datasets in this experiment.

<div class="center">

<div id="table:SST2">

|                         |         |
|:-----------------------:|:--------|
|      # of samples       | 67 349  |
| Average sentence length | 19.8    |
|      # of classes       | 2       |
|        Language         | English |

Features of the SST2 dataset.

</div>

</div>

<div class="center">

<div id="table:IMDB">

|                         |         |
|:-----------------------:|:--------|
|      # of samples       | 50 000  |
| Average sentence length | 231.73  |
|      # of classes       | 2       |
|        Language         | English |

Features of the IMDB dataset.

</div>

</div>

<div class="center">

<div id="table:RTE">

|                         |        |
|:-----------------------:|:-------|
|      # of samples       | 2 490  |
| Average sentence length | 68.6   |
|      # of classes       | 2      |
|        Language         | Multi. |

Features of the RTE dataset.

</div>

</div>

## Pretrained model selection

The experiments regarding the scorers have been done regarding a
pretrained encoder. We apply the various scorers on the BERT model. The
selected model has been pretrained and fine-tuned on SST2 dataset and
will be used to extract features from the input text for both IN and OOD
datasets. We have chosen this model because it is among the most widely
used and effective model for NLP tasks.

## Simplistic supervised OOD Detection

Our supervised approach of the OOD detection is simplistically framed.
Using PyTorch and the BERT model already fine-tuned on the SST2 dataset,
we iterated on batches of SST2 samples in order to retrieve hidden
states corresponding to each of the 13 hidden layers.

For a batch of 8 sentences, we therefore have for each layer 8 matrix of
dimension *T* × *d* where *T* is the number of tokens per sentences and
*d* is the embedding dimension.

We introduce the following notation:

-   ∀*b* ∈ {1, ..., *B*} where B is the number of batches

-   ∀*l* ∈ {1, ..., *L*} where L is the number of layers (13)

-   ∀*x*<sub>*i*, *b*</sub> for *i* ∈ \[1,8\] , 8 being the size of the
    batch

-   *H*<sub>*x*<sub>*i*, *b*</sub></sub><sup>*l* = 1</sup> = (*h*<sub>*i*, *j*</sub><sup>*l* = 1</sup>) ∈ *M*<sub>*T* × 768</sub>

-   $\\bar{x}\_{i,b} = (\\frac{1}{13}\\sum^{13}\_{l=1} h^{l}\_{1,1}, ..., \\frac{1}{13}\\sum^{13}\_{l=1} h^{l}\_{1,768})$

-   *X*<sub>*S**S**T*2</sub> = (*x̄*<sub>1, 1</sub>,...,*x̄*<sub>8, 1</sub>,*x̄*<sub>1, 2</sub>,...)

The same steps are processed for IMDB (out-ds) and RTE (very-out)
datasets.

We then concatenated *X*<sub>*S**S**T*2</sub> and
*X*<sub>*I**M**D**B*</sub>, and *X*<sub>*S**S**T*2</sub> and
*X*<sub>*R**T**E*</sub>, completed by labels, 0 if in-ds, 1 if out or
very-out ds.

We apply a classification supervised algorithm. In particular we applied
a XGBoost , one of the current most performant classification algorithm.

## Self-supervised OOD Detection : Scorers

For our experiments we considered two different methods:

-   The Mahalanobis based score : this score measures the distance
    between a given input and the distribution of ID examples in the
    latent space of a pretrained language model. The Mahalanobis
    distance takes into account the covariance matrix of the ID
    examples, and thus can better capture the distribution of the data
    than other distance metrics like Euclidean distance or Cosine
    distance. It can be computed as:

    <div class="center">

    $$d\_{Mah}(\\textbf{x}) = \\sqrt{((f(\\textbf{x})-\\mu)^TS^{-1}(f(\\textbf{x})-\\mu))}$$

    </div>

    where f(**x**) represents the latent representation for a given
    input example **x**, *μ* is the mean of the ID training data, and
    *S* is the covariance matrix of the ID training data. In our study
    we will consider the latent representation to be either the vector
    of activations in the last hidden layer of the neural network and
    the average representation of all the layers.

-   The Cosine Projection based score: this is a commonly used metric to
    compare the similarity between a given input and the distribution of
    ID examples using the Cosine similarity. Given two vectors *u* and
    *v*, their Cosine similarity score is defined as:
    $$\\text{cosine\\\_sim}(u,v) = \\frac{u\\cdot v}{\|\|u\|\| \|\|v\|\|}$$
    More precisely, the cosine similarity score can be used to compare
    the similarity between the latent representations of a given text
    sample and the ID examples used during the training of a language
    model. Specifically, given a test sample **x** and a set of ID
    examples *X*<sub>in</sub>, we can compute the Cosine similarity
    between the latent representation of **x**, denoted as *f*(**x**),
    and the average latent representation of the ID examples, denoted as
    *f*(*X*<sub>in</sub>):
    cos_score(**x**) = cosine_sim(*f*(**x**),*f*(*X*<sub>in</sub>))
    Again, we will on one side consider only the latent representation
    of the last layer, and on the other side the average of the
    representations of each layer.

In both cases, a threshold is used to classify the test sample as either
ID or OOD. If the score is below the threshold, the sample is classified
as ID, otherwise it is classified as OOD. The threshold can be set using
a validation set or a predefined value.

We can also note that both methods require access to ID training data to
estimate the mean and covariance matrix for the Mahalanobis-based score,
and the mean for the Cosine Projection score.

## Last vs average of every embedding layer

Following , we tested the scorer on the embeddings of the last layer and
on the average embedding of all hidden layers.

To proceed so is a way to distinguish whether the intuition that more
information comes from agregating results of each layer.

Hence, for the second case, the input $\\math{x}$ is such that
$\\math{x} \\in \\math{R}^{d=768}$ and
$\\math{x} = \\sum^{L}\_{l=1} \\math{x}^{l}\_{1}$ where
$\\math{x}^{l}\_{1} \\in \\math{R}^{d=768}$ corresponds to the l-th
hidden state of the first token

## Evaluation metrics

There exist several ways to measure the effectiveness of an OOD method.
Here will focus on the Area Under the Receiver Operating Characteristic
curve (AUROC) metric , which is a commonly used evaluation metric to
assess the performance of a model in distinguishing between ID and OOD
samples.

The ROC curve refers to a plot of the True Positive Rate (TPR) against
the False Positive Rate (FPR) for different threshold values. In the
context of OOD detection, the TPR represents the proportion of correctly
identified OOD samples, while the FPR represents the proportion of
incorrectly identified ID samples as OOD.

AUROC corresponds to the area under the ROC curve, ranging from 0.0 to
1.0. An AUROC score of 1.0 indicates perfect performance, while a score
of 0.5 indicates random guessing. A score below 0.5 indicates poor
performance, which means the model is worse than random guessing.

# Results

## Simplisitic supervised approach

This approach did not give convincing results. In fact, the XGBoost gave
a too perfect accuracy for classifying SST2 vs IMDB and SST2 vs RTE to
make this approach worth it.

After a deeper study of the results, this perfect accuracy was induced
by a dimension of the mean embedding, the 135th, that was always
negative for SST2 and always positive for IMBD.

We did not figure out yet why this happened. However it is worth
noticing that this approach has obvious limits :

-   This supposes having access to numerous diversed OOD sentences and
    implies a too heavy labelling work on real issues.

## Self-supervised approach 

The experimentation performed on the methods and datasets mentioned
above enable to have some insights regarding which scorers are the most
adapted taking into account the model used and the datasets. We can for
instance see in Figure
<a href="#table:figure_distrib" data-reference-type="ref" data-reference="table:figure_distrib">1</a>
the distribution of the scorers for each dataset. These graphs allow us
to notice that the scorers distributions of in-ds and (very) out-ds are
more clearly distinguishable for the Mahanalobis-based scorer that uses
the last layer, and the Cosine Projection scorer that uses the average
latent representation.

<div class="center">

<figure>
<img src="output2.png" id="table:figure_distrib" alt="Scorers distribution on datasets (in_ds[SST2],out_ds[IMDB],very_out[RTE])" /><figcaption aria-hidden="true">Scorers distribution on datasets (in_ds[SST2],out_ds[IMDB],very_out[RTE])</figcaption>
</figure>

</div>

We can draw the same observations if we look at the AUROC given in the
Figure
<a href="#figure:figure_auroc" data-reference-type="ref" data-reference="figure:figure_auroc">2</a>.
Again, the AUROC is higher for the Mahanalobis-based scorer that uses
the last layer, and the Cosine Projection scorer that uses the average
latent representation than for the other scores.

<div class="center">

<figure>
<img src="output3.png" id="figure:figure_auroc" alt="AUROC for each scorer." /><figcaption aria-hidden="true">AUROC for each scorer.</figcaption>
</figure>

</div>

# Discussion/Conclusion

## Computational issues

To do this work we encountered several computational issues that impeded
our progress. Unfortunately, these issues caused delays in the project
timeline and impacted the scope of our experimentation.

## Elements to be further explored 

With more time during this project, we would have liked to bring the
following experiments:

-   It would have been interesting to test other detectors, especially
    TRUSTED introduced by .

-   Also, as mentionned in , it is important to test the different
    scorers on various models to fully evaluate their performance. We
    would have liked to carry out our experiments on the DISTILBERT
    model.

## Pairing datasets as IN/OUT

The unsatisfying results obtained on last layer scorers might come from
the pairing of SST2/IMDB as these are close semantic datasets that
requires high-end OOD detectors using information of many hidden-layers
and not only the last one.

## Conclusion

In conclusion, our experimentation focused on the task of
out-of-distribution (OOD) detection for text classification. We used the
SST2, IMDB and RTE datasets to evaluate the performance of different OOD
detection methods, namely Mahalanobis-based and Cosine Projection-based
scores. Our results showed that both Mahalanobis-based and Cosine
Projection-based scores are effective in OOD detection for text
classification. Specifically, the Mahalanobis-based score performed best
using only the last layer for the latent representation, and the Cosine
Projection-based score performed best using the average latent
representation. However, our supervised approach did not perform as well
as expected. Despite some computational issues, our findings suggest
that OOD detection methods are promising for text classification tasks
and warrant further investigation.

[^1]: <https://github.com/joevincentgaltie/OOD_Detection_ENSAE.git>

[^2]: <https://huggingface.co/datasets/sst2>

[^3]: <https://huggingface.co/datasets/imdb>

[^4]: <https://huggingface.co/datasets/SetFit/rte>


The repositories [Todd](https://github.com/icannos/Todd) and [ToddBenchmark](https://github.com/icannos/ToddBenchmark) enabled us to do this work. 

