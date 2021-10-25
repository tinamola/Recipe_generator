# FIT3144 NLP Project (Recipe Generator)
Having some ingredients, but wondering how to make a good meal out of it? We can train a (deep) neural model to do the job for us!

I'm a part of this research project that is developing a Sequence-to-Sequence(Seq2seq)+att+pgen+coverage model, built on top of lstm. 

## Critical Dependencies & Acknowledgements
*  I. Sutskever, O. Vinyals, and Q. V. Le, “Sequence to sequence learningwith  neural  networks,”  inAdvances in Neural Information ProcessingSystems(Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Q.Weinberger, eds.), vol. 27, Curran Associates, Inc., 2014
*  T. Linzen, E. Dupoux, and Y. Goldberg, “Assessing the ability of lstms tolearn syntax-sensitive dependencies,”CoRR, vol. abs/1611.01368, 2016.
*  C. Kiddon, L. Zettlemoyer, and Y. Choi, “Globally coherent text genera-tion with neural checklist models,” inProceedings of the 2016 Conferenceon Empirical Methods in Natural Language Processing, (Austin, Texas),pp. 329–339, Association for Computational Linguistics, Nov. 2016.
*  A. See, P. J. Liu, and C. D. Manning, “Get to the point:  Summarizationwith pointer-generator networks,”CoRR, vol. abs/1704.04368, 2017 [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
* The code framework is based on <https://github.com/Dai-Wenxun/Pointer-Generator-Networks>  
* We thank [Prof. Ehsan Shareghi](https://eehsan.github.io/), for supervising the research and experiment stages of this project, as well as suggestions of using coverage mechanism to reduce repeatedly generated words.    

## Datasets
* I use the [Now You're Cooking Dataset](https://drive.google.com/file/d/1qyiBz1kMqkcvIgVlm1go6x3WgzlWhe0F/view) dataset in this project. The ingredients are the input and the recipe is treated as the ground truth for training and testing our models. I filtered data with ingredient length greater than 80 words and recipe length greater than 170 to reduce training time. The vocabulary is restricted to 20000, both for avoiding resource exhaustion issues and for checking how the pointer-generator mechanism work when our model encounter [UNK]s in the input data. After the preprocessing, the dataset contains 131191 training samples, 12800 validation samples and 993 testing samples. For simplicity, I removed the unit of measurement in ingredients during preprocessing, but in future it is better adding them back for generating more accurate results.

## Software Tools
* I used Tensorflow to train the Seq2Seq + attention model, and compared the results generated from models using seperate/global embeddings, and the results with no attention.
* I used Pytorch to train the Seq2Seq + attention model incorperated with coverage mechanism, with comparison on the model with and without the pointer generator mechanism. 

## Tutorials & Articles
* The <https://www.tensorflow.org/tutorials> website is a great resource that showcases basic tensorflow implementation of word2vec, rnn, seq2seq and bert model.
* The Youtube videos from <https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/> is a good starting point for learning NLP problems and principles behind the structure of each model.

## Result
