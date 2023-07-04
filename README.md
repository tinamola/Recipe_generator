# FIT3144 NLP Project (Recipe Generator)
Have some ingredients, but no idea how to cook with them? We can train a deep neural network to do the work for us!

As part of this research project, I am building a Sequence-to-Sequence(Seq2Seq)+attention+pgen+coverage model for generating recipes from ingredients inputs.

## Critical Dependencies & Acknowledgements
*  I. Sutskever, O. Vinyals, and Q. V. Le, [“Sequence to sequence learningwith  neural  networks,”  inAdvances in Neural Information ProcessingSystems(Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Q.Weinberger, eds.), vol. 27, Curran Associates, Inc., 2014](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks)
*  C. Kiddon, L. Zettlemoyer, and Y. Choi, [“Globally coherent text genera-tion with neural checklist models,” inProceedings of the 2016 Conferenceon Empirical Methods in Natural Language Processing, (Austin, Texas),pp. 329–339, Association for Computational Linguistics, Nov. 2016.](https://homes.cs.washington.edu/~yejin/Papers/emnlp16_neuralchecklist.pdf)
*  A. See, P. J. Liu, and C. D. Manning, “Get to the point:  Summarizationwith pointer-generator networks,”CoRR, vol. abs/1704.04368, 2017 [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
* The  pytorch code framework of seq2seq+attention+pgen+coverage can be found at <https://github.com/Dai-Wenxun/Pointer-Generator-Networks>  
* Many thanks to [Prof. Ehsan Shareghi](https://eehsan.github.io/) for supervising the research and tesing stages of this project, and for his suggestion of using coverage mechanism to reduce repeatedly generated words.    

## Datasets
* I use the [Now You're Cooking Dataset](https://drive.google.com/file/d/1qyiBz1kMqkcvIgVlm1go6x3WgzlWhe0F/view) dataset in this project. The ingredients are the input and the recipe is treated as the ground truth for training and testing our models. I filtered data with ingredient length greater than 80 words and recipe length greater than 170 to reduce training time. The vocabulary is restricted to 20000, both for avoiding resource exhaustion issues and for checking how the pointer-generator mechanism work when our model encounter [UNK]s in the input data. After the preprocessing, the dataset contains 131191 training samples, 12800 validation samples and 993 testing samples. For simplicity, ingredients were cleaned of non-word tokens and stripped of amounts(eg. 1 tsp), meanwhile in future it is better to collapse multi-word ingredient names into single tokens, as suggested by [Globally Coherent Text Generation with Neural Checklist Models](https://homes.cs.washington.edu/~yejin/Papers/emnlp16_neuralchecklist.pdf).

## Software Tools
* I trained the Seq2Seq + attention model in Tensorflow and compared the validation loss between separate embeddings, global embeddings, and the model with no attention.
* I used Pytorch to evaluate the Seq2Seq + att + pgen + coverage model on its validation loss as compared to the Seq2Seq + att + coverage model.

## Tutorials & Articles
* A great source of tensorflow tutorials can be found at https://www.tensorflow.org/tutorials, featuring basic word2vec, rnn, seq2seq, and bert models.
* In addition, for learning NLP problems and the principles behind the structure of each model, useful videos can be found at <https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/>. 
