### Architecture deep dive:

Note:

Original paper has layer normalization after attention step, however a 2020 paper(arXiv:2002.04745) showed that normalization before attention stabilizes training, removing need for learning rate warmup. That change is represented here. In the original step they do "Laver Norm( $X+Z(X))^{\\\\prime\\\\prime}$ the found-to-be-better approach does $"X+Z(LayerNorm(X))^{\\\\prime\\\\prime}.$



1\.  Tokenizing: Inputs must be tokenized into numerical representations of words

    a. A token can represent a character or a short segment of characters



2\.  Embedding: Tokens are then embedded based on a one hot encoding of the token times an embedding matrix



3\.  Positional Encoding: Embedding vectors are then mapped to a matrix by taking sin and cosine representations of the position to create a unique probability frequency representing the location of the embedding vector in the input.

    a. Alternates matrix frequencies by sin and cos to ensure uniqueness other position 1 and 6 in a space of 5 would have the same frequency

    b. Takes a 1xL embedding vector and creates a dxl, space where L is the length of the input and d is a determined positive even integer parameter

    i. d is a tunable parameter and often is the same as the word embedding dimension (for easier math), often 512, 768, or 1024 by your embedding function

    c. N is a parameter in positional encoding which should be significantly larger than the longest input you would expect.

    d. By using matrixes, we allow for computationally simple matrix multiplication to go from one encoded position to another



4\.  Encoder block: The matrix containing the positional encoding of the embedding vectors is the passed into the encoding block

    a. Layer normalization

    i. \[https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)

    ii. In order to reduce training time and stabilize training we do a layer normalization on the original input matrix

    1\\) Layer Norm(X)

    2\\) Layer norm is just the z-score normalization with a small constant added to variance for numerical stability

    b. Multi head attention

    i. Links I used to understand

    1\\) \[https://www.reddit.com/r/MachineLearning/comments/qidpqx/d\\\_how\\\_to\\\_truly\\\_understand\\\_attention\\\_mechanism\\\_in/](https://www.reddit.com/r/MachineLearning/comments/qidpqx/d\_how\_to\_truly\_understand\_attention\_mechanism\_in/)

    2\\) \[https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

    ii. Self-Attention mechanism intuition

    1\\) To find the relative importance of each word in comparison to one word in a sentence (toy example)

    a) We calculate the vector multiplication of word 1(q,1) across word 2 (through n, k\\\_1 to k\\\_n) to get a score

    i) $(q1\*k1)$ for k(1 to dk)

    b) We divide the scores by the square root of the sentence length (key vector dimension), this leads to more stable gradients

    i) $(q1\*k1)/sqrt(d\\\_k)$ for k(1 to dk)

    c) We then take the SoftMax of each $(q1\*k1)/sqrt(d\\\_k)$, this gives us the relative importance of each word

    i) SoftMax(list( $(q1\*k1)/sqrt(d\\\_k)$ for k(1 to dk))) for k(1 to dk)

    d) We then multiply each SoftMax value (relative importance of each word to first word) to our value vector (representation of original word vector), this maintains the representation of the words we want to focus on and minimizes the presence of the insignificant words

    i) $v1\*softmax\\\_{v1}(list((q1\*k1)/sqrt(d\\\_k)$ for k(1 to dk))) for k(1 to dk)

    e) We then sum up our weighted value vectors, this produces the output of the self-attention layer, a vector representing the importance of each word in the sentence relative to a single word

    i) $z1 = \\\\text{Sum}(v1 \\\* \\\\text{Softmax}\\\_{v1}(\\\\text{list}((q1 \\\\cdot k1)/\\\\sqrt{d\\\_k})) \\\\text{ for } k(1 \\\\text{ to } d\\\_k))$

    2\\) In practice, this is done via matrixes for faster computations.

    a) Little fundamentally changes, as instead of individual z1 vectors we have a matrix z holding all z1 vectors.

    iii. Multi-headed Self Attention mechanism intuition

    1\\) Multi-headed attention provides the model with a way to consider multiple positions at once, since often z1 is dominated by itself, allowing for multiple representation subspaces. Basically here's one opinion of the importance of each word, here's a second, and a third, and so on. When looking at a painting, we as humans each notice something unique, the multiple attention heads do the same.

    2\\) This works by computing 'z' n times, in the original paper, it was 8. We randomly initiate our W\\\_Q, W\\\_K, W\\\_V matrixes to allow for different gradients. When we have our 8 'Z' matrixes we then concatenate our matrixes and project them back to original dimension of Z. This is for the feed forward matrix which is expecting one matrix of dimensions Z

    a) Does so by concatenating our Z\\\_0 to Z\\\_n matrixes then multiplying them by a W\\\_O projection matrix to reduce it back to a single Z matrix rowsxcolumns

    b) This sums our attention heads 'opinions of relevance' to one opinion

    c. Add in original input

    i. The addition of the original input has been found to avoid vanishing gradient issues and stabilize the training process. So here we add in the normalized positional encoded embedding matrix to our attention output.

    1\\) X+Z(normalized X)

    2\\) (to add our opinion of relevance to our positional encoded input embedding)

    d. Layer normalization

    e. Feed forward

    i. \[https://aclanthology.org/2021.emnlp-main.446.pdf](https://aclanthology.org/2021.emnlp-main.446.pdf)

    ii. Intuitively this takes in inputted numerical representation and maps it to a latent representation space then the FNN acts as a key-value pair to produce a probability distribution over the vocabulary from the input.

    iii. The feedforward network is a 2-layered multilayer perceptron module of weights and bias vectors along an activation function $FFN(x)=\\\\sigma(xW1+b1)W2+b2$. The activation function originally was a ReLU activation function

    iv. The number of neurons in the middle layer is typically larger (\\~4 times) than the embedding size and is referred to as the intermediate or filter feedforward size.

    f. Add in un-normalized data (post attention but pre second normalization)

    g. Into next encoder layer or decoder

    h. One last layer normalization



5\.  Decoder block:

    a. Output embedding input

    i. The first encoder is either nothing or the previously outputted token, subsequent layers take the previous decoders output

    b. Normalize input

    c. Masked Multi-head Self Attention

    i. Same as other multi-head attention operation except here we set all future position values to be -inf, masking the values from consideration

    ii. This intuitively considers what tokens are most relevant given what it has seen before

    d. Add original un-normalized input to attention output

    e. Layer normalization

    f. Encoder decoder Multi-head Cross Attention

    i. Here we initialize our K and V matrixes with the output of the encoders from before, Q is the output of the previous attention mechanism

    ii. This draws relevant information from the encodings generated by the encoders

    iii. Masking is not needed here as it attends to vectors computed before the decoder started decoding.

    iv. This intuitively considers which encoded latent knowledge spaces are most relevant to our normalized 'opinion of relevance' encoding

    g. Add in un-normalized data (post attention but pre second normalization)

    h. Layer Normalization

    i. Feed Forward

    j. Add in un-normalized data (post cross attention but pre third normalization)

    k. Into next decoder or linear SoftMax output



6\.  The output is un-embedded through a linear soft-max layer. Which turns the embedded vector into a probability vector over tokens.

    a. Linear layer

    i. A simple fully connected neural network that projects the decoder output into a much larger logits vector

    ii. Intuitively this projects the output into a vector representing our entire vocabulary with an associated logit for each output

    b. SoftMax layer

    i. SoftMax takes this logit vector and turns it into a probability vector adding up to 1

    ii. We then take the value with the highest probability and return the corresponding token (token since it could be a word or a punctuation)



### How it is trained:

Usually first pre-trained on a large generic dataset using self-supervising learning. Then fine-tuned on a small task-specific dataset.



Example loss functions are:



  - Masked task: Correctly guessing a masked word in an input

  - Autoregressive task: Entire sequence is masked, model produces an output for first token, output is revealed and error computed, does second token and so on

  - prefixML task: Same as above but first half of sequence is immediately available.

  - Translations between different languages, contexts, or modalities

  - Judging linguistic and pragmatic acceptability of the language



### All together:

Takes in an input and converts it to a vector, calculates the position of each token with respect to the entire input and adds the representation to the vector, then stores all the vectors as a matrix and passes it into the encoder or decoder.



The encoder takes the matrix, normalizes it, and finds the contextual relevance of each word with respect to each word; then adding this contextual importance back to the original matrix. The matrix is then normalized again and passed to a FFN which acts as key-value lookup of the matrix to the latent space returning a probability distribution over the vocabulary, which when added to our input begins the process of slowly converging on a result.



The decoder takes the input matrix representation, normalizes it, and finds the future-location-masked contextual relevance of each word with respect to each word; then adding this contextual importance back to the original matrix. The matrix is then normalized again and passed to a cross-attention system which takes in the learned encoder key and value matrixes to enrich our input with contextually relevant "information" from our latent knowledge space, this context is added back to our input. This matrix is then normalized again and passed to a FFN which acts as key-value lookup of the matrix to the latent space returning a probability distribution over the vocabulary, which when added to our input begins the process of slowly converging on a result.



Many layers later, the final matrix undergoes a linear transformation to logit representations across the vocabulary, then SoftMax returns probabilities of the logits. We then return the highest probability token.



The encoder process is repeated an incredible number of times minimizing various loss functions to create an incredibly rich and knowledgeable "brain" or latent representation of information that the decoder can pull from after training.



The decoder process is repeated until a terminal token is highest probability, whether through the models belief or via the models system prompt instructions.



Alternate implementation mechanics:

\[https://en.wikipedia.org/wiki/Transformer\\\_(deep\\\_learning\\\_architecture)\\#Subsequent\\\_work](https://www.google.com/search?q=https://en.wikipedia.org/wiki/Transformer\_\\(deep\_learning\_architecture\\)%23Subsequent\_work)



Different models use different:



  - Activation functions

  - Normalization algorithms

  - Normalization locations

  - Positional encodings

  - Implementation efficiencies

  - Alternate attention graphs

  - Multimodal approaches



### Links:



  - \[https://peterbloem.nl/blog/transformers](https://peterbloem.nl/blog/transformers)

  - \[https://en.wikipedia.org/wiki/Transformer\\\_(deep\\\_learning\\\_architecture)\\#Subsequent\\\_work](https://www.google.com/search?q=https://en.wikipedia.org/wiki/Transformer\_\\(deep\_learning\_architecture\\)%23Subsequent\_work)

  - \[https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

  - \[https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)

  - \[https://aclanthology.org/2021.emnlp-main.446.pdf](https://aclanthology.org/2021.emnlp-main.446.pdf)

  - \[https://www.reddit.com/r/MachineLearning/comments/qidpqx/d\\\_how\\\_to\\\_truly\\\_understand\\\_attention\\\_mechanism\\\_in/](https://www.reddit.com/r/MachineLearning/comments/qidpqx/d\_how\_to\_truly\_understand\_attention\_mechanism\_in/)

