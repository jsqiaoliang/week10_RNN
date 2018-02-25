# Report of Week 10 homework

## Word Embedding

Based on the supported code I made the following changes to make the word embedding for QuanSongCi

~~~python
file = open('QuanSongCi.txt',encoding="utf8")
dataTotal = file.read()
chars = list(set(dataTotal))
data_size, num_classes = len(dataTotal), len(chars)
words = chars
~~~

 This will read the data from QuanSongCi.txt and make each of the word inside the text file unique and determine the number of class in the text as well.

As Chinese 1 char is equal to 1 word, therefore words = chars in the end.

~~~~python
vocabulary_size = 5000
~~~~

In this task we set and allow only 5000 different words even there are actually more than 6000 unique words in the text. So words that is not common will be put into the category of UNK through the build_dataset function;

~~~~python
batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
~~~~

These parameter will determine the details of word embedding. Embedding_size of 128 means that each word will have a representation in 128 dimensions. In the space of 128 dimension, the word will have it's own position.

```python
data: ['搊', '坚', '彷', '駈', '谪', '骍', '屼', '咻']

with num_skips = 2 and skip_window = 1:
    batch: ['坚', '坚', '彷', '彷', '駈', '駈', '谪', '谪']
    labels: ['彷', '搊', '坚', '駈', '彷', '谪', '骍', '駈']

with num_skips = 4 and skip_window = 2:
    batch: ['彷', '彷', '彷', '彷', '駈', '駈', '駈', '駈']
    labels: ['谪', '駈', '坚', '搊', '彷', '谪', '坚', '骍']
```

This explained the meaning of  num_skips and skip_window.  num_skips determines how many times to reuse an input to generate a label. The skip_window determines how many words to consider left and right.

~~~~python
from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))
except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
~~~~

In the end, use TSNE method to lower the dimension from 128 to 2. So that we can see the distribution of words on X and Y coordinate. The similar related words are placed at the similar position of the graph as follow:



The Details of code are shown in Word Embedding.ipynb of the repository



## RNN-LSTM

## 