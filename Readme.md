### Intro
The code is for the paper Anonymous Walk Embeddings (https://arxiv.org/abs/1805.11921) that creates embeddings for entire graphs, based on feature-based and data-driven approaches, and evaluates the results on graph classification task.

### Tutorial

Check a notebook `Tutorial.ipynb` to see how to get network embeddings, load/save embeddings matrix, calculate kernel matrix, and perform SVM calculation.

### Running code

The following command will run data-driven model with the parameters.
```
python AnonymousWalkEmbeddings.py --dataset mutag --root ./Datasets --window_size 16 --batch_size 100 --batches_per_epoch 100 --num_samples 64 --steps 10 --results_folder mutag_results
```

The following command will run data-driven model with the parameters.
```
python AnonymousWalkKernel.py --dataset mutag --root ./Datasets --steps 10 --method sampling --MC 10000 --results_folder mutag_results
```

Alternatively, you can change default parameters in the file doc2vec.py.

Below is the list of all arguments:
- **dataset** Name of the dataset (and folders to save/load data)
- **batch_size** Number of (context, target) pairs per batch
- **window_size** Number of words in the context
- **embedding_size_w** Dimension of word embeddings
- **embedding_size_d** Dimensions of doc embeddings
- **num_samples** number of (negative) samples for every target word.
- **concat** Concatenate *or* Average embeddings of words.
- **loss_type** Sampled softmax loss *or* NCE loss
- **optimize** Adagrad *or* SGD
- **learning_rate** Learning rate
- **root** Location of dataset
- **ext** Extension of the graph files
- **steps** Number of steps in a random walk.
- **epochs** Number of global iterations.
- **batches_per_epoch** number of batches per epoch for each graph
- **candidate_func** None (loguniform by default) or uniform
- **graph_labels** None, edges, nodes, edges_nodes
- **method** Sampling or Exact.
- **MC** Number of Monte-Carlo iterations.
- **results_folder** Folder for storing results.

#### Running with Docker
To run the code on docker, we first need to obtain an appropriate image. After installing [Docker](https://docs.docker.com/engine/installation/) on your computer, you first [pull](https://docs.docker.com/docker-hub/repos/) an image `gcr.io/tensorflow/tensorflow` for Tensorflow:
```
docker pull gcr.io/tensorflow/tensorflow
```

After that you need to build a Dockerfile. You should have in the current directory Dockerfile. To avoid compression of all the files under current directory, we recommend to create a separate folder for Dockerfile and navigate there.
```
cd docker
docker build -t awe-docker .
```

You can test available images on your computer with `docker images`. Once you have `awe-docker` image you can run the code with.

```
docker run -i --user $(id -u):$(id -g) --mount type=bind,source=/home/sivanov/Datasets,target=/src/Datasets --mount type=bind,source=/home/sivanov/awe/,target=/src/awe/ --name='sergey.mutag.1' -t awe-docker:latest python /src/awe/AnonymousWalkEmbeddings.py --dataset mutag --root /src/Datasets/ --window_size 16 --batch_size 100 --batches_per_epoch 100 --num_samples 64 --results_folder /src/awe/mutag_results
```
Here, `--user $(id -u):$(id -g)` makes sure your results will be written with your user permission. `--mount type=bind,source=/home/sivanov/Datasets,target=/src/Datasets` mounts your host directory with datasets into docker image under /src/Datasets directory. `--mount type=bind,source=/home/sivanov/awe/,target=/src/awe/` mounts your host directory /home/sivanov/awe/ with scripts into an image directory /src/awe/. `--name='sergey.mutag.1'` sets the name of container. `-t awe-docker:latest` provides a name of the image. 

After running, you should get results in your host machine at awe/mutag_results directory.

### Citation
If you use the code, please consider citing our work.
```
@InProceedings{pmlr-v80-ivanov18a,
  title = 	 {Anonymous Walk Embeddings},
  author = 	 {Ivanov, Sergey and Burnaev, Evgeny},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {2191--2200},
  year = 	 {2018},
  editor = 	 {Dy, Jennifer and Krause, Andreas},
  volume = 	 {80},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Stockholmsmässan, Stockholm Sweden},
  month = 	 {10--15 Jul},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v80/ivanov18a/ivanov18a.pdf},
  url = 	 {http://proceedings.mlr.press/v80/ivanov18a.html}
  }
```
