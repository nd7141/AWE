# Running code

The following command will run `doc2vec` model with the parameters.
```
python AnonymousWalkEmbeddings.py --dataset mutag --root ./ --window_size 16 --batch_size 100 --batches_per_epoch 100 --num_samples 64 --results_folder mutag_results
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
- **results_folder** Folder for storing results.

#### Running with Docker
To run the code on docker, we first need to obtain an appropriate image. After installing [Docker](https://docs.docker.com/engine/installation/) on your computer, you first [pull](https://docs.docker.com/docker-hub/repos/) an image `gcr.io/tensorflow/tensorflow` for Tensorflow:
```
docker pull gcr.io/tensorflow/tensorflow
```

After that you need to build a Dockerfile. 
```
docker build -t awe-docker .
```

You can test available images on your computer with `docker images`. Once you have `awe-docker` image you can run the code with.

```
docker run -i --user $(id -u):$(id -g) --mount type=bind,source=/home/sivanov/Datasets,target=/src/Datasets --mount type=bind,source=/home/sivanov/awe/,target=/src/awe/ --name='sergey.mutag.1' -t awe-docker:latest python /src/awe/AnonymousWalkEmbeddings.py --dataset mutag --root /src/Datasets/ --window_size 16 --batch_size 100 --batches_per_epoch 100 --num_samples 64 --results_folder /src/awe/mutag_results
```
Here, `--user $(id -u):$(id -g)` makes sure your results will be written with your user permission. `--mount type=bind,source=/home/sivanov/Datasets,target=/src/Datasets` mounts your host directory with datasets into docker image under /src/Datasets directory. `--mount type=bind,source=/home/sivanov/awe/,target=/src/awe/` mounts your host directory /home/sivanov/awe/ with scripts into an image directory /src/awe/. `--name='sergey.mutag.1'` sets the name of container. `-t awe-docker:latest` provides a name of the image. 

After running, you should get results in your host machine at awe/mutag_results directory. 
