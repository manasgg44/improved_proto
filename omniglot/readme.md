The Prototypical Network model is meta-trained and saved in the logs folder. One may use the same pre-trained model for meta-testing to save time.
The Omniglot dataset needs to be downloaded and the folder be named as data. To download the dataset or to retrain the model one may run :
  
    python train.py --folder "../data" --output-folder "logs" --num-steps 5 --num-shots 5 --num-ways 5 --num-batches 60000 --use-cuda --download

For meta-test using our approach, run the following :

    python test_our.py --folder "../data" --output-folder "logs" --num-steps 5 --num-shots 5 --num-ways 5 --num-batches 100 --use-cuda
