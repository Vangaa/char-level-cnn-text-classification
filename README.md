# Char-level cnn short-text classifier
Short-text classifier based on cnn and char level representations of words.
Trained and tested on russian twitter corpus http://study.mokoron.com/.

Inspired by [Denny Britz blogpost](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).

## Requirements

- Python 3
- Tensorflow > 1.0
- Numpy

Print parameters:

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --test_perc TEST_PERC
                        Percentage of the training data to use for validation
  --pos_data_file POS_DATA_FILE
                        Data source for the positive data.
  --neg_data_file NEG_DATA_FILE
                        Data source for the negative data.
  --words_take WORDS_TAKE
                        Comma-separated filter sizes (default: '2,3,4')
  --learning_rate LEARNING_RATE
                        Learning rate (default: 1e-4)
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 256)
  --dropout DROPOUT     Dropout keep probability (default: 0.2)
  --checkpoint_dir CHECKPOINT_DIR
                        Checkpoints dir (default: models)
  --batch_size BATCH_SIZE
                        Batch Size (default: 128)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 10)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
```
