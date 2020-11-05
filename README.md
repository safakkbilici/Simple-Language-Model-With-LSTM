# Simple Language Model With LSTM
## Usage
### Train

```console
python train.py -d [datapath] -e [number of epochs]
```

### Generate

```console
python generate.py -d [datapath] -t [new file name] -n [number of generated words]
```

### For More Detailed Argument Explanation 

```console
python train.py --help
python generate.py --help
```

## Example

```console
python train.py -d ./data/bible.txt -e 40
```

```console
python generate.py -d ./data/bible.txt -t ./generated/biblev2.txt -n 500
```

## Model

The summary of the model is:

    Author(
      (embed): Embedding(33443, 128)
      (lstm): LSTM(128, 1024, batch_first=True)
      (linear): Linear(in_features=1024, out_features=33443, bias=True)
    )
    --------------------------------------------------------------------
    Total number of trainable parameters: 43286563

