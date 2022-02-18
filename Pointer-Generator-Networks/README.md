## About this repository  

This repo contains an Pytorch implementation for the ACL 2017 paper *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*. The code framework is based on [TextBox](https://github.com/RUCAIBox/TextBox).

---

## Environment 

- `python >= 3.8.11` 
- `torch >= 1.6.0`

Run `install.sh` to install other requirements.

## Paramters

```yaml
# overall settings
data_path: 'data/'
checkpoint_dir: 'saved/'
generated_text_dir: 'generated/'
# dataset settings
max_vocab_size: 50000
src_len: 400
tgt_len: 100

# model settngs
decoding_strategy: 'beam_search'
beam_size: 4
is_attention: True
is_pgen: True
is_coverage: True
cov_loss_lambda: 1.0
```
Log file is located in `./log`, more details can be found in [yamls](./yamls).

**Note**: Distributed Data Parallel (DDP) is not supported yet.


## Train & Evaluation

### From scratch run `fire.py`.
```python 
if __name__ == '__main__':
    config = Config(config_dict={'test_only': False,
                                 'load_experiment': None})
    train(config)
```

If you want to resume from a checkpoint, just set the `'load_experiment': './saved/$model_name$.pth'`. Similarly, when `'test_only'` is set to `True`, `'load_experiment'` is required.

