# QPA

## Parameter introduction

We have added two new parameters based on GPTQ:

1. --sparsity: Specifies the percentage of parameters to be retained after pruning. The default value is set to 1, indicating no pruning.
2. --is_layered：Determines whether to merge sparse and quantized matrices into a single matrix.

## Environment

```
pip install -r requirements.txt
```

## Running QPA & measuring the perplexity (PPL)

- OPT

  ```
  #2-bit quantization + 99%sparsity
  CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 2 --new-eval --sparsity 1
  ```

  

- BLOOM

  ```
  #2-bit quantization + 99%sparsity
  CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 2 --new-eval --sparsity 1
  ```

  

- LLaMA

  ```
  #2-bit quantization + 99%sparsity
  CUDA_VISIBLE_DEVICES=0 python llama.py "model path" c4 --wbits 2 --new-eval --sparsity 1 --true-sequential --act-order 
  ```

  

# zero-shot

First, quantize the model and then save the model path：

```
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 2 --new-eval --save output/model.pth
```

Measuring zero-shot tasks：

```
CUDA_VISIBLE_DEVICES=0 python zeroshot.py \
facebook/opt-125m \
--load output/model.pth \
--batch_size 8 \
--task llambada-openai
```

