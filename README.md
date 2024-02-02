# QPA

## Parameter introduction

We have added two new parameters based on GPTQ (https://github.com/IST-DASLab/gptq):

1. --sparsity: Specifies the percentage of parameters to be retained after pruning. The default value is set to 1, indicating no pruning.
2. --is_layered：Determines whether to merge sparse and quantized matrices into a single matrix.

## Environment

```
pip install -r requirements.txt
```

## Files

- result :Recorded the compression results of OPT, BLOOM, and LLaMA models
- Zero-shot：Test zero-shot accuracy
- bloom.py : Compressing BlOOM family using QPA algorithm
- opt.py: Compressing OPTfamily using QPA algorithm 
- llama.py: Compressing llama family using QPA algorithm
- datautils.py：Load dataset
- gptq.py ：Extend the GPTQ algorithm to the QPA algorithm
- modelutils.py: Initialize the model
- quant.py: Calculate quantization parameters

## Running QPA & measuring the perplexity (PPL)

- **OPT**

  ```
  # Compute full precision (FP16) results
  CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4
  
  #2-bit quantization + 99% sparsity
  CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 2 --new-eval --sparsity 1
  
  #3-bit quantization + 99% sparsity
  CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 3 --new-eval --sparsity 1
  
  #4-bit quantization + 99% sparsity
  CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --new-eval --sparsity 1
  ```

  To run other OPT models replace `opt-125m` with one of: `opt-350m`, `opt-1.3b`, `opt-2.7b`, `opt-6.7b`, `opt-13b`, `opt-66b`.

- **BLOOM**

  ```
  #2-bit quantization + 99% sparsity
  CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 2 --new-eval --sparsity 1
  ```

  To run other BLOOM models replace `bloom-560m` with one of: `bloom-1b1`, `bloom-1b7`, `bloom-3b`, `bloom-7b1`

- **LLaMA**

  ```
  #2-bit quantization + 99% sparsity
  CUDA_VISIBLE_DEVICES=0 python llama.py "model path" c4 --wbits 2 --new-eval --sparsity 1 --true-sequential --act-order 
  ```


# zero-shot

First, quantize the model and then save the model path：

```
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 2 --new-eval --save output/model.pth
```

second, measuring zero-shot tasks：

```
CUDA_VISIBLE_DEVICES=0 python zeroshot.py \
facebook/opt-125m \
--load output/model.pth \
--batch_size 8 \
--task llambada-openai
```

