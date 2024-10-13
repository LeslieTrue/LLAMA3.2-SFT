# Fine-tuning Llama3.2-Vision

This repo is mostly borrowed from https://github.com/2U1/Llama3.2-Vision-Finetune except some small modifications.

## Launch

```
pip install -r requirements.txt
```

Remark: Not fully finished, plz prepare for yourself

```
bash scripts/finetune.sh
sbatch scripts/finetune.sh
```

Either script works. If you are using slurm server, plz use the second one.

## Setting

Current setting support text-only inputs and image-text pair input. It may not support data with mixed vision-text pairs and text-only inputs.

Per gpu batch size is set as 16 for A100/A800/H100/H800, which works well.

If you find it useful, give me a star and give https://github.com/2U1/Llama3.2-Vision-Finetune a star also! 