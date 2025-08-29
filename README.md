Setup:

```
uv pip install -e .
uv pip install flash-attn --no-build-isolation
wandb login {YOUR TOKEN}
huggingface-cli login --token {YOUR TOKEN}
```

To run SFT:

Multi-GPU (in this case, 4 GPUs, so --nproc_per_node=4)

`torchrun --nproc_per_node 4 train_demo/training.py`

Adjust hyperparameters in `train_demo/config.py`

This will train a model with TRL SFT to talk like a pirate. To test this out, run `inference_demo.py` in interactive mode. This lets you send prompts to both the original and trained model. You should see that the trained model talks like a pirate.

In response to: 

*"Alice's parents have three daughters: Amy, Jessy, and what’s the name of the third daughter?"*

We  get pirate responses like:

*"According to th' question, alice's parents have three daughters: Amy, jessy. Since ye have forgotten th' name of alice, th' third daughter be alice herself."*

Some notes: It takes me 20 minutes to train on 10k datapoints on Qwen3-14B with a 4x RTX 6000 Ada setup. To increase speed, you can experiment with disabling gradient checkpointing (will also require decreasing batch size) or turning on `packing=True` in the `SFTConfig`.

VLLM inference note: In the notebook, I have a bunch of commented environment flags to disable P2P. This is because I had previously had issues with VLLM tensor parallel on a runpod 4x GPU instance, which I fixed by disabling P2P. I tested on my current setup and it actually worked with no issues. If you get hangs, try enabling the P2P_DISABLE flag.

I'm not super sure what TRL is doing for tokenization. I see there are some tokenization warnings like this:

```
warnings.warn(
/root/trl_train_demo/.venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:630: UserWarning: Mismatch between tokenized prompt and the start of tokenized prompt+completion. This may be due to unexpected tokenizer behavior, whitespace issues, or special token handling. Verify that the tokenizer is processing text consistently.
```

But it appears to train correctly and I haven't investigated further. It may also be worth verifying that `completion_only_loss=True,` is doing the correct loss masking for your particular dataset.

If running into flash_attn install issues feel free to disable it in this line: `attn_implementation="flash_attention_2",`
