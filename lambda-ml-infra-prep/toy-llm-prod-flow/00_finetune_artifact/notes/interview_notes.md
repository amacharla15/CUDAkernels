# Fine-Tuning Interview Notes

## Fine-tuning vs serving

Fine-tuning creates a model artifact.
Serving loads that artifact and exposes an API.

Fine-tuning output can be:
1. A full fine-tuned model folder.
2. A small LoRA adapter folder.
3. A merged model folder after merging LoRA into the base model.

vLLM does not fine-tune the model.
vLLM serves the model.

## LoRA

Normal linear layer:

y = xW

LoRA version:

y = xW + scaling * xBA

W is the frozen base model weight.
A and B are small trainable adapter matrices.

Physical meaning:
- Frozen base model weights live in GPU memory.
- LoRA adapter weights live in GPU memory.
- Gradients are stored only for LoRA adapter weights.
- Optimizer states are stored only for LoRA adapter weights.

Why it matters:
LoRA reduces training memory because the full model is not updated.

Interview answer:
LoRA freezes the original model and trains small low-rank adapter matrices. In the forward pass, the model computes the original xW output and adds a learned low-rank update. This makes fine-tuning cheaper because gradients and optimizer states are only needed for the adapter weights.

## QLoRA

QLoRA = quantized base model + LoRA adapter training.

The base model is loaded in 4-bit.
The LoRA adapter remains trainable in higher precision.

Forward pass:
1. Read quantized frozen base weight.
2. Dequantize enough for computation.
3. Compute base output.
4. Compute LoRA adapter update.
5. Add base output and LoRA update.

Why it matters:
QLoRA saves memory because the large base model is stored in 4-bit while only small adapter weights are trained.

Interview answer:
QLoRA is memory-efficient LoRA. It keeps the frozen base model quantized, usually 4-bit, and trains small LoRA adapters in higher precision. This allows fine-tuning larger models on smaller GPUs.

## Knowledge distillation

Teacher model:
Large model with better output distribution.

Student model:
Smaller model being trained.

Loss:

final_loss = alpha * CE(student_logits, true_labels)
           + (1 - alpha) * KL(student_distribution, teacher_distribution)

Cross-entropy teaches the correct label.
KL divergence teaches the student to imitate the teacher's probability distribution.

Temperature softens the teacher distribution.

Interview answer:
In distillation, the student learns from both the ground truth and the teacher model. The teacher's soft probability distribution gives more information than a single hard label because it shows which wrong answers are still relatively plausible.

## Lambda relevance

For Lambda field engineering, fine-tuning itself is not the main job.
The important production connection is:

fine-tuned artifact -> serving engine -> benchmark -> debug -> optimize

A strong answer:
I understand fine-tuning as artifact creation. LoRA or QLoRA produces an adapter artifact, while full fine-tuning produces a full model artifact. Serving systems like vLLM load those artifacts for inference. Then the field-engineering work is benchmarking TTFT, TPOT, throughput, GPU memory, concurrency, and diagnosing performance problems.
