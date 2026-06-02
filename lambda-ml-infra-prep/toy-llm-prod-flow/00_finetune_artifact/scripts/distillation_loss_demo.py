import torch
import torch.nn.functional as F

def main():
    student_logits = torch.tensor([[2.0, 1.0, 0.1, -0.5]])
    teacher_logits = torch.tensor([[3.0, 1.5, 0.5, -1.0]])
    true_label = torch.tensor([0])

    temperature = 2.0
    alpha = 0.5

    hard_label_loss = F.cross_entropy(student_logits, true_label)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    distillation_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    ) * (temperature * temperature)

    final_loss = alpha * hard_label_loss + (1.0 - alpha) * distillation_loss

    print("student_logits:", student_logits)
    print("teacher_logits:", teacher_logits)
    print("true_label:", true_label.item())
    print("temperature:", temperature)
    print("alpha:", alpha)
    print("hard_label_loss:", hard_label_loss.item())
    print("distillation_loss:", distillation_loss.item())
    print("final_loss:", final_loss.item())

    print()
    print("Meaning:")
    print("Hard-label loss teaches the student the correct answer.")
    print("Distillation loss teaches the student to imitate the teacher distribution.")
    print("Temperature softens the distributions so the student learns relative preferences, not only the top class.")

if __name__ == "__main__":
    main()
