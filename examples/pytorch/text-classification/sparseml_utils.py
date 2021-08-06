from typing import Any

import numpy
import torch
import torch.nn.functional as F

from sparseml.pytorch.utils import ModuleExporter

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.sparse import SparseMLTrainer


class SparseMLGLUETrainer(SparseMLTrainer):
    """
    GLUE trainer with SparseML integration

    :param recipe: recipe for model sparsification
    :param teacher: teacher model for distillation
    :param distill_hardness: ratio of loss by teacher targets (between 0 and 1)
    :param distill_temperature: temperature for distillation
    :param args, kwargs: arguments passed into parent class
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computing loss using teacher/student distillation
        """
        if not self.recipes or self.teacher is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        outputs = model(**inputs)
        if self.teacher is None:
            loss = outputs["loss"]
        else:
            input_device = inputs["input_ids"].device
            label_loss = outputs["loss"]
            self.teacher = self.teacher.to(input_device)
            logits_student = outputs["logits"]
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=inputs["input_ids"],
                    token_type_ids=inputs["token_type_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            logits_teacher = teacher_outputs["logits"]
            teacher_loss = F.kl_div( input=logits_student, target=logits_teacher, reduction="batchmean",) * (self.distill_temperature ** 2)
            loss = ((1 - self.distill_hardness) * label_loss) + (self.distill_hardness * teacher_loss)
        return (loss, outputs) if return_outputs else loss


class GLUEModuleExporter(ModuleExporter):
    """
    Module exporter class for Sequence Classification
    """

    @classmethod
    def get_output_names(self, out: Any):
        if not isinstance(out, SequenceClassifierOutput):
            raise ValueError("Expected SequenceClassifierOutput, got {type(out)}")
        expected = ["logits"]
        if numpy.any([name for name in expected if name not in out]):
            raise ValueError("Expected output names not found in model output")
        return expected