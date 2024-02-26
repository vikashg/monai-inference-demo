from model_def import ModelDefinition
import torch

model_def =ModelDefinition(model_name="SegResNet")
model = model_def.get_model()
model_fn = './best_metric_model.pth'
model.load_state_dict(torch.load(model_fn))


x = torch.zeros(1, 1, 256, 256, 24)
traced_model = torch.jit.trace(model, x)
traced_model.save('./traced_segres_model.pt')

