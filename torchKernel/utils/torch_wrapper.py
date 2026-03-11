import torch

# based on 
# https://github.com/educating-dip/educated_deep_image_prior/blob/103c52dfb53e98e381ae5c7cd775795be63abb21/src/dataset/walnuts.py#L532

class _objectiveFunctionModule3D(torch.autograd.Function):
    @staticmethod
    def forward( ctx, x, image_template, sirf_obj):
        ctx.device = x.device
        ctx.sirf_obj = sirf_obj
        ctx.image_template = image_template
        ctx.x = x.detach().cpu().numpy().squeeze()
        ctx.x = ctx.image_template.fill(ctx.x)
        value_np = ctx.sirf_obj.get_value(ctx.x)
        return torch.tensor(value_np).to(ctx.device)

    @staticmethod
    def backward(
            ctx, 
            in_grad):
        grads_np = ctx.sirf_obj.get_gradient(ctx.x).as_array()
        grads = torch.from_numpy(grads_np).to(ctx.device) * in_grad
        return grads.unsqueeze(dim=0), None, None, None

class ObjectiveFunctionModule3D(torch.nn.Module):
    def __init__(self, image_template, obj_fun):
        super().__init__()
        """ Objective function module for integration of SIRF and PyTorch

        Attributes:
            image_template: Template needed to 
            obj_fun  The locale where these birds congregate to reproduce.
        """
        self.image_template = image_template.clone()
        self.obj_fun = obj_fun
    def forward(self, out):
        """
        Objective function module for integration of SIRF and PyTorch

        Args:
            image_template: This is the first param.
            obj_fun: This is a second param.

        Returns:
            This is a description of what is returned.

        Raises:
            KeyError: Raises an exception.
        """
        obj_fun_value_batch = torch.zeros(1, device=out.device)
        for out_i in out:
            obj_fun_value = _objectiveFunctionModule3D.apply(
                out_i, self.image_template, self.obj_fun
                )
            obj_fun_value_batch = obj_fun_value_batch + obj_fun_value
        return obj_fun_value


class _objectiveFunctionModule2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, image_template, sirf_obj):
        ctx.sirf_obj = sirf_obj
        ctx.x = x
        ctx.image_template = image_template

        x_np = x.detach().cpu().numpy()
        x_np = ctx.image_template.fill(x_np)
        value_np = ctx.sirf_obj.get_value(x_np)
        value = torch.tensor(
            value_np).to(x.device)

        return value

    @staticmethod
    def backward(ctx, in_grad):

        grads_np = ctx.sirf_obj.get_gradient(
            ctx.image_template.fill(
                ctx.x.detach().cpu().numpy()
                )
            ).as_array()

        grads = torch.from_numpy(
            grads_np).to(in_grad.device
            ) * in_grad

        return grads, None, None, None

class ObjectiveFunctionModule2D(torch.nn.Module):
    def __init__(self, image_template, obj_fun):
        super().__init__()
        self.image_template = image_template.clone()
        self.obj_fun = obj_fun

    def forward(self, out):

        obj_fun_value_batch = torch.zeros(1, device=out.device)
        for out_i in out:
            obj_fun_value = _objectiveFunctionModule2D.apply(
                out_i, self.image_template, self.obj_fun
                )
            obj_fun_value_batch = obj_fun_value_batch + obj_fun_value
        return obj_fun_value



class _PETAcquisitionModelNumpyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, image_template, data_template, sirf_obj):
        ctx.sirf_obj = sirf_obj
        ctx.image_template = image_template
        ctx.data_template = data_template
        x_np = x.detach().cpu().numpy()
        x_np = ctx.image_template.fill(x_np[None])
        proj_data_np = ctx.sirf_obj.forward(x_np).as_array()
        proj_data = torch.from_numpy(proj_data_np).requires_grad_().to(x.device)
        return proj_data

    @staticmethod
    def backward(ctx, data):
        data_np = data.detach().cpu().numpy()
        data_np = ctx.data_template.fill(data_np)
        grads_np = ctx.sirf_obj.backward(data_np).as_array()
        grads = torch.from_numpy(grads_np).requires_grad_().to(data.device)
        return grads, None, None, None, None

class PETAcquisitionModelModule(torch.nn.Module):
    def __init__(self, image_template, data_template, acq_model):
        super().__init__()
        self.image_template = image_template.clone()
        self.data_template = data_template.clone()
        self.acq_model = acq_model

    def forward(self, image):
        # x.shape: (N, C, H, W) or (N, C, D, H, W)
        image_nc_flat = image.view(-1, *image.shape[-2:])
        acquired_data_nc_flat = []
        for x_i in image_nc_flat:
            sym_data_i = _PETAcquisitionModelNumpyFunction.apply(
                            x_i, 
                            self.image_template, 
                            self.data_template, 
                            self.acq_model)
            acquired_data_nc_flat.append(sym_data_i)
        acquired_data = torch.cat(acquired_data_nc_flat)
        return acquired_data