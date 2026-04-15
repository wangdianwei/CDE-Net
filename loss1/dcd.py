import torch

class DCDCosineLoss(nn.Module):
    def __init__(self):
        super(DCDCosineLoss, self).__init__()

    def rgb_to_lab(self, image):
        image_np = image.detach().cpu().numpy()
        # 确保 image_np 是四维的，添加批次维度如果需要
        if len(image_np.shape) == 3:
            image_np = np.expand_dims(image_np, axis=0)

        # 交换批次和通道维度
        image_np = image_np.transpose((0, 2, 3, 1))
        lab_image = rgb2lab(image_np)

        # 检查 lab_image 是否为空
        if lab_image is None:
            raise ValueError("LAB conversion failed.")

        # 交换回原始维度顺序，并添加批次维度
        lab_image = torch.from_numpy(lab_image).permute(0, 3, 1, 2).to(image.device)
        return lab_image

    def forward(self, input, target):
        lab_input = self.rgb_to_lab(input)
        lab_target = self.rgb_to_lab(target)

        a_input, b_input = lab_input[1], lab_input[2]
        a_target, b_target = lab_target[1], lab_target[2]

        dot_product = a_input * a_target + b_input * b_target
        norm_input = torch.sqrt(a_input ** 2 + b_input ** 2)
        norm_target = torch.sqrt(a_target ** 2 + b_target ** 2)

        cosine_distance = 1 - (dot_product / (norm_input * norm_target + 1e-8))

        dcd_loss = torch.norm(cosine_distance, p='fro')
        return dcd_loss