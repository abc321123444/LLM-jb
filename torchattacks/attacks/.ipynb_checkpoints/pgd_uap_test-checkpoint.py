import torch
import torch.nn as nn
from torchvision import transforms
from ..attack import Attack


def my_norm(image):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    normalize = transforms.Normalize(mean, std)
    image_norm = normalize(image)

    return image_norm


def test_trainset(model, adv_img):
    
    image_emb, _ = model.model.encode_img(adv_img)  # NOBUG
    image_list = []
    image_list.append(image_emb)
    
    for p in range(model.train_num):
        print('-------------------------Response Goal ' + str(model.train_goal_index[p] + 1) + ':' + model.mprompt[p] + '----------------------------------------')

        q_conv = model.q_conv[p]
        embs, _ = model.get_context_emb(q_conv, image_list, False)

        # generation_dict = self.answer_prepare(conv, img_list, **kargs)
        max_new_tokens = 500
        min_length = 1
        max_length = 2000

        num_beams = 1
        top_p = 0.9
        repetition_penalty = 1.05
        length_penalty = 1
        temperature = 1.0

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        generation_dict = dict(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
        )

        output_token = model.model.llama_model.generate(**generation_dict)[0]
        output_text = model.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
        #jailbroken = not any([prefix in output_text[:100] for prefix in _test_prefixes])
        #em = model.answers[p] in output_text
        print(output_text)  # output_token.cpu().numpy()

def test_testset(model, adv_img):
    
    image_emb, _ = model.model.encode_img(adv_img)  # NOBUG
    image_list = []
    image_list.append(image_emb)
    
    for p in range(model.test_num):
        print('-------------------------Response Goal ' + str(model.test_goal_index[p] + 1) + ':' + model.test_goal[
            p] + '----------------------------------------')

        test_conv = model.test_conv[p]

        embs, _ = model.get_context_emb(test_conv, image_list, False)

        # generation_dict = self.answer_prepare(conv, img_list, **kargs)
        max_new_tokens = 500
        min_length = 1
        max_length = 2000

        num_beams = 1
        top_p = 0.9
        repetition_penalty = 1.05
        length_penalty = 1
        temperature = 1.0

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        generation_dict = dict(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
        )

        output_token = model.model.llama_model.generate(**generation_dict)[0]
        output_text = model.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
        print(output_text)  # output_token.cpu().numpy()
    


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, nprompt=1, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.nprompt = nprompt
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter('./logs_syy/')

    def forward(self, images, labels, step):
        r"""
        Overridden.
        """
        #print(images)
        #print(torch.min(images))

        #images = images.clone().detach().to(self.device)
        images_ = []
        adv_images_ = []
        for image in images:
            image = image.clone().detach().to(self.device)
            adv_image = image.clone().detach().to(self.device)
            images_.append(image)
            adv_images_.append(adv_image)


        if self.targeted:
            target_labels = labels

        #loss = nn.CrossEntropyLoss()
        loss = nn.CrossEntropyLoss(ignore_index=-200)

        #adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        st = 0
        universal = 1 #universal noise
        if universal == 1:
            noise = torch.zeros(1, 3, 224, 224).to(self.device)
            #print(noise)


        for i in range(self.steps):
            #print('step: '+str(_))
            cost_step = 0
            for k in range(len(images_)):
                image = images_[k]

                image_ = image.clone()
                for p in range(self.nprompt):

                    image_.requires_grad = True
                    inp = []

                    if universal == 1:
                        adv_image = image_ + noise  #universal noise

                    inp.append(adv_image)
                    inp.append(p)
                    cost = self.get_logits(inp)
                    # Update adversarial images
                    grad = torch.autograd.grad(
                        cost, adv_image, retain_graph=False, create_graph=False
                    )[0]


                    cost_step += cost.clone().detach()

                    adv_image = adv_image.detach() + self.alpha * grad.sign()
                    delta = torch.clamp(adv_image - image, min=-self.eps, max=self.eps)
                    adv_image = torch.clamp(image + delta, min=0, max=1).detach()

                    if universal == 1:
                        noise = adv_image - image #universal noise
                    '''
                    if 1:
                        for p in range(self.nprompt):
                            inp = []
                            inp.append(adv_image)
                            inp.append(p)
                            cst = self.get_logits(inp)
                            name = "loss" + str()
                            self.writer.add_scalar(name,cst,st)
                            
                            
                            # Update adversarial images
                    st += 1
                    print('iter: {}'.format(st))
                    '''
                            
                    if universal == 1:
                        noise = adv_image - image #universal noise

            print('step: {}: {}'.format(i, cost_step))
            #print(adv_image)
            
            #if i+1 in step:
                #print(adv_image)
                #adv_img = my_norm(adv_image)
                #test_trainset(self.model, adv_img)
                #if i+1 == 500:
                    #test_testset(self.model, adv_img)
                


        if universal == 1: #universal noise
            images_outputs_ = []
            delta = torch.clamp(noise, min=-self.eps, max=self.eps)
            for k in range(len(images_)):
                adv_image = torch.clamp(images_[k] + delta, min=0, max=1).detach()
                images_outputs_.append(adv_image.detach())
                
            #print("in atk images")
            #print(images_outputs_)

            return images_outputs_





