from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from torchvision import transforms
import argparse
from transformers import StoppingCriteriaList, TextIteratorStreamer

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

from PIL import Image
from torchattacks.attacks.pixle import *
from torchattacks.attacks.bim import *
from torchattacks.attacks.pgd_uap_v1 import *
from torchattacks.attacks.pgdl2 import *
import os

import torch.backends.cudnn as cudnn
import random




import csv
import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def add_prefix_to_png(image_path):
    base_name, ext = os.path.splitext(image_path)
    if ext.lower() == '.png':
        modified_name = base_name + 'adv' + ext
        return modified_name
    else:
        print(f"Warning: The file '{image_path}' does not have a '.png' extension.")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigpt4_llama2_eval.yaml', help="path to configuration file.")
    parser.add_argument("--class_tag", type=str, default="S1")
    parser.add_argument("--img_path", type=str, default="./images/1.png")
    parser.add_argument("--device_id", type=int, default="0")
    parser.add_argument("--attack_power", type=int, default="128")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args



    
args = parse_args() 
###########################PGD 强度############################

attack_mode = 'PGD'
attack_power = args.attack_power
attack_iters = 500
#############################################Experiment Config####################################

class_tag = args.class_tag
print(class_tag)

#==============================================================#
Output_log_file_path = "./Result_new2/" + class_tag + "/output_" + str(attack_power) + '_'+ os.path.basename(args.img_path) + ".log"
adv_img_save_path = add_prefix_to_png(args.img_path)

device_id = 'cuda:' + str(args.device_id)
device = torch.device(device_id)   # device for LLaMA Guard 2

print(args)
print(Output_log_file_path)
print(adv_img_save_path)





sys.stdout = Logger(Output_log_file_path, sys.stdout)



def save_image(image_array: np.ndarray, f_name: str) -> None:
    from PIL import Image
    image = Image.fromarray(image_array)
    image.save(f_name)



class MiniGPT(nn.Module):
    def __init__(self, class_tag):
        super(MiniGPT, self).__init__()

        # ========================================
        #             Model Initialization
        # ========================================

        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                     'pretrain_llama2': CONV_VISION_LLama2}
        # random_number = random.randint(1, 2000)
        # #random_number = 1216
        # random.seed(random_number)
        # np.random.seed(random_number)
        # torch.manual_seed(random_number)
        # print('Random seed 1: ', random_number)
        cudnn.benchmark = False
        cudnn.deterministic = True

        print('Initializing Chat')
        args = parse_args()
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.device_id
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(device)
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model = self.model.eval()
        
        CONV_VISION = conv_dict[model_config.model_type]
        self.device = device
        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(self.device) for ids in stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        print('Initialization Finished')

        conv = CONV_VISION.copy()
        

        #image = torch.load('./images/vis_processed_merlion_minigpt4_vicuna7B.pt')
        image = torch.load('./images/vis_processed_merlion_minigpt4_llama.pt')
        image = image.to(self.device)
        # print(image.shape)
        # image = torch.zeros(1,3,448,448).to(device)

        image_emb, _ = self.model.encode_img(image)
        image_list = []
        image_list.append(image_emb)

        self.train_prompt = []
        self.test_prompt = []
        self.train_target = []
        self.test_target = []
        
        
        if class_tag == "S1":
            self.train_goal_index = [9, 106, 208, 316, 334, 366, 383, 467, 493, 514, 518, 522, 525, 526, 531]
            self.test_prompt_index = [34, 48, 56, 66, 190, 255, 310, 336, 406, 409, 411, 469, 513, 519, 530, 533, 534, 536]
        elif class_tag == "S2":
            self.train_goal_index = [30, 32, 87, 117, 122, 151, 172, 191, 246, 268, 332, 364, 380, 398, 423]
            self.test_prompt_index = [3, 13, 29, 61, 73, 97, 118, 121, 125, 132, 133, 135, 136, 137, 141, 143, 148, 153, 165, 168, 170, 173, 182, 201, 223, 225, 227, 230, 234, 239, 269, 275, 288, 323, 324, 328, 341, 343, 344, 347, 358, 362, 365, 382, 386, 389, 395, 407, 417, 437, 455, 461, 462, 472, 478, 528]
        elif class_tag == "S3":
            self.train_goal_index = [2, 10, 15, 58, 102, 115, 119, 124, 258, 260, 265, 271, 292, 305, 322]
            self.test_prompt_index = [27, 39, 54, 64, 77, 80, 82, 86, 90, 91, 92, 100, 101, 105, 109, 110, 113, 128, 142, 145, 161, 164, 185, 194, 198, 199, 200, 202, 205, 206, 215, 221, 224, 228, 235, 245, 247, 263, 266, 276, 279, 281, 282, 285, 287, 291, 299, 307, 312, 335, 339, 340, 350, 355, 359, 367, 377, 420, 440, 441, 453, 463, 479, 482]
        elif class_tag == "S4":
            self.train_goal_index = [53, 98, 108, 213, 278, 319, 396, 494, 495, 496, 497, 498, 499, 501, 517]
            self.test_prompt_index = [7, 81, 140, 160, 189, 243, 289, 372, 385, 405, 451, 458, 500, 502, 503]
        elif class_tag == "S5":
            self.train_goal_index = [20, 56, 65, 106, 138, 255, 297, 406, 410, 514, 515, 517, 519, 521, 522]
            self.test_prompt_index = [11, 19, 48, 152, 163, 190, 310, 336, 366, 383, 443, 513, 516, 518, 520]
        elif class_tag == "S6":
            self.train_goal_index = [16, 20, 21, 23, 123, 131, 167, 373, 402, 411, 450, 524, 525, 526, 527]
            self.test_prompt_index = [22, 42, 47, 49, 54, 60, 192, 226, 270, 325, 353, 361, 422, 432, 523]
        elif class_tag == "S7":
            self.train_goal_index = [13, 136, 137, 168, 180, 227, 269, 287, 317, 330, 362, 382, 417, 437, 461]
            self.test_prompt_index =  [90, 151, 162, 173, 174, 182, 201, 223, 239, 345, 358, 386, 455, 478, 528]
        elif class_tag == "S8":
            self.train_goal_index = [0, 60, 254, 334, 342, 357, 384, 409, 426, 428, 529, 530, 532, 534, 536]
            self.test_prompt_index =  [9, 23, 49, 74, 163, 169, 189, 238, 325, 464, 467, 526, 531, 533, 535]

        self.train_num = len(self.train_goal_index)
        self.test_num = len(self.test_prompt_index)
        
        print("train_set: ", self.train_goal_index)
        print("test_set: ", self.test_prompt_index)
        
        rr = 0
        with open('./dataset/advbench/harmful_behaviors_2.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if rr in self.train_goal_index:
                    self.train_prompt.append(row['goal'])
                    self.train_target.append(row['target'])
                elif rr in self.test_prompt_index:
                    self.test_prompt.append(row['goal'])
                    self.test_target.append(row['target'])
                rr += 1
        self.q_conv = []
        for test_text in self.train_prompt:
            conv_tmp = conv.copy()
            if len(conv_tmp.messages) > 0 and conv_tmp.messages[-1][0] == conv_tmp.roles[0] \
                    and conv_tmp.messages[-1][1][-6:] == '</Img>':  # last message is image.
                conv_tmp.messages[-1][1] = ' '.join([conv_tmp.messages[-1][1], test_text +" <Img><ImageHere></Img>"])
            else:
                conv_tmp.append_message(conv_tmp.roles[0], test_text +" <Img><ImageHere></Img>")
            conv_tmp.append_message(conv_tmp.roles[1], None)
            self.q_conv.append(conv_tmp)

        self.test_conv = []
        for test_text in self.test_prompt:
            conv_tmp = conv.copy()
            if len(conv_tmp.messages) > 0 and conv_tmp.messages[-1][0] == conv_tmp.roles[0] \
                    and conv_tmp.messages[-1][1][-6:] == '</Img>':  # last message is image.
                conv_tmp.messages[-1][1] = ' '.join([conv_tmp.messages[-1][1], test_text +" <Img><ImageHere></Img>"])
            else:
                conv_tmp.append_message(conv_tmp.roles[0], test_text +" <Img><ImageHere></Img>")
            conv_tmp.append_message(conv_tmp.roles[1], None)

            self.test_conv.append(conv_tmp)

        self.conv = []
        self.target_len = []
        self.shift_labels = []
        for k in range(self.train_num):
            goal = self.train_prompt[k]
            target = self.train_target[k]

            conv_ = conv.copy()

            # print(text)
            if len(conv_.messages) > 0 and conv_.messages[-1][0] == conv_.roles[0] \
                    and conv_.messages[-1][1][-6:] == '</Img>':  # last message is image.
                conv_.messages[-1][1] = ' '.join([conv_.messages[-1][1], goal +" <Img><ImageHere></Img>"])
            else:
                conv_.append_message(conv_.roles[0], goal +" <Img><ImageHere></Img>")
            conv_.append_message(conv_.roles[1], target)
            self.conv.append(conv_)

            embs, inputs_tokens = self.get_context_emb(conv_, image_list, True)

            target_len_ = inputs_tokens.shape[1]
            self.target_len.append(target_len_)

            shift_labels_ = inputs_tokens[..., 1:].contiguous()
            self.shift_labels.append(shift_labels_)

    def get_context_emb(self, conv, img_list, flag):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')

        '''
        #llama-2
        if flag==True:
            #print(prompt_segs)
            prompt_segs[1] = prompt_segs[1][:-3]
        '''
        # print(prompt_segs)
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        # print('debug device: ', self.device)
        # print('debug model device: ', self.model.device)
        # print(seg_tokens)
        # print(seg_tokens[0].shape)
        # print(seg_tokens[1].shape)

        inputs_tokens = []
        inputs_tokens.append(seg_tokens[0])
        #inputs_tokens.append( torch.from_numpy(np.ones((1,32))*(-200)).to(device) ) #for 224*224 num_Vtokens=32
        inputs_tokens.append(torch.from_numpy(np.ones((1, 64)) * (-200)).to(self.device))  # for 448*448 num_Vtokens=256
        inputs_tokens.append(seg_tokens[1])

        dtype = inputs_tokens[0].dtype
        inputs_tokens = torch.cat(inputs_tokens, dim=1).to(dtype)
        # print(inputs_tokens)
        # print(inputs_tokens.shape)
        seg_embs = [self.model.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs, inputs_tokens

    def forward(self, inp):
        r"""
        Overridden.

        """
        images = inp[0]
        k = inp[1]

        image_emb, _ = self.model.encode_img(images)
        image_list = []
        image_list.append(image_emb)

        shift_logits = []

        loss_fct = nn.CrossEntropyLoss(ignore_index=-200)

        loss = 0
        if 1:
            conv_ = self.conv[k]
            target_len_ = self.target_len[k]
            shift_labels_ = self.shift_labels[k]

            embs, _ = self.get_context_emb(conv_, image_list, True)

            max_new_tokens = 300
            min_length = 1
            max_length = 2000

            current_max_len = embs.shape[1] + max_new_tokens
            if current_max_len - max_length > 0:
                print('Warning: The number of tokens in current conversation exceeds the max length. '
                      'The model will not see the contexts outside the range.')
            begin_idx = max(0, current_max_len - max_length)
            embs = embs[:, begin_idx:]

            outputs = self.model.llama_model(inputs_embeds=embs)
            logits = outputs.logits

            dtype = logits.dtype

            lm_logits = logits[:, :target_len_, :]

            # Shift so that tokens < n predict n
            shift_logits_ = lm_logits[..., :-1, :].contiguous()
            shift_logits.append(shift_logits_)

            loss += loss_fct(shift_logits_.view(-1, shift_logits_.size(-1)), shift_labels_.view(-1))

        return -loss



def denorm(image):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)

    image_denorm = image * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

    return image_denorm

def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = Guard_2.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

# Step 1: Create the model
random_number = random.randint(1, 2000)
setup_seed(random_number)
print("random_number:", random_number)

llama2_dict_emb = torch.load('./dataset/llama2_dict_embeddings.pt')
llama2_dict_emb = llama2_dict_emb.to(device)
model = MiniGPT(class_tag)
model = model.eval()


########################################################################################End init LLama Guard 2#######################################################

if attack_mode == 'PGD':
    attack = PGD(model, eps=attack_power / 255, alpha=1 / 255, steps=attack_iters, nprompt=model.train_num,
                 random_start=False)  # UAP need not rand_start, #universal noise
elif attack_mode == 'BIM':
    attack = BIM(model, eps=attack_power / 255, alpha=1 / 255, steps=attack_iters)

attack.set_mode_targeted_by_label()
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
attack.set_normalization_used(mean, std)

print("+++++++++++++++++++++++++++++++++++++++++++++++Generate train_prompt adv image+++++++++++++++++++++++++++++++++++++++++++")
#image = torch.load('./images/vis_processed_white_img_v2.pt')
# image = torch.zeros(1, 3, 224, 224).to(device)
# image[:, 0, :, :] = 128/255  # R
# image[:, 1, :, :] = 128/255  # G
# image[:, 2, :, :] = 128/255  # B
raw_image = Image.open(args.img_path).convert('RGB')
image = model.vis_processor(raw_image).unsqueeze(0).to(device)
# 保存为.png图像
#tensor_to_pil.save(random_init_save_path)

images = []
images.append(image)
adv_img = attack(images, model.shift_labels)

adv_image = denorm(adv_img[0])
save_img = (adv_image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
save_image(save_img, adv_img_save_path)

def change_extension(path: str, new_extension: str) -> str:
    base_name, ext = os.path.splitext(path)
    return f"{base_name}{new_extension}"

save_path = change_extension(adv_img_save_path, ".pt")

image_emb, _ = model.model.encode_img(adv_img[0])

torch.save(image_emb, save_path)