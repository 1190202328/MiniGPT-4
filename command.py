import argparse
import json
import os

from tqdm import tqdm

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION


# imports modules for registration


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default='eval_configs/minigpt4_eval.yaml', help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def main():
    # ========================================
    #             Model Initialization
    # ========================================

    print('Initializing model')
    args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Model Initialization Finished')

    # upload image
    # 上传自己的图片
    domains = ['Art', 'Clipart', 'Product', 'Real World']

    # 选择domain
    domain = domains[3]

    msg = {
        'Art': ['Describe this painting in artistic language.', 'Translate what you just said into Chinese.'],
        'Clipart': ['Describe this clip art briefly and lovingly.'],
        'Product': ['Advertise the product in the picture.', 'Translate what you just said into Korean.'],
        'Real World': ['Describe this photo objectively and in detail.', 'Translate what you just said into Japanese.']
    }

    output_root_path = '/nfs/ofs-902-1/object-detection/jiangjing/datasets/office_home/office_home_text/'
    check_and_mkdir(output_root_path)

    root_path = '/nfs/ofs-902-1/object-detection/jiangjing/datasets/office_home/office_home/'
    domain_dir = f'{root_path}/{domain}'
    output_domain_dir = f'{output_root_path}/{domain}'
    check_and_mkdir(output_domain_dir)
    for _class in os.listdir(domain_dir):
        print(f'start to get domain [{domain}] class [{_class}].........')
        class_dir = f'{domain_dir}/{_class}'
        output_class_dir = f'{output_domain_dir}/{_class}'
        check_and_mkdir(output_class_dir)
        for image_name in tqdm(os.listdir(class_dir)):
            print(f'start to get domain [{domain}] class [{_class}] image [{image_name}]')
            image_path = f'{class_dir}/{image_name}'
            prefix, _ = os.path.splitext(image_name)
            output_text_path = f'{output_class_dir}/{prefix}.json'
            result_list = []

            # 初始化
            chat_state = CONV_VISION.copy()
            img_list = []
            chat.upload_img(image_path, chat_state, img_list)

            # ask first question
            user_message_0 = msg[domain][0]
            chat.ask(user_message_0, chat_state)

            # get answer
            llm_message_0 = chat.answer(conv=chat_state,
                                        img_list=img_list,
                                        num_beams=args.num_beams,
                                        temperature=args.temperature,
                                        max_new_tokens=300,
                                        max_length=2000)[0]
            print(llm_message_0)
            result_list.append(llm_message_0)

            if len(msg[domain]) == 2:
                # ask second question
                user_message_1 = msg[domain][1]
                chat.ask(user_message_1, chat_state)

                # get answer
                llm_message_1 = chat.answer(conv=chat_state,
                                            img_list=img_list,
                                            num_beams=args.num_beams,
                                            temperature=args.temperature,
                                            max_new_tokens=300,
                                            max_length=2000)[0]
                print(llm_message_1)
                result_list.append(llm_message_1)

            with open(output_text_path, mode='w', encoding='utf-8') as f:
                f.write(json.dumps(result_list))


def check_and_mkdir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == "__main__":
    main()
