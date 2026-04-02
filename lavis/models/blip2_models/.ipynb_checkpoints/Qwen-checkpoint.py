from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch import Tensor

class Qwen():
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct/", torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct/")
    def getResult(
        self,
        img:[str,list],
        text:str="请问这两张图片的差异大吗，大的话输出1，小的话输出0，不需要输出其他任何东西."
    ):
        if isinstance(img, list):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img[0]},
                        {"type": "image", "image": img[1]},
                        {"type": "text", "text": text},
                    ],
                }
            ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
            )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

if __name__=="__main__":
    q = Qwen()
    image_path = ["/root/autodl-tmp/models/assets/p1.png","/root/autodl-tmp/models/assets/p2.png"]
    q.getResult(img = image_path,text="请问这两张图片的差异大吗，差异大的话就输出1，差异小的话就输出0，不需要输出其他任何东西")