
from openai import OpenAI
import base64
import time
class MLLMA3:
  def __init__(self,model_name="Qwen/Qwen3-VL-235B-A22B-Instruct"):
    self.client = OpenAI(
        base_url="https://api2.aigcbest.top/v1",
        api_key="sk-3eQDEhf5HTXmauGG44qGLrr47RF56K9LCToWaLVMd80w19eV"
    )
    self.model = model_name
    self.sys_prompt = """
    You are a visual-language expert specializing in multimodal understanding, logical reasoning, and detailed analysis. Your core task is to accurately analyze a Composed Image Retrieval (CIR) triplet, which consists of a "Reference Image," a "Target Image," and a "Modification Text," to determine if it exhibits a "Noisy Triplet Correspondence" (NTC) problem. Your judgment must be based on rigorous logical reasoning, and you must be able to clearly articulate your rationale.
   """
    self.prompt1 = """

**Input:** Reference Image (The first image), Target Image (The second image), Modification Text {mod}.

**Goal:** Classify this triplet as "Clean" or "Noisy" and output the result in JSON.

**Definitions:**


**Instructions:**
Analyze the images and text. Deconstruct their content, compare the differences, check if the text matches the visual difference, and apply the NTC definitions. 

**Required Output:**
{{
  "analysis_workflow": {{
    "full_analysis": "Include your observation, comparison, and reasoning here."
  }},
  "final_judgment": {{
    "verdict": "Clean | Noisy",
    "rationale": "The core reason."
  }}
}}
    """
  def encode_image(self,image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

  def decode_response(self,response):
    return response.choices[0].message.content
  def getLabels(self,img1,img2,mod):
      message_history = [
        {
          "role": "system",
          "content": self.sys_prompt
        },
          {
                "role": "user",
                "content": [
                    
                    {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{img1}"
                      }
                    },
                    {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{img2}"
                      }
                    },
                    {
                      "type": "text", 
                      "text": self.prompt1.format(mod=mod)
                    },
                ]
          }
      ]
      response = self.get_response(message_history)
      repsponse = response.lower()
      ans = 0
      if 'noisy' in repsponse:
        ans = 0
      elif 'clean' in repsponse:
        ans = 1
      return ans
  def get_response(self,message_history):
      response = self.client.chat.completions.create(
          model=self.model,
          messages=message_history,
          n=1
      )
      reply = response.choices[0].message.content
      return reply

class MLLMA2:
  def __init__(self,model_name="Qwen/Qwen3-VL-235B-A22B-Instruct"):
    self.client = OpenAI(
        base_url="https://api2.aigcbest.top/v1",
        api_key="sk-3eQDEhf5HTXmauGG44qGLrr47RF56K9LCToWaLVMd80w19eV"
    )
    self.model = model_name
    self.sys_prompt = """
    You are a visual-language expert specializing in multimodal understanding, logical reasoning, and detailed analysis. Your core task is to accurately analyze a Composed Image Retrieval (CIR) triplet, which consists of a "Reference Image," a "Target Image," and a "Modification Text," to determine if it exhibits a "Noisy Triplet Correspondence" (NTC) problem. Your judgment must be based on rigorous logical reasoning, and you must be able to clearly articulate your rationale.
    """
    self.prompt1 = """
    

  **Input:**
- Reference Image (The first image)
- Target Image (The second image)
- Modification Text {mod}

**Task:**
Directly analyze the logical coherence of this triplet.

**STRICT JUDGMENT CRITERIA (Must Follow):**

**1. The "Clean" Standard (Partial Match Principle):**
* **Definition:** A triplet is Clean if the **core visual change** described in the text actually occurred.
* **Crucial Tolerance:** Human annotations are imperfect. Ignore minor discrepancies (e.g., slight pose shifts, background noise, lighting changes). **Do not nitpick.**
* **Fashion Domain Note:** In fashion, if the text says "change color", and the item changes color but retains the same style, it is Clean. If the item structure changes completely (e.g., from a T-shirt to a dress) without text instruction, it is Noisy.

**2. The "Noisy" Standard (NTC Definitions):**
A triplet is Noisy ONLY IF there is a fundamental logical break. Classify into one of these three:
* **Type A: Mismatched Modification Text:** The actual visual change ($\Delta I$) is completely unrelated to the text instruction (e.g., Image: "Car turns blue", Text: "Add a chair").
* **Type B: Mismatched Reference Image:** The Ref image is irrelevant to the transformation logic (e.g., Text/Target refer to a "Church", but Ref is a "Forest").
* **Type C: Mismatched Target Image:** The Ref + Text create a clear expectation, but the Target is completely different.

**Instruction:**
Compare the images, read the text, and apply the criteria above. Generate a detailed **Reasoning Chain** explaining your decision process.
    """
    self.prompt2 = """
    **Task:**
Based on the reasoning you just provided, verify the conclusion and output the final verdict in the specified JSON format.

**Output Format:**
{
  "analysis_workflow": {
    "reasoning_summary": "Summarize your direct comparison logic here."
  },
  "final_judgment": {
    "verdict": "Clean | Noisy",
    "rationale": "Concise summary of the core reason."
  }
}
    """
  def encode_image(self,image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
  def decode_response(self,response):
    return response.choices[0].message.content
  def getLabels(self,img1,img2,mod):
      message_history = [
        {
          "role": "system",
          "content": self.sys_prompt
        },
          {
                "role": "user",
                "content": [
                    
                    {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{img1}"
                      }
                    },
                    {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{img2}"
                      }
                    },
                    {
                      "type": "text", 
                      "text": self.prompt1.format(mod=mod)
                    },
                ]
          }
      ]
      response = self.get_response(message_history)
      message_history.append({"role": "assistant", "content": response})
      message_history.append({"role": "user", "content": self.prompt2})
      response = self.get_response(message_history)
      repsponse = response.lower()
      ans = 0
      if 'noisy' in repsponse:
        ans = 0
      elif 'clean' in repsponse:
        ans = 1
      return ans
  def get_response(self,message_history):
      response = self.client.chat.completions.create(
          model=self.model,
          messages=message_history,
          n=1
      )
      reply = response.choices[0].message.content
      return reply

class MLLMA1:
  def __init__(self,model_name="Qwen/Qwen3-VL-235B-A22B-Instruct"):
    self.client = OpenAI(
        base_url="https://api2.aigcbest.top/v1",
        api_key="sk-3eQDEhf5HTXmauGG44qGLrr47RF56K9LCToWaLVMd80w19eV"
    )
    self.model = model_name
    self.sys_prompt = """
    You are a visual-language expert specializing in multimodal understanding, logical reasoning, and detailed analysis. Your core task is to accurately analyze a Composed Image Retrieval (CIR) triplet, which consists of a "Reference Image," a "Target Image," and a "Modification Text," to determine if it exhibits a "Noisy Triplet Correspondence" (NTC) problem. Your judgment must be based on rigorous logical reasoning, and you must be able to clearly articulate your rationale.
    """
    self.prompt1 = """
  **Input:**
- Reference Image (The first image)
- Target Image (The second image)
- Modification Text {mod}

**Task:**
You must analyze these three components **independently** and strictly. Do not compare them yet. Do not make assumptions. Just describe what you see.

1. **Analyze Reference Image:** Identify the main object, its attributes (color, shape, texture), the background, and the scene layout.
2. **Analyze Target Image:** Identify the main object, its attributes, background, and scene layout.
3. **Analyze Modification Text:** Analyze the text linguistically. What specific action is it requesting? (e.g., add object, change color, remove item, replace background).

**Output Format:**
Please provide your observations in the following structure:
- **[Fact-Ref]:** (Description of Reference Image)
- **[Fact-Tgt]:** (Description of Target Image)
- **[Fact-Txt]:** (Analysis of the Text Instruction)
    """
    self.prompt2 = """
    **Input:**
Using the facts extracted in the previous turn ([Fact-Ref], [Fact-Tgt], [Fact-Txt]).

**Task:**
Immediately classify the triplet based on these facts. **Do not output a reasoning chain.**

**DECISION RULES (Internalize these before judging):**

**RULE 1: The Tolerance Rule (For "Clean"):**
* Is the *primary* action in [Fact-Txt] visible in the change from [Fact-Ref] to [Fact-Tgt]?
* If YES, output "Clean" (even if there are other minor errors).
* *Example:* Text: "Change to red". Ref: Blue shirt. Tgt: Red shirt (but folded differently). -> **Verdict: Clean.**

**RULE 2: The NTC Rule (For "Noisy"):**
* Is there a total disconnect?
* *Example:* Text: "Change to red". Ref: Car. Tgt: Dog. -> **Verdict: Noisy.**

**Output Format:**
Directly output the JSON:
```json
{
  "analysis_workflow": {
    "step_1_observations": "Summary of facts",
  },
  "final_judgment": {
    "verdict": "Clean | Noisy",
    "rationale": "A single sentence summary based on Rule 1 or Rule 2."
  }
}
    """
  def encode_image(self,image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
  def decode_response(self,response):
    return response.choices[0].message.content
  def getLabels(self,img1,img2,mod):
      message_history = [
        {
          "role": "system",
          "content": self.sys_prompt
        },
          {
                "role": "user",
                "content": [
                    
                    {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{img1}"
                      }
                    },
                    {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{img2}"
                      }
                    },
                    {
                      "type": "text", 
                      "text": self.prompt1.format(mod=mod)
                    },
                ]
          }
      ]
      response = self.get_response(message_history)
      message_history.append({"role": "assistant", "content": response})
      message_history.append({"role": "user", "content": self.prompt2})
      response = self.get_response(message_history)

      repsponse = response.lower()
      ans = 1
      if 'noisy' in repsponse:
        ans = 0
      elif 'clean' in repsponse:
        ans = 1
      return ans
  def get_response(self,message_history):
      response = self.client.chat.completions.create(
          model=self.model,
          messages=message_history,
          n=1
      )
      reply = response.choices[0].message.content
      return reply
  

class MLLM:
  def __init__(self,model_name="Qwen/Qwen3-VL-235B-A22B-Instruct"):
    self.client = OpenAI(
        base_url="https://api2.aigcbest.top/v1",
        api_key="sk-3eQDEhf5HTXmauGG44qGLrr47RF56K9LCToWaLVMd80w19eV"
    )
    self.model = model_name
    self.sys_prompt = """
    You are a visual-language expert specializing in multimodal understanding, logical reasoning, and detailed analysis. Your core task is to accurately analyze a Composed Image Retrieval (CIR) triplet, which consists of a "Reference Image," a "Target Image," and a "Modification Text," to determine if it exhibits a "Noisy Triplet Correspondence" (NTC) problem. Your judgment must be based on rigorous logical reasoning, and you must be able to clearly articulate your rationale.
    Before beginning your analysis, you must understand the following core definitions:
    * **Standard CIR Triplet:**
    * **Reference Image:** Provides the initial visual content.
    * **Modification Text:** Describes how to modify the reference image to obtain the target image.
    * **Target Image:** Shows the final visual result after applying the modification text.

    * **Clean Triplet:**
      * A logically self-consistent triplet. The modification text accurately describes **at least one major, significant visual change** from the reference image to the target image.
      * **Crucial Principle:** Modification texts annotated by humans **are allowed to have incomplete information**. Annotators typically describe only the most central changes and may ignore minor or less obvious differences (e.g., a slight shift of a small object in the background, subtle changes in lighting conditions, minor variations in a person's pose, etc.). **As long as the core change described in the modification text genuinely occurred between the reference and target images, the triplet should be judged as "Clean," even if these unmentioned minor discrepancies exist.**

    * **Noisy Triplet / NTC:**
      * A triplet with a severe logical fracture or contradiction. The cause is usually that one element of the triplet is **completely unrelated** to the other two. This manifests in one of the following three scenarios:
          1.  **Mismatched Reference Image:** A clear logical connection exists between the modification text and the target image (e.g., the text describes features of the target image), but the reference image is entirely irrelevant to this transformation process.
          2.  **Mismatched Modification Text:** A clear, describable visual change exists between the reference and target images, but the provided modification text describes a completely different and unrelated operation.
          3.  **Mismatched Target Image:** The reference image and modification text together point to a clear, expected visual outcome, but the provided target image is completely inconsistent with this expectation.
                       """
    self.prompt1 = """
  **Input:**
  - Reference Image (The first image)
  - Target Image (The second image)
  - Modification Text {mod}

  **Task:**
  You must analyze these three components **independently** and strictly. Do not compare them yet. Do not make assumptions. Just describe what you see.

  1. **Analyze Reference Image:** Identify the main object, its attributes (color, shape, texture), the background, and the scene layout.
  2. **Analyze Target Image:** Identify the main object, its attributes, background, and scene layout.
  3. **Analyze Modification Text:** Identify the specific modification action linguistically. What specific action is it requesting? (e.g., add object, change color/texture/shape, remove item, replace background).

  **Output Format:**
  Please provide your observations in the following structure:
  - **[Fact-Ref]:** (Description of Reference Image)
  - **[Fact-Tgt]:** (Description of Target Image)
  - **[Fact-Txt]:** (Analysis of the Text Instruction)
    """
    self.prompt2 = """
  **Context:**
  Based on the objective descriptions provided in the previous turn (Fact-Ref, Fact-Tgt, Fact-Txt), your task is to perform a logical cross-validation to determine the validity of the triplet.

  **Core Principles (Crucial):**
  1. **Identify Actual Visual Differences:** First, ignore the text. Compare [Fact-Ref] and [Fact-Tgt] to determine what *actually* changed visually.
  2. **Allow Partial Matches (The "Clean" Standard):** Human annotations are imperfect. If the *core change* described in the text truly happened, the triplet is **"Clean"**, even if there are minor unmentioned differences (e.g., slight pose shift, background noise). Do not be nitpicky.
  3. **Identify NTC (The "Noisy" Standard):** A triplet is **"Noisy"** only if there is a fundamental logical break, falling into one of these categories:
    - *Mismatched Reference:* The Reference image is completely irrelevant to the Text/Target logic.
    - *Mismatched Text:* The Text describes an action totally unrelated to the actual visual change.
    - *Mismatched Target:* The Target image contradicts the Reference + Text expectation.

  **Task:**
  Generate a step-by-step **Reasoning Chain**:
  1. **Visual Diff Check:** What is the actual $\Delta$ between Ref and Target?
  2. **Semantic Alignment:** Does the [Fact-Txt] align with the Visual Diff?
  3. **Principle Application:** Is it a valid match, a partial match (acceptable), or a fundamental NTC failure? Diagnose the specific NTC type if noisy.

  **Output:**
  Provide your detailed reasoning logic text. Do not output JSON yet. Focus on the "Why".
    """
    self.prompt3 = """
  **Task:**
  Based on the detailed reasoning chain and analysis you just generated, summarize the findings and output the final verdict in the specified JSON format.

  **Requirements:**
  - **Verdict:** Must be strictly "Clean" or "Noisy".
  - **Rationale:** A concise 1-2 sentence summary of the core reason.
  - **Consistency:** Ensure the JSON content matches your previous reasoning perfectly.

  **Output Format:**
  {{
    "analysis_workflow": {{
      "step_1_reference_image_description": "Summarize observations from Step 1",
      "step_2_target_image_description": "Summarize observations from Step 1",
      "step_3_modification_text_analysis": "Summarize observations from Step 1",
      "step_4_synthesis_and_reasoning": "Paste or summarize the reasoning chain from Step 2, highlighting the NTC diagnosis or Partial Match logic."
    }},
    "final_judgment": {{
      "verdict": "Clean | Noisy",
      "rationale": "Concise summary of why."
    }}
  }}
    """
  def encode_image(self,image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
  def decode_response(self,response):
    return response.choices[0].message.content
  def getLabels(self,img1,img2,mod):
      message_history = [
        {
          "role": "system",
          "content": self.sys_prompt
        },
          {
                "role": "user",
                "content": [
                    
                    {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{img1}"
                      }
                    },
                    {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{img2}"
                      }
                    },
                    {
                      "type": "text", 
                      "text": self.prompt1.format(mod=mod)
                    },
                ]
          }
      ]
      response = self.get_response(message_history)
      message_history.append({"role": "assistant", "content": response})
      message_history.append({"role": "user", "content": self.prompt2})
      response = self.get_response(message_history)
      message_history.append({"role": "assistant", "content": response})
      message_history.append({"role": "user", "content": self.prompt3})
      response = self.get_response(message_history)

      repsponse = response.lower()
      ans = 0
      if 'noisy' in repsponse:
        ans = 0
      elif 'clean' in repsponse:
        ans = 1
      return ans
  
  def get_response(self,message_history):
      response = self.client.chat.completions.create(
          model=self.model,
          messages=message_history,
          n=1
      )
      reply = response.choices[0].message.content
      return reply


def getMLLM(class_name="MLLM",MLLM_name="MLLM"):
  if class_name == "MLLM":
    return MLLM(MLLM_name)
  if class_name == "MLLM-a1":
    return MLLMA1(MLLM_name)
  if class_name == "MLLM-a2":
    return MLLMA2(MLLM_name)
  if class_name == "MLLM-a3":
    return MLLMA3(MLLM_name)

if __name__ == "__main__":

  img_1 = "/root/autodl-tmp/cirr_data/CIRR/train/34/train-11041-2-img0.png"
  img_2 = "/root/autodl-tmp/cirr_data/CIRR/train/55/train-5703-0-img1.png"
  mod   = "Has a black dining carpet"
  gpt4o = MLLMA3("MLLM")
  img_1 = gpt4o.encode_image(img_1)
  img_2 = gpt4o.encode_image(img_2)
  ans = gpt4o.getLabels(img_1,img_2,mod)
  print(ans)
  