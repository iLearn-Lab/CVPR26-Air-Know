
from openai import OpenAI
import base64
import time

class GPT4o:
  def __init__(self,model_name="gpt-5-mini-2025-08-07"):
    self.client = OpenAI(
        base_url="https://api2.aigcbest.top/v1",
        api_key="sk-3eQDEhf5HTXmauGG44qGLrr47RF56K9LCToWaLVMd80w19eV"
    )
    self.model = model_name
    self.sys_prompt = """
You are a visual-language expert specializing in multimodal understanding, logical reasoning, and detailed analysis. Your core task is to accurately analyze a Composed Image Retrieval (CIR) triplet, which consists of a "Reference Image," a "Target Image," and a "Modification Text," to determine if it exhibits a "Noisy Triplet Correspondence" (NTC) problem. Your judgment must be based on rigorous logical reasoning, and you must be able to clearly articulate your rationale.
                       """
    self.prompt1 = """
            You will be provided with a CIR triplet containing a reference image (the first image), a target image(the second image), and a modification text "{mod}". Please follow the definitions, principles, and steps below to analyze the triplet and output your final judgment and analysis process in the specified JSON format.

****1. Core Concept Definitions****

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

****2. Important Considerations & Common Pitfalls****

To ensure the accuracy of your judgment, please pay close attention to the following points:

* **Focus on Core Contradictions:** Your task is to identify fundamental logical breaks within the triplet, not to nitpick minor imperfections. Always operate under the guiding principle that "the annotation process allows for the omission of minor differences."
* **The Challenge of Fashion Domains:** Be particularly careful when handling fashion datasets (e.g., FashionIQ). Because items within a category (like t-shirts or dresses) can be visually very similar, a randomly swapped-in reference image might still look like a plausible "starting point." In such cases, you must more finely assess the **plausibility of the transformation**. For example, if the reference is a blue t-shirt, the target is a black t-shirt with a complex print, and the text is "change to black and add a print," it is clean. However, if the text is "make the sleeves shorter," but the reference and target are two garments with entirely different designs, cuts, and colors, it is highly likely a **noisy** triplet, even if both are t-shirts.
* **Avoid Over-Imagination:** Do not make inferences based on vague similarities. The hallmark of a noisy triplet is **"obvious irrelevance."** If it requires complex, multi-step creative thinking to connect the three elements, it is likely noisy.

****3. Structured Analysis Workflow****

Please strictly adhere to the following five steps in your thinking and analysis, and reflect this process in your final output:

* **Step 1: Deconstruct the Reference Image**
    * Identify and briefly describe the main objects, attributes, scene, and their relationships in the reference image.

* **Step 2: Deconstruct the Target Image**
    * Identify and briefly describe the main objects, attributes, scene, and their relationships in the target image.

* **Step 3: Deconstruct the Modification Text**
    * Accurately understand the instruction described in the modification text. Identify the modification actions it specifies (e.g., add, remove, replace, change color/texture/shape, etc.).

* **Step 4: Synthesize, Reason & Cross-Validate**
    1.  **Identify Actual Visual Differences:** By comparing the results of Step 1 and Step 2, objectively describe the **actual visual changes** (both major and minor) that occurred from the reference image to the target image.
    2.  **Validate Text-to-Difference Match:** Determine if the operation described in the modification text (from Step 3) aligns with the **major visual differences** identified in Step 4.1.
    3.  **Apply NTC Definitions for Judgment:**
        * If the text matches the major differences, judge the triplet as "Clean."
        * If it does not match, analyze further to determine which NTC scenario applies. For instance, if the actual change between images is A, but the text describes change B, it is a "Mismatched Modification Text." Or, if the text and target image describe state C, but the reference image is state D, it is a "Mismatched Reference Image."

* **Step 5: Formulate the Final Conclusion**
    * Based on the analysis above, provide a final verdict of "Clean" or "Noisy."
    * Clearly and concisely summarize the core reason for this judgment in one to two sentences.

****4. Examples****

Here are some examples to help you understand the judgment criteria:

* **Example 1: Typical Clean Triplet**
    * **Reference Image:** A yellow dog on the grass.
    * **Modification Text:** "Change the dog to a black cat."
    * **Target Image:** A black cat on the grass.
    * **Analysis:** The text accurately describes the change of the main subject from a "yellow dog" to a "black cat," with the background remaining consistent. This is **Clean**.

* **Example 2: Clean Triplet with Acceptable Minor Differences**
    * **Reference Image:** A man in a blue long-sleeved shirt, standing indoors.
    * **Modification Text:** "Make the shirt red."
    * **Target Image:** The same man in a red long-sleeved shirt, with a slightly different pose, and a book in the background has been removed.
    * **Analysis:** The core modification "shirt becomes red" has been accurately executed. The changes in pose and the background book are negligible minor differences. This is **Clean**.

* **Example 3: Noisy Triplet (Mismatched Modification Text)**
    * **Reference Image:** A red sedan parked on the street.
    * **Modification Text:** "Add an airplane to the sky."
    * **Target Image:** A blue sedan parked on the street.
    * **Analysis:** The actual change in the images is the car's color from red to blue, whereas the text describes a completely unrelated operation of "adding an airplane." This is **Noisy**.

* **Example 4: Noisy Triplet (Mismatched Reference Image)**
    * **Reference Image:** A landscape photo of a forest.
    * **Modification Text:** "Change the color of the steeple from red to gold."
    * **Target Image:** A church with a golden steeple.
    * **Analysis:** The modification text and the target image together describe a scene involving a "church steeple," but the reference image is a "forest," which is completely unrelated to the transformation process. This is **Noisy**.

****5. Required Output Format****

You must return your analysis in the following JSON format(Avoid duplicate output):

```json
{{
  analysis_workflow": {{
    "step_1_reference_image_description": "A brief description of the reference image's content.",
    "step_2_target_image_description": "A brief description of the target image's content.",
    "step_3_modification_text_analysis": "An analysis of the instruction in the modification text.",
    "step_4_synthesis_and_reasoning": "The detailed synthesis and reasoning process, including identification of actual differences, text-to-difference validation, and the logic for applying the NTC definitions."
  }},
  "final_judgment": {{
    "verdict": "Clean | Noisy",
    "rationale": "A concise summary of the core reason for the verdict."
  }}
}}
    """
  def encode_image(self,image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
  def decode_response(self,response):
    return response.choices[0].message.content
  def getLabels(self,r_img,t_img,text):
    response = self.client.chat.completions.create(
      # "claude-sonnet-4-5-20250929"
      model=self.model,
      messages=[
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
                          "url": f"data:image/jpeg;base64,{r_img}"
                      }
                    },
                    {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{t_img}"
                      }
                    },
                    {
                      "type": "text", 
                      "text": self.prompt1.format(mod=text)
                    },
                ]
            }
      ]
    )
    response = self.decode_response(response)
    repsponse = response.lower()
    ans = 0
    if 'noisy' in repsponse:
      ans = 0
    if 'clean' in repsponse:
      ans = 1
    return ans


if __name__ == "__main__":
  gpt4o = GPT4o()
  img_1 = "/root/autodl-tmp/cirr_data/CIRR/train/34/train-11041-2-img0.png"
  img_2 = "/root/autodl-tmp/cirr_data/CIRR/train/55/train-5703-0-img1.png"
  mod   = "Has a black dining carpet"
  a = time.time()
  img_1 = gpt4o.encode_image(img_1)
  img_2 = gpt4o.encode_image(img_2)
  c = gpt4o.getLabels(img_1,img_2,mod)
  print(c)
  b = time.time()
  print(b-a)