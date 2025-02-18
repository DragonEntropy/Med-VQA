import os
from abc import ABC, abstractmethod

class Model(ABC):
    cache_dir = os.path.join(os.getcwd(), "alex", "model_cache")

    def __init__(self, data_path, image_path):
        self.data_path = data_path
        self.image_path = image_path
        self.base_model = None
        self.processor = None

        self.examples = [(
            "What modality is used to take this image?",
            "xmlab102/source.jpg",
            """Let's think step by step.
1. The image is a medical image, which suggests that it is related to healthcare or anatomy.
2. The image is a cross-sectional view of the human body, which indicates that it is likely an X-ray or CT scan.
3. The image is black and white, which is a common characteristic of medical images.

Based on these observations, the modality used to take this image is most likely an X-ray or a CT scan.
So, the answer is a CT scan\n"""
        )]

        self.examples = {
            "Abnormality": [(
                "Where is/are the abnormality located?",
                "xmlab102/source.jpg",
                "Let's think step by step. 1. The image is a CT scan of the chest. 2. The abnormality appears as a dark area in the right lung field. 3. This dark area is not consistent with normal lung tissue and could indicate an abnormality such as a mass, consolidation, or other pathology. 4. The abnormality is located in the right lung field.\nSo, the answer is: The abnormality is located in the right lung field.\n"
            ),
            (
                "Which organ is abnormal, heart or lung?",
                "xmlab135/source.jpg",
                "Let's think step by step. 1. **Heart**: The heart appears to be slightly enlarged, which is indicated by the increased opacity and the blurring of the borders of the heart silhouette.2. **Lungs**: The lungs appear to be clear, with no visible signs of consolidation, fluid, or masses.Given these observations, the abnormal organ in this image is the **heart**.\nSo, the answer is: Heart\n"
            ),
            (
                "What diseases are included in the picture?",
                "xmlab105/source.jpg",
                "Let's think step by step. First, I need to identify the type of image. This is a CT scan image of the chest. Next, I should look for any abnormalities or unusual features in the image. In this image, I can see a large, dark area in the middle of the chest, which could be indicative of a mass or a lesion. Given the presence of a large, dark area, potential diseases that could be considered include:1. **Lung Cancer**: A large, dark area in the chest could be a sign of a lung mass, which is common in lung cancer.2. **Pneumonia**: Inflammation of the lung tissue can cause a large, dark area on a CT scan.3. **Pneumothorax**: A large, dark area could also indicate a pneumothorax, where air has entered the pleural space and compressed the lung.4. **Lymphoma**: Large masses in the chest can also be seen in lymphoma.\nSo, the answer is: The most likely disease that could be included in the picture is Lung Cancer.\n"
            )
            ],
            "Color": [(
                "What color is the lung in the picture?",
                "xmlab378/source.jpg",
                "Let's think step by step. 1. The image is a black-and-white X-ray.2. In X-ray images, the lungs appear as dark areas because they are filled with air, which is less dense than bone and other tissues.3. The dark areas in the image correspond to the lungs.Therefore, the lung in the picture is black.\nSo, the answer is black.\n"
            ),
            (
                "What color is the ears in the picture?",
                "xmlab405/source.jpg",
                "Let's think step by step. 1. The image is a CT scan of the head, which is a medical imaging technique.2. CT scans typically show different tissues and structures in grayscale.3. The ears are typically visible in CT scans as dark areas due to the air inside them.Given these points, the ears in the picture are likely to appear as dark areas.\nSo, the answer is dark.\n"
            ),
            (
                "What color is the right temporal lobe in the picture?",
                "xmlab401/source.jpg",
                "Let's think step by step. 1. The image is a grayscale CT scan of the head.2. The right temporal lobe is located on the right side of the image.3. In a CT scan, the brain tissue appears in varying shades of gray, with different intensities representing different densities.4. The right temporal lobe is typically darker than the surrounding structures due to its lower density.Therefore, the right temporal lobe in the picture is gray.\nSo, the answer is: The right temporal lobe in the picture is gray.\n"
            )],
            "KG": [(
                "Which organs/organ in the picture belong to the respiratory system?",
                "xmlab312/source.jpg",
                "Let's think step by step. First, we need to identify the organs in the image. The image is an X-ray of a human chest. The organs visible in this X-ray are:1. **Lungs**: The dark areas on either side of the image represent the lungs. The lungs are responsible for gas exchange, where oxygen is absorbed into the bloodstream and carbon dioxide is expelled.2. **Heart**: The heart is located in the center of the chest, slightly to the left. It is a muscular organ that pumps blood throughout the body.3. **Diaphragm**: The diaphragm is a muscle that separates the chest from the abdomen. It plays a crucial role in breathing by contracting and relaxing to allow air to enter and leave the lungs.These organs are all part of the respiratory system, which is responsible for the exchange of gases between the body and the environment.\nSo, the answer is: The lungs belong to the respiratory system.\n"
            ),
            (
                "What kind of symptoms will the disease on the upper right lung bring to the patient?",
                "xmlab344/source.jpg",
                "Let's think step by step. First, we need to identify the disease on the upper right lung in the image. The disease appears as a darker area in the upper right lung field. This could be indicative of several conditions, such as pneumonia, tuberculosis, or a lung abscess. Next, we need to consider the potential symptoms that these diseases might cause. Common symptoms of lung diseases include:1. Coughing, especially if it is productive (producing mucus or pus)2. Shortness of breath or difficulty breathing3. Chest pain or tightness4. Fever5. Fatigue6. Wheezing (if the disease involves the airways)Given the appearance of the disease in the image, the patient might experience symptoms such as coughing, shortness of breath, chest pain, and possibly fever. It's important to consult a healthcare professional for a proper diagnosis and treatment plan.\nSo, the answer is: The patient might experience coughing, shortness of breath, chest tightness, and possibly fatigue.\n"
            ),
            (
                "How to prevent the occurrence of the disease on the lower left lobe of lung in this image?",
                "xmlab311/source.jpg",
                "Let's think step by step. 1. The image shows a chest X-ray with a clear view of the lungs. The left lower lobe of the lung appears to have a darker area, which could indicate a potential issue such as a lung infection, pneumonia, or other lung conditions.2. To prevent the occurrence of the disease in the left lower lobe of the lung, it is important to maintain good overall health and hygiene.\nSo, the answer is: To prevent the occurrence of the disease in the left lower lobe of the lung, it is important to keep healthy.\n"
            )   
            ],
            "Modality": [(
                "What modality is used to take this image?",
                "xmlab102/source.jpg",
                "Let's think step by step. 1. The image is a cross-sectional view of the chest, which is characteristic of a computed tomography (CT) scan. 2. CT scans use X-rays to create detailed images of the body's internal structures. 3. The image shows clear, detailed structures such as bones, muscles, and air-filled spaces, which are typical features of a CT scan.Given these observations, the correct answer is:CT scan.\nSo, the answer is CT scan.\n"
            ),
            (
                "What modality is shown in this image?",
                "xmlab77/source.jpg",
                "Let's think step by step. The image shows a cross-sectional view of the brain, which is typically seen in MRI (Magnetic Resonance Imaging) scans. The smooth, detailed texture and the contrast between different brain structures are characteristic of MRI images. Therefore, the correct answer is:MRI.\nSo, the answer is MRI.\n"
            ),
            (
                "What modality is shown in this image?",
                "xmlab386/source.jpg",
                "Let's think step by step. 1. The image shows a clear view of the chest, including the ribs, spine, and lungs.2. The image is in black and white, which is characteristic of X-ray images.3. The presence of the ribs, spine, and lungs is typical of a chest X-ray.Given these observations, the modality shown in this image is a chest X-ray.\nSo, the answer is chest X-ray.\n"
            )],
            "Organ": [(
                "What is the main organ in the image?",
                "xmlab102/source.jpg",
                "Let's think step by step. The image is a cross-sectional view of a human chest, likely from a CT scan. The main organ visible in this image is the heart, located in the center of the image. The heart is surrounded by the lungs, which are visible on either side of the heart. The ribs and other structures are also visible around the heart and lungs.\nSo, the answer is: Heart.\n"
            ),
            (
                "Does the picture contain liver?",
                "xmlab103/source.jpg",
                "Let's think step by step. 1. The image is a cross-sectional view of the chest, not the abdomen. 2. The liver is typically located in the upper right quadrant of the abdomen. 3. The structures visible in this image include the heart, lungs, and parts of the chest wall, but there are no visible liver structures. Therefore, the picture does not contain the liver.\nSo, the answer is no.\n"
            ),
            (
                "What is the organ located in the lower right corner of the body in this image?",
                "xmlab208/source.jpg",
                "Let's think step by step. 1. The image is a cross-sectional view of the abdomen, likely from a CT scan.2. The organ located in the lower right corner of the body in this image is a kidney. The left kidney is typically located in the lower left quadrant of the abdomen, and it appears as a bean-shaped structure in this image.\nSo, the answer is: The organ located in the lower right corner of the body in this image is the left kidney.\n"
            )],
            "Plane": [(
                "What is the scanning plane of this image?",
                "xmlab219/source.jpg",
                "Let's think step by step. 1. The image shows a cross-sectional view of the chest, with the heart and major blood vessels visible in the center. 2. The ribs are visible on the sides, indicating that the image is a transverse (axial) scan. 3. The vertebral column is also visible at the bottom of the image, further confirming that this is a transverse scan.Therefore, the scanning plane of this image is transverse (axial).\nSo, the answer is transverse (axial).\n"
            ),
            (
                "What scanning plane does this image belong to?",
                "xmlab367/source.jpg",
                "Let's think step by step. 1. The image is a frontal view, as we can see the entire chest and shoulders. 2. The ribs and clavicles are clearly visible, which are typically seen in a frontal view. 3. The heart and lungs are also visible, which are also typically seen in a frontal view. 4. The image does not show any oblique or lateral features that would indicate a different scanning plane. Therefore, the image belongs to the coronal plane.\nSo, the answer is: The image belongs to the coronal plane.\n"
            ),
            (
                "Is this sagittal plane?",
                "xmlab329/source.jpg",
                "Let's think step by step. 1. The image is a chest X-ray, which typically shows the front view of the chest. 2. The X-ray shows the ribs, spine, and heart clearly, which are all visible in a frontal view. 3. The X-ray does not show any structures that would be visible in a sagittal plane, such as the spine or ribs from the side.Given these observations, the image is not in the sagittal plane.\nSo, the answer is no.\n"
            )],
            "Position": [(
                "Where is/are the abnormality located?",
                "xmlab102/source.jpg",
                "Let's think step by step. 1. The image is a CT scan of the chest. 2. The abnormality appears as a dark area in the right lung field. 3. This dark area is not consistent with normal lung tissue and could indicate an abnormality such as a mass, consolidation, or other pathology. 4. The abnormality is located in the right lung field.\nSo, the answer is: The abnormality is located in the right lung field.\n"
            ),
            (
                "Which part of the body does this image belong to?",
                "xmlab102/source.jpg",
                "Let's think step by step. This image is a cross-sectional view of the chest, which is typically obtained using a computed tomography (CT) scan. The image shows the following structures:1. **Lungs**: The dark areas on either side of the image represent the lungs.2. **Heart**: The bright, round structure in the center of the image represents the heart.3. **Ribs**: The white, curved structures surrounding the lungs represent the ribs.4. **Spine**: The central, dark, elongated structure running vertically in the middle of the image represents the spine.Given these observations, this image is of the thoracic region, specifically the chest. Therefore, the correct answer is:**The chest**\nSo, the answer is: The chest.\n"
            ),
            (
                "Where is the liver?",
                "xmlab208/source.jpg",
                "Let's think step by step. 1. The liver is located on the right side of the image, as it is typically positioned in the upper left quadrant of the abdomen. 2. The liver is a large, flat organ that is primarily responsible for detoxifying the blood and producing bile. 3. In this image, the liver appears as a large, dark area on the left side of the abdomen.\nSo, the answer is: The liver is on the left side of the body.\n"
            )],
            "Quantity": [(
                "How many kidneys in this image?",
                "xmlab208/source.jpg",
                "Let's think step by step. 1. **Identify the kidneys**: In a CT scan of the abdomen, the kidneys are typically visible as two bean-shaped structures located on either side of the spine.2. **Count the visible structures**: In the image provided, there are two distinct bean-shaped structures on either side of the spine.Therefore, there are **2 kidneys** in this image.\nSo, the answer is **2**.\n"
            ),
            (
                "How many organs are there in this image?",
                "xmlab219/source.jpg",
                "Let's think step by step. First, we need to identify the organs in the image. The image appears to be a cross-sectional view of the chest, likely from a CT scan. Here are the organs we can identify:1. **Lungs**: The dark areas on either side of the image represent the lungs.2. **Heart**: The bright, round structure in the center of the image represents the heart.3. **Diaphragm**: The curved structures at the bottom of the image represent the diaphragm.4. **Spine**: The white, curved structure in the center of the image represents the spine.So, there are at least 4 organs visible in this image.\nSo, the answer is 4.\n"
            ),
            (
                "How many lungs are shown in this image?",
                "xmlab299/source.jpg",
                "Let's think step by step. 1. The image is a chest X-ray.2. A chest X-ray typically shows both lungs.3. In this image, there are two distinct areas that represent the lungs.Therefore, the image shows two lungs.\nSo, the answer is 2."
            )],
            "Shape": [(
                "What is the shape of larynx in this image?",
                "xmlab427/source.jpg",
                "Let's think step by step. 1. The image is a cross-sectional view of the neck, likely from a CT scan.2. The larynx is the part of the neck that contains the vocal cords and is responsible for producing sound.3. In this image, the larynx appears as a dark, oval-shaped structure in the center of the image.4. The shape of the larynx in this image is consistent with the typical appearance of the larynx in a cross-sectional view.Therefore, the shape of the larynx in this image is an oval.\nSo, the answer is oval.\n"
            ),
            (
                "What is the shape of trachea about this patient?",
                "xmlab429/source.jpg",
                "Let's think step by step. 1. The image is a cross-sectional view of the neck, likely from a CT scan.2. The trachea is the airway that runs down the center of the neck.3. In this image, the trachea appears to be slightly curved, with a gentle bend in the middle.4. The curvature of the irregular indicating some abnormality.\nSo, the answer is: The trachea is slightly curved and has an irregular shape.\n"
            ),
            (
                "What is the shape of the kidney in the picture?",
                "xmlab252/source.jpg",
                "Let's think step by step. 1. The kidney is a bean-shaped organ located in the retroperitoneal space, behind the peritoneum. 2. The shape of the kidney is typically described as bean-shaped or bean-like. 3. In the image, the kidney appears to have an inflated oval shape, with a smooth, rounded contour. 4. Thus, the kidney has an irregular shape.\nSo, the answer is irregular shape.\n"
            )],
            "Size": [(
                "What is the largest organ in the picture?",
                "xmlab102/source.jpg",
                "Let's think step by step. The image is a CT scan of the chest. The largest organ in the picture is the lung. The lungs are the two large, spongy organs located in the chest cavity, occupying a significant portion of the thoracic cavity. They are responsible for the exchange of gases between the air we breathe and the bloodstream.\nSo, the answer is: lung.\n"
            ),
            (
                "Which is the biggest in this image,lung,liver or heart?",
                "xmlab103/source.jpg",
                "Let's think step by step. 1. The image is a cross-sectional view of the chest.2. The heart is centrally located and appears as a dark, round structure in the middle of the image.3. The lungs are visible on either side of the heart, filling the spaces around it.4. The liver is not visible in this image because it is located in the upper abdomen, not in the chest.Based on this analysis, the lung is the biggest structure in this image.\nSo, the answer is lung.\n"
            ),
            (
                "Which is smaller in this image,liver or right kidney?",
                "xmlab538/source.jpg",
                "Let's think step by step. 1. The image is a cross-sectional view of the abdomen, likely from a CT scan.2. The liver is typically located on the right side of the image, while the kidneys are on either side of the spine.3. The right kidney is usually larger than the left kidney due to the position of the liver.Given these points, the right kidney is smaller than the liver in this image.\nSo, the answer is: right kidney.\n"
            )]
        }

    @abstractmethod
    def provide_initial_prompts(self, data, examples=[], batch_size=1, direct=False, example_image=True, max=-1):
        yield None
    
    @abstractmethod
    def generate_final_prompts(self, outputs, example_image=True):
        return None
    
    @abstractmethod
    def run_model(self, prompts, images, store_attention=False):
        return None