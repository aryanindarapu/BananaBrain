from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch
from matplotlib import pyplot as plt
from matplotlib import patches
import cv2
import numpy as np
import io

# image = Image.open("data/images.jpeg")
og_img = cv2.imread("data/images.jpeg")
tmp_img = og_img.copy()
# print(og_img.shape)
image = cv2.cvtColor(og_img, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
# print(image.size)

image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# print(image.shape)
target_sizes = torch.tensor([image.size[::-1]])
# target_sizes = torch.tensor([image.shape[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
# fig, ax = plt.subplots()
# plt.imshow(image)
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]}, {label.item()} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
    
    cv2.rectangle(tmp_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    
    # draw boxes on image
    # p = patches.Rectangle(
    #     (box[0], box[1]),
    #     box[2] - box[0],
    #     box[3] - box[1],
    #     linewidth=1,
    #     edgecolor="r",
    #     facecolor="none",
    # )
    # ax.add_patch(p)

# cv2.imshow("stop sign detection", tmp_img)
print(tmp_img.shape)
cv2.imwrite("data/stop_sign_detection.jpeg", tmp_img)
# plt.show()
# plt.figure()
# fig.canvas.draw()
# # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
# img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
# print(fig.canvas.get_width_height(), image.size)
# img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
# plt.close(fig)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imshow("stop sign detection", img)
# cv2.imwrite("data/stop_sign_detection.jpeg", img)

# img_buf = io.BytesIO()
# plt.savefig(img_buf, format='png')
# # print(img_buf.getvalue())

# im = Image.open(img_buf)
# im.show()
# open_cv_image = np.array(im)
# open_cv_image = open_cv_image[:, :, ::-1].copy()

# cv2.imshow("stop sign detection", open_cv_image)

    
    