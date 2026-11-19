1. SceneSense

SceneSense is my image segmentation project. I made it to practice working with a pretrained deep learning model and building a simple Python application. The app lets me upload a photo and see which parts of the image belong to different categories, such as people, dogs, cats, trees, and grass etc.

2. Model

I used "MaskFormer ResNet-50 COCO-Stuff" from Hugging Face.

a. Model: "facebook/maskformer-resnet50-coco-stuff"
b. Classes: 171
c. Runs on: CPU
d. Interface: Gradio

I use the pretrained model directly, so I did not train a model or collect a dataset for this project.

3. What the app does

a. Upload a photo
b. Show a colored segmentation mask
c. Show the mask over the original photo
d. Adjust the overlay strength
e. List the detected categories and how much of the image each covers

4. Project files

hos015/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── segmentation/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── pipeline.py
│   ├── visualize.py
│   └── analytics.py
├── scripts/
│   ├── __init__.py
│   └── verify_model.py
└── tests/
    ├── __init__.py
    ├── test_analytics.py
    ├── test_pipeline.py
    └── test_visualize.py


5. How to run

Open a terminal in the "hos015" folder and install the required packages:

python -m pip install -r requirements.txt

Start the app:

python app.py

Open the local address shown in the terminal, usually "http://127.0.0.1:7860".

Upload a photo, adjust the overlay strength if needed, and click "Segment Image". The pretrained model weights will download automatically the first time the app runs.

6. Testing

I used pytest to check the image processing, mask creation, and category calculations.

Run the tests with:

python -m pytest -q


To check the model with an image, run:

python -m scripts.verify_model --image "test_images/dogs.png"

Replace the image path with a photo on your computer.

7. Limitations

The model does not always recognize every object correctly. The results also depend on the photo. It labels image regions by category but does not give each object its own separate ID. The percentages shown in the results describe how much of the image was assigned to each category. They are not accuracy scores.

8. Reference

MaskFormer pretrained model: https://huggingface.co/facebook/maskformer-resnet50-coco-stuff
