# Simple PyTorch Model Pipeline

This is a simple PyTorch model pipeline that can be used to train and evaluate a model. The pipeline is designed to be used with a simple feedforward neural network, but can be easily adapted to other models.

This is for experimental around model provenance and integrity.

A simple github action is used to run the pipeline.

## Usage

Set up a virtualenv and install the requirements:

```bash
virtualenv -p python3 venv

source venv/bin/activate

pip install -r requirements.txt
```

Generate the training set:

```bash
python generate_dataset.py
```

Train the model:

```bash
python train_model.py
```

Inference:

```bash
python run_inference.py
```

