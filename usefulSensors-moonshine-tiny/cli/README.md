# Moonshine Optimization

This folder contains a sample use case of Olive to optimize a [UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny) model.

## How to run

### Pip requirements

Install the necessary python packages:

```bash
python -m pip install -r requirements.txt
```

### Optimize the model

```bash
olive optimize -m UsefulSensors/moonshine-tiny --exporter dynamo_exporter -t automatic-speech-recognition
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
