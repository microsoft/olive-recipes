# How to review changes to AITK folder

When user submits pr to add new model project or update existing model project, reviewer should pay attention to the following changes:

## Model Project Structure

The model project should have necessary files like olive configs, UX config for olive configs, README.md, info.yml etc.

### Requirements file

To convert the model, a **detailed** requirements file should be added or referenced. See more in [Create Requirements](./HowToCreateReq.md).

## E2E Test and Evaluation Result

In the README.md, the **evaluation result** should be included to show that the model could be successfully converted.

If UX config is not auto-generated, a screenshot for UX should also be provided in pr description.

## Configs update

When new models are added or there are changes to configuration, `.aitk/configs/checks.json` and `.aitk/configs.model_list.json` are also updated. Please make sure the changes match the changes in the pr.

## Version update

When updating existing model project, version may need to be updated. Refer to [Versioning](./Versioning.md) for more details.
