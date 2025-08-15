# Design considerations

## Where data should go

- If the data is not related to a specific olive json
    - we want to update it on-the-fly: model_list.json / parameter_template.json
    - on-the-fly is not needed: model_lab.workspace.config / model_project.config
- If the data is related to olive json
    + on-the-fly is needed: *.json.config
    + on-the-fly not needed: model_project.config
