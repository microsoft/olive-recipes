import torch
def llm_concat_forecast_embedding(inputs_embeds, path_to_ssd_pt):
    '''
    This API concatenate inputs_embeds and SSD forecast_embeddings
    inputs_embeds | forecast_embeddings

    params:
    inputs_embeds: inputs_embeds is the torch.nn.Embedding layer
    path_to_ssd_pt: the path where the SSD forecast embeddings are stored
    '''
    ssd_params= torch.load(path_to_ssd_pt)
    forecast_embeddings = ssd_params['forecast_embedding'].to(device = inputs_embeds.weight.device, dtype = inputs_embeds.weight.dtype)
    # inputs_embeds.weight is of type nn.Parameter, hence we concat with .data and assign to .data which is a tensor, otherwise we cannot assign a tensor to a nn.Pramater object,
    inputs_embeds.weight.data = torch.cat([inputs_embeds.weight.data, forecast_embeddings], dim=0)

    return inputs_embeds