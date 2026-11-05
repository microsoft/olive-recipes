# GPT-OSS chat-template assets

`chat_template.patch` updates the exported GPT-OSS Harmony template shared by the 20B and 120B models. It renders all calls, ends intermediate call messages with message completion, and hands off after the last call. Parallel results must supply their function participant in `name`; the previous single-call name inference remains compatible. Visible pre-call content is commentary, separate from `thinking` and its existing retention policy.

Apply this template-only patch after exporting the model. Do not modify ONNX weights, calibration scales, tokenizer token IDs, or EOS configuration. The expected original `chat_template.jinja` SHA-256 is `a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146`; the updated SHA-256 is `df7d4486e2004d0e36e2493828ec0c875008fdba7be9200eed2d4e1459ce9110`. Check the original hash before applying:

```sh
patch --batch --forward --fuzz=0 -p1 --directory /path/to/exported/model \
  --input /absolute/path/to/olive-recipes/gpt-oss/chat_template.patch
sha256sum /path/to/exported/model/chat_template.jinja
```

The internal projection fields `function.arguments_json`, tool-result `content_json`, and schema `default_json`/`enum_json` accept complete serialized JSON values instead of applying `tojson` again. An inference host using protected prompt fragments must substitute whole serialized slots and encode their restored bytes with an independent literal tokenizer. These fields are not additions to public tool schemas. Normal object arguments and result content still use JSON serialization when no slot is supplied.

Rendering multiple calls is a transport capability, not a claim that the pretrained model generates parallel calls reliably. Keep tool handoff in the EOS set; never remove it to make sequential tool turns appear parallel.
