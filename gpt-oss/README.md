# GPT-OSS chat-template assets

`chat_template.patch` updates the exported GPT-OSS Harmony template shared by the 20B and 120B models. It renders all calls, ends intermediate call messages with message completion, and hands off after the last call. Parallel results must supply their function participant in `name`; the previous single-call name inference remains compatible. Visible pre-call content is commentary, separate from `thinking` and its existing retention policy. Completed turns retire earlier tool-call thinking whether the final message's `tool_calls` field is absent, empty, or null.

Apply this template-only patch after exporting the model. Do not modify ONNX weights, calibration scales, tokenizer token IDs, or EOS configuration. The expected original `chat_template.jinja` SHA-256 is `a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146`; the updated SHA-256 is `09f6ffeb72197c4166ab0d220a31de3e45951d6bfda560c61273172472793456`. Check the original hash before applying:

```sh
patch --batch --forward --fuzz=0 -p1 --directory /path/to/exported/model \
  --input /absolute/path/to/olive-recipes/gpt-oss/chat_template.patch
sha256sum /path/to/exported/model/chat_template.jinja
```

The internal projection fields `function.arguments_json`, tool-result `content_json`, and schema `default_json`/`enum_json` accept complete serialized JSON values instead of applying `tojson` again. An inference host using protected prompt fragments must substitute whole serialized slots and encode their restored bytes with an independent literal tokenizer. These fields are not additions to public tool schemas. Normal object arguments and result content still use JSON serialization when no slot is supplied.

The internal `function.native_freeform` flag declares a grammar-framed custom tool as `FREEFORM` and renders its historical `arguments_json` slot as raw text without JSON format metadata. The host supplies that slot as a whole protected raw body; function tools and unframed text custom tools keep JSON declarations and history. The host owns transport selection and preserves each generated call's historical encoding. Raw custom input is not unescaped or interpreted as JSON.

Rendering multiple calls is a transport capability, not a claim that the pretrained model generates parallel calls reliably. Keep tool handoff in the EOS set; never remove it to make sequential tool turns appear parallel.
