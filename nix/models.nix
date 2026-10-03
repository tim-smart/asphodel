# The model dir baked into the image, laid out as `asphodel models fetch`
# writes it (docs/models.md). Each file is a fixed-output fetch at the
# revision and SHA-256 the manifest in
# `crates/asphodel-core/src/models/manifest.rs` pins, so nothing downloads at
# runtime.
#
# The list mirrors the manifest. During an embedding model change the
# manifest carries both models, and so must this list. The build checks the
# result against the manifest compiled into the binary: `models fetch` skips
# every file already present with the right checksum, and any other file
# would need the network, which the build sandbox doesn't have.
{
  lib,
  fetchurl,
  runCommand,
  asphodel,
}: let
  # Where each of the manifest's files comes from in its repository.
  repoPaths = {
    "model.onnx" = "onnx/model_quantized.onnx";
    "tokenizer.json" = "tokenizer.json";
    "config.json" = "config.json";
    "special_tokens_map.json" = "special_tokens_map.json";
    "tokenizer_config.json" = "tokenizer_config.json";
  };

  models = [
    {
      # bge-small-en-v1.5:int8
      dir = "bge-small-en-v1.5-int8";
      repo = "Xenova/bge-small-en-v1.5";
      revision = "ea104dacec62c0de699686887e3f920caeb4f3e3";
      sha256 = {
        "model.onnx" = "6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4";
        "tokenizer.json" = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66";
        "config.json" = "fa73f90bf92c8cace1fbcb709626306f2bdbc9ea3e5b5f94b440df9b6aa56350";
        "special_tokens_map.json" = "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3";
        "tokenizer_config.json" = "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3";
      };
    }
    {
      # jina-reranker-v1-turbo-en:int8
      dir = "jina-reranker-v1-turbo-en-int8";
      repo = "jinaai/jina-reranker-v1-turbo-en";
      revision = "b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2";
      sha256 = {
        "model.onnx" = "3defdef1ae34e119bd704216087743e79665934c96aebabcb6077c239dc3ae66";
        "tokenizer.json" = "0046da43cc8c424b317f56b092b0512aaaa65c4f925d2f16af9d9eeb4d0ef902";
        "config.json" = "e050ff6a15ae9295e84882fa0e98051bd8754856cd5201395ebf00ce9f2d609b";
        "special_tokens_map.json" = "06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f";
        "tokenizer_config.json" = "d291c6652d96d56ffdbcf1ea19d9bae5ed79003f7648c627e725a619227ce8fa";
      };
    }
  ];

  link = model: name: let
    file = fetchurl {
      name = "${model.dir}-${name}";
      url = "https://huggingface.co/${model.repo}/resolve/${model.revision}/${repoPaths.${name}}";
      sha256 = model.sha256.${name};
    };
  in "ln -s ${file} $out/${model.dir}/${name}";

  install = model: ''
    mkdir -p $out/${model.dir}
    ${lib.concatMapStringsSep "\n" (link model) (lib.attrNames repoPaths)}
  '';
in
  runCommand "asphodel-models" {} ''
    ${lib.concatMapStringsSep "\n" install models}

    # Fails, wanting the network, unless every file the manifest lists is
    # here with its checksum.
    ${lib.getExe asphodel} models fetch --model-dir $out
  ''
