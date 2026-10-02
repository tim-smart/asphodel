# The `asphodel` binary. ONNX Runtime is loaded at runtime (ort's
# `load-dynamic`), so the wrapper points `ORT_DYLIB_PATH` at nixpkgs'
# `onnxruntime`, as the dev shell does. `--set-default` lets the environment
# override it.
{
  lib,
  stdenv,
  rustPlatform,
  makeBinaryWrapper,
  onnxruntime,
  gitSha ? null,
}:
rustPlatform.buildRustPackage {
  pname = "asphodel";
  version = (lib.importTOML ../Cargo.toml).workspace.package.version;

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Cargo.toml
      ../Cargo.lock
      ../clippy.toml
      ../crates
    ];
  };

  cargoLock.lockFile = ../Cargo.lock;
  cargoBuildFlags = ["--package" "asphodel"];

  nativeBuildInputs = [makeBinaryWrapper];

  # CI runs `cargo test` in the dev shell. The scenarios the tests read sit
  # outside this source, and the build only has to produce the binary.
  doCheck = false;

  # build.rs reads the SHA from git, which the nix source has no checkout
  # of. Replay reports embed it.
  env = lib.optionalAttrs (gitSha != null) {ASPHODEL_GIT_SHA = gitSha;};

  postInstall = ''
    wrapProgram $out/bin/asphodel \
      --set-default ORT_DYLIB_PATH ${onnxruntime}/lib/libonnxruntime${stdenv.hostPlatform.extensions.sharedLibrary}
  '';

  meta = {
    description = "Asphodel: brain-like memory for the Hermes agent";
    homepage = "https://github.com/tim-smart/asphodel";
    license = lib.licenses.mit;
    mainProgram = "asphodel";
  };
}
