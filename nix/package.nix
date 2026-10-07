# The `asphodel` binary. ONNX Runtime is loaded at runtime (ort's
# `load-dynamic`), so the wrapper points `ORT_DYLIB_PATH` at nixpkgs'
# `onnxruntime`, as the dev shell does. `--set-default` lets the environment
# override it.
#
# Built with crane: the dependencies are their own derivation, keyed only on
# Cargo.toml and Cargo.lock, so a change to the workspace's code reuses them.
{
  lib,
  stdenv,
  craneLib,
  makeBinaryWrapper,
  onnxruntime,
  gitSha ? null,
}: let
  common = {
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

    strictDeps = true;
    cargoExtraArgs = "--locked --package asphodel";

    # CI runs `cargo test` in the dev shell. The scenarios the tests read sit
    # outside this source, and the build only has to produce the binary.
    doCheck = false;
  };

  cargoArtifacts = craneLib.buildDepsOnly (common
    // {
      # Only `cargo build` is cached; nothing here runs `cargo check`.
      cargoCheckCommand = "true";
    });
in
  craneLib.buildPackage (common
    // {
      inherit cargoArtifacts;

      nativeBuildInputs = [makeBinaryWrapper];

      # build.rs reads the SHA from git, which the nix source has no checkout
      # of. Replay reports embed it. Only the final build sees it, so a new
      # commit doesn't rebuild the dependencies.
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
    })
