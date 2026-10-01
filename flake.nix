{
  description = "Asphodel: brain-like memory for the Hermes agent";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs = {nixpkgs, ...}: let
    forAllSystems = function:
      nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed
      (system: function nixpkgs.legacyPackages.${system});
  in {
    devShells = forAllSystems (pkgs: let
      python = pkgs.python3.withPackages (ps: [ps.pytest]);
    in {
      default = pkgs.mkShell {
        packages = [
          pkgs.cargo
          pkgs.rustc
          pkgs.clippy
          pkgs.rustfmt
          pkgs.rust-analyzer
          pkgs.pkg-config
          pkgs.openssl
          pkgs.sqlite
          pkgs.onnxruntime
          python
        ];

        RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
        # fastembed's ONNX Runtime is loaded from nixpkgs rather than downloaded at build time.
        ORT_DYLIB_PATH = "${pkgs.onnxruntime}/lib/libonnxruntime${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}";
      };
    });

    formatter = forAllSystems (pkgs: pkgs.alejandra);
  };
}
