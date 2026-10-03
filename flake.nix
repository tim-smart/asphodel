{
  description = "Asphodel: brain-like memory for the Hermes agent";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    inherit (nixpkgs) lib;
    forAllSystems = function:
      lib.genAttrs lib.systems.flakeExposed
      (system: function nixpkgs.legacyPackages.${system});
  in {
    packages = forAllSystems (pkgs: let
      asphodel = pkgs.callPackage ./nix/package.nix {
        gitSha = self.rev or null;
      };
      models = pkgs.callPackage ./nix/models.nix {inherit asphodel;};
    in
      {
        inherit asphodel models;
        default = asphodel;
      }
      // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
        # `nix build .#image`, then `docker load < result`.
        image = pkgs.callPackage ./nix/image.nix {inherit asphodel models;};
      });

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
          pkgs.nodejs
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
