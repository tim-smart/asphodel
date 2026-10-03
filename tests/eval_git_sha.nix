# Run with: nix eval --json --file tests/eval_git_sha.nix
let
  flake = import ../flake.nix;
  lib = {
    systems.flakeExposed = [ "test" ];
    genAttrs = names: f: builtins.listToAttrs (map (name: { inherit name; value = f name; }) names);
    optionalAttrs = condition: attrs: if condition then attrs else {};
  };
  pkgs = {
    callPackage = path: args: args;
    stdenv.hostPlatform.isLinux = false;
  };
  gitSha = self: (flake.outputs {
    inherit self;
    nixpkgs = { inherit lib; legacyPackages.test = pkgs; };
  }).packages.test.asphodel.gitSha;
in
assert gitSha { rev = "clean-revision"; dirtyRev = "dirty-revision"; } == "clean-revision";
assert gitSha { dirtyRev = "dirty-revision"; } == "dirty-revision";
assert gitSha {} == null;
true
