{
  description = "Python Various";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.devshell.url = "github:numtide/devshell";

  outputs = { self, nixpkgs, flake-utils, devshell }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ devshell.overlay ];
        };

        myPython = pkgs.python310.withPackages (p: with p; [
          # Basic
          rich

          numpy
          matplotlib
        ]);
      in {
        devShell = pkgs.devshell.mkShell {
          name = "Python Various";

          packages = with pkgs; [
            myPython
            nodePackages.pyright # LSP
            # jetbrains.pycharm-professional
          ];

          # Use $1 for positional args
          commands = [
            # {
            #   name = "ide";
            #   help = "Run pycharm for this project";
            #   command = "pycharm . &>/dev/null &";
            # }
          ];
        };
      });
}
