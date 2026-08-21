{
  description = "PyTorch reference environment for the pure Zig SAM 3 port";

  inputs.nixpkgs.url = "flake:nixpkgs";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      py = pkgs.python3.withPackages (ps: with ps; [
        torch          # CPU build from nixpkgs
        torchvision
        transformers   # provides the SAM 3 modelling code
        safetensors
        pillow
        numpy
        tokenizers
        huggingface-hub
      ]);
    in {
      packages.${system}.default = py;
      devShells.${system}.default = pkgs.mkShell {
        packages = [ py ];
        shellHook = ''
          export OMP_NUM_THREADS=''${OMP_NUM_THREADS:-8}
          export HF_HUB_OFFLINE=1
        '';
      };
    };
}
