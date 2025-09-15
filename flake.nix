{
  description = "NAM (.nam) → AIDA-X (.aidax): clean CMake build + minimal Python trainer (no pip/git at runtime)";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          # CPU-only? set cudaSupport=false. Keeping true lets PyTorch use CUDA if available.
          config = { allowUnfree = true; cudaSupport = true; };
        };

        # --- C++ tool: builds your CMake project producing the re-amp binary ---
        nam-reamp = pkgs.stdenv.mkDerivation {
          pname = "nam-reamp";
          version = "1.0.0";
          src = ./.;

          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs       = [ pkgs.libsndfile ];

          # CMake builder runs from its build dir; after build, the executable is ./NamReamp
          cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" ];

          installPhase = ''
            runHook preInstall
            mkdir -p $out/bin
            # current dir is the CMake build dir (thanks to cmake setup hook)
            install -m755 NamReamp $out/bin/nam-reamp
            runHook postInstall
          '';
        };

        # --- Python env for training/export (all declarative; no pip) ---
        pythonEnv = pkgs.python311.withPackages (ps: [
          ps.pytorch
          ps.numpy
          ps.scipy
          ps.librosa
          ps.soundfile
          ps.pandas
          ps.scikit-learn
          ps.tqdm
          ps.pyyaml
          ps.tensorboard
          ps.matplotlib
        ]);

        # --- One-shot app that stitches reamp → train → export ---
        nam2aidax = pkgs.writeShellApplication {
          name = "nam-to-aidax";
          runtimeInputs = [
            nam-reamp
            pythonEnv
            pkgs.coreutils
            pkgs.findutils
          ];
          text = ''
            set -euo pipefail

            usage() {
              cat >&2 <<'EOF'
Usage:
  nix run .#convert -- \
    --nam <model.nam> \
    --di <input_di.wav> \
    --trainer </path/to/Automated-GuitarAmpModelling> \
    [--out out.aidax] [--epochs 200] [--model-type {Lightest,Light,Standard,Heavy}] [--skip]

Notes:
  • No pip or git at runtime. Trainer must be a local checkout you point to with --trainer.
  • --di is the stimulus DI WAV that will be pushed through the NAM model.
EOF
              exit 2
            }

            nam="" di="" trainer="" out="out.aidax" epochs="200" model_type="Standard" skip=""

            while [ $# -gt 0 ]; do
              case "$1" in
                --nam) nam="$2"; shift 2;;
                --di) di="$2"; shift 2;;
                --trainer) trainer="$2"; shift 2;;
                --out) out="$2"; shift 2;;
                --epochs) epochs="$2"; shift 2;;
                --model-type) model_type="$2"; shift 2;;
                --skip) skip="--skip-connection"; shift 1;;
                -h|--help) usage;;
                *) echo "Unknown arg: $1" >&2; usage;;
              esac
            done

            [ -n "$nam" ] && [ -f "$nam" ] || { echo "Missing/invalid --nam"; usage; }
            [ -n "$di" ] && [ -f "$di" ]   || { echo "Missing/invalid --di"; usage; }
            [ -n "$trainer" ] && [ -d "$trainer" ] || { echo "Missing/invalid --trainer"; usage; }

            work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
            mkdir -p "$work/res" "$work/dataset"

            # 1) Prepare inputs
            ln -s "$(readlink -f "$nam" || realpath "$nam")" "$work/res/0.nam"
            cp -f "$(readlink -f "$di"  || realpath "$di")"  "$work/res/input.wav"

            # 2) Re-amp: NAM model → output.wav
            nam-reamp "$work/res/0.nam" "$work/res/input.wav" "$work/res/output.wav"

            # Trainer expects dataset/input.wav & dataset/output.wav
            cp -f "$work/res/input.wav"  "$work/dataset/input.wav"
            cp -f "$work/res/output.wav" "$work/dataset/output.wav"

            # 3) Train + export (no pip, no git). Uses your simplified train_min.py.
            python3 "${./python/train_min.py}" \
              --data-dir "$work/dataset" \
              --trainer  "$(readlink -f "$trainer" || realpath "$trainer")" \
              --epochs "$epochs" \
              --model-type "$model_type" \
              $skip \
              --out-dir "$work/res"

            aidax="$(find "$work/res" -maxdepth 1 -type f -name '*.aidax' | head -n1 || true)"
            [ -n "$aidax" ] || { echo "No .aidax produced; check logs above." >&2; exit 1; }

            cp -f "$aidax" "$out"
            echo "✅ wrote: $out"
          '';
        };
      in {
        # buildable things
        packages.default   = nam-reamp;
        packages.nam-reamp = nam-reamp;

        # runnable things
        apps = rec {
          default = { type = "app"; program = "${nam2aidax}/bin/nam-to-aidax"; };
          convert = default;
        };

        # optional dev shell (nice for hacking)
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.cmake pkgs.pkg-config pkgs.gcc pkgs.libsndfile pythonEnv ];
        };

        # optional formatter
        formatter = (pkgs.nixfmt-rfc-style or pkgs.nixfmt-classic);
      }
    );
}
