#!/usr/bin/env python3
import argparse, os, shutil, subprocess, sys, torch, librosa

def prepare_data(data_dir: str) -> str:
    inp = os.path.join(data_dir, "input.wav")
    out = os.path.join(data_dir, "output.wav")
    i_aud, i_sr = librosa.load(inp, sr=None, mono=True)
    t_aud, t_sr = librosa.load(out, sr=None, mono=True)
    assert abs(len(t_aud)/t_sr - len(i_aud)/i_sr) < 3.0, "Input/Target lengths differ too much"
    assert i_sr == t_sr, "Sample rates differ"
    # Upstream helper to normalize/CSV if available:
    try:
        from colab_functions import prep_audio, create_csv_nam_v1_1_1
        create_csv_nam_v1_1_1(inp + ".csv")
        prep_audio([inp, out], file_name="amp", norm=True, csv_file=False)
    except Exception as e:
        print(f"[warn] prep helpers missing/failed ({e}); continuing with raw wavs.", file=sys.stderr)
    return "amp"

def train(trainer_dir: str, file_name: str, model_type: str, skip: bool, epochs: int):
    cfg = {"Lightest":"LSTM-8","Light":"LSTM-12","Standard":"LSTM-16","Heavy":"LSTM-20"}[model_type]
    sc = 1 if skip else 0
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    cmd = ["python3","dist_model_recnet.py","-l",cfg,"-fn",file_name,"-sc",str(sc),"-eps",str(epochs)]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=trainer_dir, env=env)
    return os.path.join(trainer_dir, "Results", f"{file_name}_{cfg}-{sc}")

def export_aidax(trainer_dir: str, model_dir: str, out_dir: str):
    best = os.path.join(model_dir, "model_best.json")
    # Convert to Keras JSON the way upstream does
    subprocess.check_call(["python3","modelToKeras.py","-lm",best], cwd=trainer_dir)
    keras_json = os.path.join(trainer_dir, "model_keras.json")
    os.makedirs(out_dir, exist_ok=True)
    aidax = os.path.join(out_dir, "model-lstm.aidax")
    shutil.copyfile(keras_json, aidax)
    # optional: also copy the best NAM json back
    shutil.copyfile(best, os.path.join(out_dir, "model-lstm.nam"))
    print(f"✅ wrote: {aidax}")

def main():
    p = argparse.ArgumentParser(description="Minimal NAM→AIDA-X trainer wrapper")
    p.add_argument("--data-dir", required=True, help="folder with input.wav & output.wav")
    p.add_argument("--trainer",  required=True, help="path to Automated-GuitarAmpModelling checkout")
    p.add_argument("--model-type", default="Standard", choices=["Lightest","Light","Standard","Heavy"])
    p.add_argument("--skip-connection", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--out-dir", default="res")
    args = p.parse_args()

    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    fn = prepare_data(args.data_dir)
    # ensure trainer has helper scripts available from its CWD
    # copy our train_min into trainer root as train.py so it can import their modules if needed
    # (not strictly required anymore, we run everything from here)
    model_dir = train(args.trainer, fn, args.model_type, args.skip_connection, args.epochs)
    export_aidax(args.trainer, model_dir, args.out_dir)

if __name__ == "__main__":
    main()
